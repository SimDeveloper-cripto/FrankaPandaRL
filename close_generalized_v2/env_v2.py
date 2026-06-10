#!/usr/bin/env python3
# close_generalized_v2/env_v2.py
#
# AdvancedGeneralizedDoorEnv — Top-Level Environment
#
# Integrates all v2.
# This class inherits from RoboSuiteDoorCloseGymnasiumEnv (the same base as v1).
#
# Observation space (v2):
#   base_flat            ~32 dim  (from RoboSuiteDoorCloseGymnasiumEnv)
#   custom_v1            8  dim   [dist, radius, friction, fsm_onehot×4, hinge_angle]
#   grasp_strategy       4  dim   [best_align, align_top, align_latL, align_latR]
#   physics_context      3  dim   [norm_stiffness, norm_damping, norm_mass]
#   ─────────────────────────────
#   Total               ~47 dim

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging
import numpy as np

from train_close import RoboSuiteDoorCloseGymnasiumEnv
from scipy.spatial.transform import Rotation as R_scipy

from robosuite.utils.log_utils import ROBOSUITE_DEFAULT_LOGGER
ROBOSUITE_DEFAULT_LOGGER.setLevel(logging.ERROR)

from close_generalized_v2.config_v2      import TrainConfigV2
from close_generalized_v2.fsm_v2         import AdaptiveFSM, PHASE_REACH, PHASE_PUSH, PHASE_HOLD, PHASE_RETREAT
from close_generalized_v2.reward_v2      import PotentialBasedReward
from close_generalized_v2.grasp_strategy import MultiApproachGrasp
from close_generalized_v2.domain_rand_v2 import ExtendedDomainRandomizer
from close_generalized_v2.beta_net       import BetaNetwork

class AdvancedGeneralizedDoorEnv(RoboSuiteDoorCloseGymnasiumEnv):
    def __init__(self, cfg: TrainConfigV2, render_mode=None):
        self._fsm            = AdaptiveFSM(cfg)          # provides .state.phase / .state.one_hot
        self._grasp_strategy = MultiApproachGrasp(cfg)   # provides .obs_features() → zeros

        # Minimal domain-rand stub so _flatten_obs() is safe before the real instance is ready.
        # Values match the MuJoCo XML defaults (same as domain_rand_v2.py base values).
        from types import SimpleNamespace
        self._domain_rand = SimpleNamespace(
            current_handle_radius   = 0.021,
            current_handle_friction = 0.8,
            obs_features            = lambda: np.zeros(3, dtype=np.float32),
        )

        self.curriculum_level = (
            cfg.fixed_curriculum_level
            if getattr(cfg, "fixed_curriculum_level", None) is not None
            else 0.0
        )

        # ── Call parent __init__ (will call _flatten_obs once) ────────────────
        super().__init__(cfg, render_mode)

        # ── Replace stubs with fully-initialised real instances ───────────────
        self.door_body_id = self._rs_env.sim.model.body_name2id("Door_main")
        self.base_pos     = self._rs_env.sim.model.body_pos[self.door_body_id].copy()
        self.base_quat    = self._rs_env.sim.model.body_quat[self.door_body_id].copy()

        # FSM was already created as a full instance above (no need to re-create)
        # Ref: Konidaris & Barto (2009), Sutton et al. (1999)

        # ── §3.2 Potential-Based Reward ───────────────────────────────────────
        # Ref: Ng, Russell & Harada (1999), Devlin & Kudenko (2012)
        self._reward_fn = PotentialBasedReward(cfg, gamma=cfg.gamma)

        # ── §3.3 Multi-Approach Grasp (already a real instance) ───────────────
        # Ref: ten Pas et al. (2017), ManipForce (2015)

        # ── §3.4 Extended Domain Randomizer (real instance replaces stub) ─────
        # Ref: Tobin et al. (2017), Zhao et al. (2020)
        self._domain_rand = ExtendedDomainRandomizer(cfg, self._rs_env.sim.model)

        # ── §3.5 Beta Network (learned termination) ───────────────────────────
        # Ref: Sutton, Precup & Singh (1999)
        self._beta_net = BetaNetwork(cfg)

        # ── Diagnostics ───────────────────────────────────────────────────────
        self._prev_action     : np.ndarray = np.zeros(self.action_space.shape)
        self._prev_eef_action : np.ndarray = np.zeros(self.action_space.shape[0] - 1)
        self._prev_door_angle : float = None
        self._prev_gripper_width : float = 0.08  # §1.17 — larghezza gripper passo prec. (init: aperto)
        self._prev_is_phys_closed : bool = False  # §1.18 — presa fisicamente chiusa al passo prec.
        self._diag_step       : int   = 0

    # ── Curriculum API (same as v1) ───────────────────────────────────────────
    def set_curriculum_level(self, level: float) -> None:
        self.curriculum_level = float(np.clip(level, 0.0, 1.0))

    # ── Properties (delegate to FSM state for external callbacks) ─────────────
    @property
    def _grasp_phase(self) -> bool:
        return self._fsm.state.phase == PHASE_PUSH

    @_grasp_phase.setter
    def _grasp_phase(self, _value) -> None:  # no-op: FSM state is authoritative
        pass

    @property
    def _success_latched(self) -> bool:
        return self._fsm.state.phase in (PHASE_HOLD, PHASE_RETREAT)

    @_success_latched.setter
    def _success_latched(self, _value) -> None:  # no-op: FSM state is authoritative
        pass

    @property
    def _ready_to_retreat(self) -> bool:
        return self._fsm.state.phase == PHASE_RETREAT

    @_ready_to_retreat.setter
    def _ready_to_retreat(self, _value) -> None:  # no-op: FSM state is authoritative
        pass

    # ── Observation ────────────────────────────────────────────────────────────
    def _flatten_obs(self, obs: dict) -> np.ndarray:
        eef_pos    = obs.get("robot0_eef_pos", np.zeros(3))
        handle_pos = obs.get("handle_pos", obs.get("door_handle_pos", eef_pos))
        dist       = float(np.linalg.norm(handle_pos - eef_pos))

        fsm_onehot = self._fsm.state.one_hot  # [REACH, PUSH, HOLD, RETREAT]

        # Door hinge angle (live from sim)
        try:
            hinge_qpos = float(self._rs_env.sim.data.qpos[self._rs_env.hinge_qpos_addr])
        except Exception:
            hinge_qpos = float(obs.get("hinge_qpos", 0.0))

        # ── §3.3 Multi-approach grasp features ────────────────────────────────
        # Ref: ten Pas et al. (2017)
        eef_quat         = obs.get("robot0_eef_quat")

        # door_body_id is not yet set during the super().__init__() call — fall back
        # to identity quaternion [w = 1, x = 0, y = 0, z = 0] so grasp_feats are all-zeros.
        _body_id         = getattr(self, "door_body_id", None)
        if _body_id is not None:
            door_quat_mujoco = self._rs_env.sim.model.body_quat[_body_id]
        else:
            door_quat_mujoco = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        grasp_feats      = self._grasp_strategy.obs_features(
            eef_quat, handle_pos, eef_pos, door_quat_mujoco
        )  # [4 dim]

        # ── §3.4 Extended physics context ─────────────────────────────────────
        # Ref: Tobin et al. (2017), Zhao et al. (2020)
        physics_feats = self._domain_rand.obs_features()  # [3 dim]

        custom = np.array([
            dist,
            self._domain_rand.current_handle_radius,
            self._domain_rand.current_handle_friction,
            *fsm_onehot,
            hinge_qpos,
        ], dtype=np.float32)

        base_flat = super()._flatten_obs(obs)
        return np.concatenate([base_flat, custom, grasp_feats, physics_feats])

    # ── Step ───────────────────────────────────────────────────────────────────
    def step(self, action: np.ndarray):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)

        # Action smoothing (same as v1)
        alpha = getattr(self.cfg, "action_smooth_alpha", 1.0)
        if alpha < 1.0:
            action = alpha * action + (1.0 - alpha) * self._prev_action

        # ── FSM action overrides (same logic as v1) ──────────────────────────
        phase = self._fsm.state.phase
        if phase == PHASE_HOLD:
            action[:-1] = 0.0  # Hard Freeze on arm  [§3.1 fix]
            # §1.18 — GRIP-LOCK anche in HOLD: se la presa era fisicamente chiusa,
            # impedisce aperture accidentali mentre si tiene la porta (anti hold_slip).
            if getattr(self.cfg, "grip_lock_enabled", True) and self._prev_is_phys_closed:
                grip_floor = min(1.0, self._fsm.grip_thresh(
                    self._domain_rand.current_handle_friction) + self.cfg.grip_lock_margin)
                action[-1] = max(float(action[-1]), grip_floor)

        elif phase == PHASE_PUSH:
            # §1.18 (env-level, ZERO modifiche al reward) — GRIP-LOCK in chiusura.
            # Causa osservata nei log: durante PUSH la policy stocastica manda ogni
            # tanto comandi di APERTURA (rumore di esplorazione: "PUSH→REACH
            # (grip=-0.56)") → presa persa, FSM retrocede, hold_slip. Qui, SE al passo
            # precedente la presa era fisicamente chiusa, il comando del gripper viene
            # clampato a >= grip_thresh(frizione)+margine: il lock è DIREZIONALE
            # (blocca solo l'apertura, non stringe oltre la richiesta della policy,
            # per non scivolare sotto la banda di contatto su maniglie sottili §3.1).
            # Deterministico, zero reward ⇒ non crea nuovi ottimi (stesso pattern di
            # §1.17 e dell'hard-freeze di HOLD). Rif.: presa validata dal contatto
            # [13]; soglia adattiva alla frizione [15].
            if getattr(self.cfg, "grip_lock_enabled", True) and self._prev_is_phys_closed:
                grip_floor = min(1.0, self._fsm.grip_thresh(
                    self._domain_rand.current_handle_friction) + self.cfg.grip_lock_margin)
                action[-1] = max(float(action[-1]), grip_floor)

        elif phase == PHASE_RETREAT:
            # §1.17 (env-level, ZERO modifiche al reward) — RILASCIO PULITO prima del
            # ritiro. Finché le dita non hanno superato la maniglia (gripper_width oltre
            # il diametro + margine), si forza il gripper completamente APERTO e si CONGELA
            # il braccio (come l'hard-freeze di HOLD): la maniglia può così "risalire"
            # (latch → riposo) senza essere pizzicata, e SOLO dopo il rilascio fisico il
            # braccio si allontana. Deterministico, non tocca il reward ⇒ non crea nuovi
            # ottimi né intacca il 100% (rilascio basato sul contatto [13]; RETREAT come
            # opzione a terminazione pulita [1]).
            handle_diam   = self._domain_rand.current_handle_radius * 2.0
            fingers_clear = self._prev_gripper_width > (handle_diam + self.cfg.retreat_clear_margin)

            if getattr(self.cfg, "retreat_clean_release", True) and not fingers_clear:
                action[:-1] = 0.0     # congela il braccio (agisce solo il gripper)
                action[-1]  = -1.0    # gripper completamente aperto → rilascio pulito
            else:
                retreat_pos = self._fsm.state.retreat_pos
                if retreat_pos is not None:
                    eef_site_id = self._rs_env.robots[0].eef_site_id
                    if isinstance(eef_site_id, dict):
                        site_id = eef_site_id.get('right', list(eef_site_id.values())[0])
                    else:
                        site_id = eef_site_id
                    eef_pos      = self._rs_env.sim.data.site_xpos[site_id]
                    dist_retreat = float(np.linalg.norm(eef_pos - retreat_pos))
                    returned     = dist_retreat < self.cfg.return_pos_tol

                    if returned or self._fsm.state.return_hold >= self.cfg.return_hold_steps:
                        action = np.zeros_like(action)

        obs, _, rs_done, info = self._rs_env.step(action)
        self._step_count += 1

        door_angle  = self._get_door_angle()
        prev_angle  = float(self._prev_door_angle) if self._prev_door_angle is not None else door_angle
        self._prev_door_angle = door_angle

        # ── §3.5 Beta-network predictions ────────────────────────────────────
        # Ref: Sutton, Precup & Singh (1999)
        fsm_s       = self._fsm.state
        dr          = self._domain_rand
        physics_obs = dr.obs_features()
        door_qpos   = float(self._rs_env.sim.data.qpos[self._rs_env.hinge_qpos_addr])
        door_qvel   = float(self._rs_env.sim.data.qvel[self._door_hinge_dof_adr])
        door_speed  = abs(prev_angle - door_angle) * self.cfg.control_freq

        beta_probs = self._beta_net.predict(
            dist_handle         = float(np.linalg.norm(
                obs.get("robot0_eef_pos", np.zeros(3))
                - obs.get("handle_pos", obs.get("door_handle_pos", np.zeros(3)))
            )),
            handle_radius        = dr.current_handle_radius,
            handle_friction      = dr.current_handle_friction,
            gripper_width        = float(np.sum(np.abs(obs.get("robot0_gripper_qpos", [0,0])))),
            gripper_action       = float(action[-1]),
            door_angle           = door_angle,
            door_speed           = door_speed,
            door_qpos            = door_qpos,
            door_qvel            = door_qvel,
            hold_duration        = fsm_s.hold_closed_duration,
            target_hold_steps    = fsm_s.target_hold_steps or 60,
            norm_latch_stiffness = float(physics_obs[0]),
            norm_door_mass       = float(physics_obs[2]),
        )

        # ── §3.1 FSM update ───────────────────────────────────────────────────
        # Ref: Konidaris & Barto (2009), ManipForce (2015)
        eef_pos     = obs.get("robot0_eef_pos", np.zeros(3))
        handle_pos  = obs.get("handle_pos", obs.get("door_handle_pos", eef_pos))
        dist_handle = float(np.linalg.norm(eef_pos - handle_pos))

        gripper_qpos = obs.get("robot0_gripper_qpos")
        if gripper_qpos is not None:
            gripper_width  = float(np.sum(np.abs(gripper_qpos)))
            handle_diam    = dr.current_handle_radius * 2.0
            is_phys_closed = (gripper_width <= handle_diam + 0.025) and (gripper_width >= 0.015)
        else:
            gripper_width  = 0.0
            is_phys_closed = float(action[-1]) > 0.65

        fsm_events = self._fsm.update(
            door_angle          = door_angle,
            success_angle       = self._success_angle,
            gripper_action      = float(action[-1]),
            dist_handle         = dist_handle,
            handle_radius       = dr.current_handle_radius,
            handle_friction     = dr.current_handle_friction,
            is_physically_closed= is_phys_closed,
            gripper_width       = gripper_width,
            prev_angle          = prev_angle,
            control_freq        = self.cfg.control_freq,
            door_qpos           = door_qpos,
            eef_pos             = eef_pos,
            door_quat_mujoco    = self._rs_env.sim.model.body_quat[self.door_body_id],
            latch_stiffness     = dr.current_latch_stiffness,
            base_latch_stiffness= dr.base_latch_stiffness or 1.0,
            beta_probs          = beta_probs if self.cfg.use_beta_net else None,
        )

        # Return-hold counter update (same logic as v1)
        if self._fsm.state.phase == PHASE_RETREAT and self._fsm.state.retreat_pos is not None:
            eef_site_id = self._rs_env.robots[0].eef_site_id
            if isinstance(eef_site_id, dict):
                site_id = eef_site_id.get('right', list(eef_site_id.values())[0])
            else:
                site_id = eef_site_id
            eef_pos_site = self._rs_env.sim.data.site_xpos[site_id]
            dist_r = float(np.linalg.norm(eef_pos_site - self._fsm.state.retreat_pos))
            if dist_r < self.cfg.return_pos_tol:
                self._fsm.state.return_hold += 1
            else:
                self._fsm.state.return_hold = 0

        # ── §3.3 Grasp alignment (multi-approach) ────────────────────────────
        # Ref: ten Pas et al. (2017)
        eef_quat         = obs.get("robot0_eef_quat")
        door_quat_mujoco = self._rs_env.sim.model.body_quat[self.door_body_id]

        best_align, best_idx, all_aligns = 0.0, 0, [0.0] * self.cfg.grasp_n_candidates
        flat_alignment = 0.0

        if eef_quat is not None:
            best_align, best_idx, all_aligns = self._grasp_strategy.compute_alignment(
                eef_quat, handle_pos, eef_pos, door_quat_mujoco
            )
            rmat          = R_scipy.from_quat(eef_quat).as_matrix()
            flat_alignment = abs(float(rmat[2, 0]))  # |eef_x[2]|

        # ── §3.2 Reward computation ───────────────────────────────────────────
        # Ref: Ng, Russell & Harada (1999)
        base_reward, terminated, truncated = super()._calculate_reward(
            action, obs, rs_done, door_angle, prev_angle, fsm_events["just_succeeded"]
        )

        latch_qpos = float(self._rs_env.sim.data.qpos[self._rs_env.handle_qpos_addr])
        joint_vel  = obs.get("robot0_joint_vel")
        height_diff = float(eef_pos[2] - handle_pos[2])
        dist_xy     = float(np.linalg.norm(eef_pos[:2] - handle_pos[:2]))

        grip_thresh = self._fsm.grip_thresh(dr.current_handle_friction)

        reward, terminated, truncated, rew_info = self._reward_fn.compute(
            fsm_state           = self._fsm.state,
            base_reward         = base_reward,
            door_angle          = door_angle,
            door_max            = 0.4,
            door_qpos           = door_qpos,
            dist_handle         = dist_handle,
            dist_xy             = dist_xy,
            height_diff         = height_diff,
            handle_radius       = dr.current_handle_radius,
            handle_friction     = dr.current_handle_friction,
            grip_thresh         = grip_thresh,
            gripper_action      = float(action[-1]),
            gripper_width       = gripper_width,
            is_physically_closed= is_phys_closed,
            gripper_qpos        = gripper_qpos,
            alignment           = best_align,      # max over K directions [§3.3]
            flat_alignment      = flat_alignment,
            joint_vel           = joint_vel,
            action              = action,
            prev_eef_action     = self._prev_eef_action,
            eef_pos             = eef_pos,
            latch_qpos          = latch_qpos,
            door_qvel           = door_qvel,
            curriculum_lvl      = self.curriculum_level,
            just_grasped        = fsm_events["just_grasped"],
            just_succeeded      = fsm_events["just_succeeded"],
            just_hold_done      = fsm_events["just_hold_done"],
            grasp_lost          = fsm_events["grasp_lost"],
            terminated          = terminated,
            truncated           = truncated,
        )

        self._prev_action     = action.copy()
        self._prev_eef_action = action[:-1].copy()
        self._prev_gripper_width = float(gripper_width)  # §1.17
        self._prev_is_phys_closed = bool(is_phys_closed)  # §1.18

        # ── Info dict ─────────────────────────────────────────────────────────
        info                       = dict(info or {})
        info["is_success"]         = self._fsm.state.phase in (PHASE_HOLD, PHASE_RETREAT)
        info["door_angle"]         = door_angle
        info["door_qpos"]          = door_qpos
        info["latch_qpos"]         = latch_qpos
        info["fsm_phase"]          = self._fsm.state.phase
        info["fsm_phase_name"]     = self._fsm.state.phase_name
        info["hold_duration"]      = self._fsm.state.hold_closed_duration
        info["target_hold_steps"]  = self._fsm.state.target_hold_steps
        info["curriculum_level"]   = self.curriculum_level
        info["best_grasp_align"]   = best_align
        info["best_grasp_dir_idx"] = best_idx

        # ── Diagnostics ───────────────────────────────────────────────────────
        self._diag_step += 1
        if self._diag_step % self.cfg.debug_print_every == 0:
            self._print_diag(
                dist_handle, height_diff, float(action[-1]),
                is_phys_closed, gripper_width, best_align,
                door_angle, latch_qpos, reward, rew_info
            )

        return self._flatten_obs(obs), reward, terminated, truncated, info

    # ── Reset ─────────────────────────────────────────────────────────────────

    def reset(self, seed: int = None, options: dict = None):
        # §1.15 — Se il livello di curriculum è fissato, lo ri-ancoriamo a ogni reset
        # così nessun callback può farlo driftare (training a posa fissa o variabile).
        if getattr(self.cfg, "fixed_curriculum_level", None) is not None:
            self.curriculum_level = float(np.clip(self.cfg.fixed_curriculum_level, 0.0, 1.0))

        # ── §3.4 Extended domain randomization ───────────────────────────────
        # Ref: Tobin et al. (2017), Zhao et al. (2020)
        self._domain_rand.randomize_episode(self.curriculum_level)

        # ── §3.1 FSM reset with adaptive hold timer ───────────────────────────
        # Ref: Konidaris & Barto (2009), ManipForce (2015)
        base_stiff = self._domain_rand.base_latch_stiffness or 1.0
        self._fsm.reset(
            latch_stiffness      = self._domain_rand.current_latch_stiffness,
            base_latch_stiffness = base_stiff,
            control_freq         = self.cfg.control_freq,
        )

        # ── §3.2 Reward reset (clear cached Φ) ───────────────────────────────
        # Ref: Ng, Russell & Harada (1999)
        self._reward_fn.reset()

        # ── Diagnostics reset ─────────────────────────────────────────────────
        self._prev_action     = np.zeros(self.action_space.shape)
        self._prev_eef_action = np.zeros(self.action_space.shape[0] - 1)
        self._prev_door_angle = None
        self._prev_gripper_width = 0.08  # §1.17
        self._prev_is_phys_closed = False  # §1.18
        self._diag_step       = 0

        # ── Door position/yaw randomization (same as v1, via curriculum) ─────
        p_var = 0.15 * self.curriculum_level
        r_var = 0.30 * self.curriculum_level

        if self.curriculum_level > 0:
            pos_offset    = np.random.uniform(-p_var, p_var, size=3)
            pos_offset[2] = 0.0
            yaw           = np.random.uniform(-r_var, r_var)
            q_scipy       = R_scipy.from_euler('z', yaw).as_quat()

            self._rs_env.sim.model.body_pos[self.door_body_id] = self.base_pos + pos_offset

            if hasattr(self.cfg, "human_dist_min") and hasattr(self.cfg, "human_dist_max"):
                dist_s = np.random.uniform(self.cfg.human_dist_min, self.cfg.human_dist_max)
                self._rs_env.sim.model.body_pos[self.door_body_id][0] = dist_s + pos_offset[0]

            q_base = R_scipy.from_quat([
                self.base_quat[1], self.base_quat[2],
                self.base_quat[3], self.base_quat[0]
            ])
            q_new = R_scipy.from_quat(q_scipy) * q_base
            res_q = q_new.as_quat()
            self._rs_env.sim.model.body_quat[self.door_body_id] = np.array([
                res_q[3], res_q[0], res_q[1], res_q[2]
            ])

        return super().reset(seed=seed, options=options)

    # ── Diagnostics ───────────────────────────────────────────────────────────
    def _print_diag(
        self, dist, dz, grip, phys, width, align, door, latch, reward, rew_info
    ) -> None:
        s         = self._fsm.state
        phase_str = f"{s.phase}:{s.phase_name}"
        phys_str  = "PHYS_OK" if phys else "PHYS_OPEN"
        print(f"┌─────────┬────────┬────────┬───────┬───────────┬───────┬───────┬───────┬───────┐")
        print(f"│  PHASE  │  DIST  │   dZ   │ GRIP  │   PHYS    │ WIDTH │ ALIGN │ DOOR  │ LATCH │")
        print(f"├─────────┼────────┼────────┼───────┼───────────┼───────┼───────┼───────┼───────┤")
        print(f"│ {phase_str:<7} │ {dist:>6.3f} │ {dz:>+6.3f} │ {grip:>+5.2f} │ {phys_str:<9} │ {width:>5.3f} │ {align:>5.2f} │ {door:>5.2f} │ {latch:>+5.2f} │")
        print(f"└─────────┴────────┴────────┴───────┴───────────┴───────┴───────┴───────┴───────┘")
        rew_str = " │ ".join([f"{k}: {v:>+5.2f}" for k, v in rew_info.items() if abs(v) > 0.001])
        print(f"  ↳ REWARDS │ {rew_str} │ TOT: {reward:>+6.2f}")
        if s.events:
            print(f"  ↳ FSM LOG │ {' | '.join(set(s.events))}")
            s.events.clear()
        print()