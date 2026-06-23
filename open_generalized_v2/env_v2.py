#!/usr/bin/env python3
# open_generalized_v2/env_v2.py
#
# AdvancedGeneralizedOpenDoorEnv — environment top-level dell'APERTURA generalizzata (v2),
# SPECULARE a close_generalized_v2/env_v2.py ma per il task di apertura, e tarato per il
# SOLO curriculum 1 (posa variabile + soglie adattive + fisica randomizzata).
#
# Pipeline di uno step (specchio della chiusura):
#   1) smoothing azione (EMA)                                            [come v1 apertura]
#   2) override DETERMINISTICI env-level a successo già acquisito:
#        §1.18 grip-lock in PULL/HOLD_OPEN  (blocca aperture accidentali della presa)
#        §1.17 rilascio pulito in RETREAT   (apri gripper, congela braccio fino a dita libere)
#        §1.21 rampa di avvio del ritiro     (scala 0→1 l'azione del braccio dopo il rilascio)
#   3) step del simulatore robosuite
#   4) lettura stato (door_angle, latch, presa) e update della FSM adattiva
#   5) reward potential-based + terminazione
#   6) info ricca (fase FSM, latch, door, goal, hold, curriculum) per test/log
#
# Inversione chiusura ↔ apertura: l'obiettivo è door_angle ≈ goal_angle (porta aperta al
# valore richiesto), non door_angle ≈ 0. Il goal è campionato a ogni reset (curriculum 1).
#
# Riferimenti: [1][2] opzioni/precondizioni, [3] shaping invariante, [8][17] domain rand,
#              [13] presa al contatto, [15] grasp 6-D.

from __future__ import annotations

import os
import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Any, Dict, List, Optional
from scipy.spatial.transform import Rotation as R_scipy

# Path setup PRIMA di qualsiasi import interno, così funziona sia lanciato come
# script (mjpython open_generalized_v2/train_curriculum_v2.py) sia come modulo (-m).
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import logging
from robosuite.utils.log_utils import ROBOSUITE_DEFAULT_LOGGER
ROBOSUITE_DEFAULT_LOGGER.setLevel(logging.ERROR)

# Import robusti: prima qualificati col package, poi fallback "piatti".
try:
    from open_generalized_v2.config_v2 import TrainConfigV2Open
    from open_generalized_v2.fsm_v2 import (AdaptiveFSMOpen, PHASE_REACH, PHASE_PULL,
                                            PHASE_HOLD_OPEN, PHASE_RETREAT)
    from open_generalized_v2.reward_v2 import PotentialBasedRewardOpen
    from open_generalized_v2.domain_rand_v2 import ExtendedDomainRandomizer
    from open_generalized_v2.grasp_strategy import MultiApproachGrasp
except ModuleNotFoundError:
    from config_v2 import TrainConfigV2Open
    from fsm_v2 import (AdaptiveFSMOpen, PHASE_REACH, PHASE_PULL,
                        PHASE_HOLD_OPEN, PHASE_RETREAT)
    from reward_v2 import PotentialBasedRewardOpen
    from domain_rand_v2 import ExtendedDomainRandomizer
    from grasp_strategy import MultiApproachGrasp


class AdvancedGeneralizedOpenDoorEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 20}

    def __init__(self, cfg: TrainConfigV2Open, render_mode: Optional[str] = None):
        super().__init__()
        self.cfg = cfg
        self.render_mode = render_mode
        self.curriculum_level = float(cfg.fixed_curriculum_level)

        import robosuite as suite
        from robosuite.controllers import load_composite_controller_config
        controller_config = load_composite_controller_config(controller="BASIC")

        self._rs_env = suite.make(
            env_name               = cfg.env_name,
            robots                 = cfg.robot,
            has_renderer           = (render_mode == "human"),
            has_offscreen_renderer = False,
            use_camera_obs         = False,
            use_object_obs         = True,
            reward_shaping         = False,
            horizon                = cfg.horizon,
            control_freq           = cfg.control_freq,
            controller_configs     = controller_config,
            ignore_done            = True,
        )
        sim = self._rs_env.sim

        # ── hinge della porta ──
        hinge = [n for n in sim.model.joint_names if ("door" in n.lower() and "hinge" in n.lower())]
        if not hinge:
            hinge = [n for n in sim.model.joint_names if "hinge" in n.lower()]
        self._door_hinge_name = hinge[0]
        jid = sim.model.joint_name2id(self._door_hinge_name)
        jmin, jmax = sim.model.jnt_range[jid]
        self._door_min = float(jmin)
        self._door_max = float(jmax)
        self._effective_max = float(min(self._door_max, self._door_min + cfg.door_open_cap_rad))
        self._door_hinge_qpos_adr = int(sim.model.jnt_qposadr[jid])
        self._door_hinge_dof_adr  = int(sim.model.jnt_dofadr[jid])

        # ── latch (handle joint) per lo stato di presa/aggancio ──
        latch = [n for n in sim.model.joint_names if "latch" in n.lower()]
        self._latch_qpos_adr = (
            int(sim.model.jnt_qposadr[sim.model.joint_name2id(latch[0])]) if latch else self._door_hinge_qpos_adr
        )

        # ── geom maniglia (per dist_handle e raggio) ──
        self.handle_geom_id = None
        for i, n in enumerate(sim.model.geom_names):
            if "handle" in n.lower():
                self.handle_geom_id = i
                break

        try:
            self.door_body_id = sim.model.body_name2id("Door_main")
        except Exception:
            self.door_body_id = 0

        # ── moduli v2 ──
        self._domain_rand   = ExtendedDomainRandomizer(cfg, sim.model)
        self._grasp_strategy = MultiApproachGrasp(cfg)
        self._fsm           = AdaptiveFSMOpen(cfg)
        self._reward_fn     = PotentialBasedRewardOpen(cfg, gamma=cfg.gamma)

        # ── default di stato necessari PRIMA del primo _flatten_obs ──
        # _flatten_obs() usa self._goal_angle per la feature di goal: va inizializzato
        # qui (default = apertura piena), poi reset() lo ricampiona a ogni episodio.
        self._goal_angle = self._effective_max

        # ── spazi ──
        obs = self._rs_env.reset()
        self._obs_keys = sorted(k for k, v in obs.items()
                                if isinstance(v, np.ndarray) and v.dtype != object and v.ndim == 1)
        flat = self._flatten_obs(obs)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=flat.shape, dtype=np.float32)
        low, high = self._rs_env.action_spec
        self.action_space = spaces.Box(low.astype(np.float32), high.astype(np.float32), dtype=np.float32)

        # ── stato episodio ──
        self._step_count = 0
        self._goal_angle = self._effective_max
        self._prev_door_angle: Optional[float] = None
        self._prev_action = None
        self._start_eef_pos: Optional[np.ndarray] = None
        # override deterministici (specchio della chiusura)
        self._prev_gripper_width = 0.08      # §1.17
        self._prev_is_phys_closed = False    # §1.18
        self._retreat_ramp_step = 0          # §1.21

    # ── helpers ──────────────────────────────────────────────────────────────────

    def set_curriculum_level(self, level: float) -> None:
        self.curriculum_level = float(level)

    def _door_angle(self) -> float:
        a = float(self._rs_env.sim.data.qpos[self._door_hinge_qpos_adr])
        return float(np.clip(a, self._door_min, self._effective_max))

    def _latch_qpos(self) -> float:
        return float(self._rs_env.sim.data.qpos[self._latch_qpos_adr])

    def _eef_pos(self) -> np.ndarray:
        eef_site = self._rs_env.robots[0].eef_site_id
        sid = eef_site.get('right', list(eef_site.values())[0]) if isinstance(eef_site, dict) else eef_site
        return np.array(self._rs_env.sim.data.site_xpos[sid], dtype=float)

    def _handle_pos(self) -> np.ndarray:
        if self.handle_geom_id is not None:
            return np.array(self._rs_env.sim.data.geom_xpos[self.handle_geom_id], dtype=float)
        return self._eef_pos()

    def _flatten_obs(self, obs: dict) -> np.ndarray:
        parts = [obs[k].ravel().astype(np.float32) for k in self._obs_keys]
        base = np.concatenate(parts, axis=0)
        eef_quat = obs.get("robot0_eef_quat")
        eef_pos  = obs.get("robot0_eef_pos", np.zeros(3))
        handle_pos = obs.get("handle_pos", obs.get("door_handle_pos", eef_pos))
        try:
            dq = self._rs_env.sim.model.body_quat[self.door_body_id]
        except Exception:
            dq = None
        grasp_feats = self._grasp_strategy.obs_features(eef_quat, np.asarray(handle_pos), np.asarray(eef_pos), dq)
        physics_feats = self._domain_rand.obs_features()
        # feature di goal (apertura): goal normalizzato nel range effettivo
        goal_norm = np.array([
            (self._goal_angle - self._door_min) / (self._effective_max - self._door_min + 1e-8)
        ], dtype=np.float32)
        return np.concatenate([base, grasp_feats, physics_feats, goal_norm], axis=0)

    # ── reset ─────────────────────────────────────────────────────────────────────

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        obs = self._rs_env.reset()

        # domain randomization fisica (sempre attiva)
        self._domain_rand.randomize_episode(self.curriculum_level)

        # curriculum 1: campiona un GOAL di apertura variabile nel range alto
        f = np.random.uniform(self.cfg.goal_frac_min, self.cfg.goal_frac_max)
        self._goal_angle = float(self._door_min + f * (self._effective_max - self._door_min))

        self._fsm.reset()
        self._reward_fn.reset()
        self._step_count = 0
        self._prev_door_angle = self._door_angle()
        self._prev_action = None
        self._start_eef_pos = self._eef_pos().copy()
        self._prev_gripper_width = 0.08
        self._prev_is_phys_closed = False
        self._retreat_ramp_step = 0

        info = {
            "goal_angle": self._goal_angle,
            "door_min": self._door_min,
            "effective_max": self._effective_max,
            "curriculum_level": self.curriculum_level,
        }
        return self._flatten_obs(obs), info

    # ── step ──────────────────────────────────────────────────────────────────────

    def step(self, action: np.ndarray):
        action = np.clip(np.asarray(action, dtype=np.float32).copy(), -1.0, 1.0)
        if self._prev_action is None:
            self._prev_action = np.zeros_like(action)
        a = float(self.cfg.action_smooth_alpha)
        action = a * action + (1.0 - a) * self._prev_action

        dr = self._domain_rand
        handle_radius = dr.current_handle_radius
        handle_friction = dr.current_handle_friction

        # presa fisica corrente (specchio chiusura): dita attorno alla maniglia
        gw = self._prev_gripper_width
        handle_diam = handle_radius * 2.0
        is_phys_closed = (gw <= handle_diam + 0.025) and (gw >= 0.015)

        phase = self._fsm.state.phase

        # azzera la rampa quando NON siamo in RETREAT (§1.21)
        if phase != PHASE_RETREAT:
            self._retreat_ramp_step = 0

        # ── §1.18 GRIP-LOCK in PULL e HOLD_OPEN ──
        if phase in (PHASE_PULL, PHASE_HOLD_OPEN):
            if getattr(self.cfg, "grip_lock_enabled", True) and self._prev_is_phys_closed:
                grip_floor = min(1.0, self._fsm.grip_thresh(handle_friction) + self.cfg.grip_lock_margin)
                action[-1] = max(float(action[-1]), grip_floor)

        # ── §1.22 ACCOMPAGNA LEVA + §1.17 RILASCIO PULITO + §1.21 RAMPA in RETREAT ──
        elif phase == PHASE_RETREAT:
            # §1.22 — PRIMA di rilasciare, accompagna la LEVA alla posizione di partenza
            # (env-level, ZERO reward, NESSUNO step extra: si MODIFICA solo l'azione).
            # Mantieni la presa e congela il braccio: la molla di richiamo del latch riporta
            # la leva a latch≈0. Solo quando |latch_qpos| è sotto soglia → rilascio pulito.
            # Specchio della chiusura, che NON termina finché la leva non è neutra.
            latch_neutral = abs(self._latch_qpos()) <= getattr(self.cfg, "retreat_latch_neutral_tol", 0.05)
            fingers_clear = self._prev_gripper_width > (handle_diam + self.cfg.retreat_clear_margin)

            # §1.26 — il ramo di accompagnamento leva è attivo SOLO entro un cap di step:
            # superati, si procede comunque a rilascio+ritiro anche se la leva non è neutra
            # (evita che il braccio resti aggrappato all'infinito quando la leva non torna a 0).
            _latch_steps_ok = self._fsm.state.retreat_steps <= getattr(self.cfg, "retreat_latch_max_steps", 20)

            if getattr(self.cfg, "retreat_latch_restore", True) and not latch_neutral and _latch_steps_ok:
                # leva ancora ruotata: tieni la presa e congela il braccio (lascia agire la molla)
                action[:-1] = 0.0
                if self._prev_is_phys_closed:
                    grip_floor = min(1.0, self._fsm.grip_thresh(handle_friction) + self.cfg.grip_lock_margin)
                    action[-1] = max(float(action[-1]), grip_floor)
                else:
                    action[-1] = 1.0
                self._retreat_ramp_step = 0
            elif getattr(self.cfg, "retreat_clean_release", True) and not fingers_clear:
                action[:-1] = 0.0      # congela il braccio
                action[-1]  = -1.0     # gripper aperto → rilascio pulito
                self._retreat_ramp_step = 0
            else:
                # §1.21 — avvio morbido del ritiro (scala SOLO il braccio)
                _R = int(getattr(self.cfg, "retreat_rampup_steps", 0))
                if getattr(self.cfg, "retreat_rampup_enabled", True) and _R > 0 \
                        and self._retreat_ramp_step < _R:
                    _scale = float(self._retreat_ramp_step + 1) / float(_R)
                    action[:-1] = action[:-1] * _scale
                    self._retreat_ramp_step += 1

        self._prev_action = action.copy()

        # ── step simulatore ──
        obs, _, rs_done, _ = self._rs_env.step(action)
        self._step_count += 1

        door_angle = self._door_angle()
        prev_angle = float(self._prev_door_angle) if self._prev_door_angle is not None else door_angle
        latch_qpos = self._latch_qpos()
        door_qpos = float(self._rs_env.sim.data.qpos[self._door_hinge_qpos_adr])

        # dist_handle dall'OSSERVAZIONE robosuite (come l'env di chiusura che funziona:
        # env_gen.py righe 70-72). I metodi site_xpos/geom_xpos davano un handle_geom
        # errato (primo geom con "handle" nel nome ≠ centro afferrabile), con dist ~0.3 m
        # costante → REACH→PULL mai raggiunto. Fallback ai metodi solo se le chiavi mancano.
        obs_eef    = obs.get("robot0_eef_pos", None)
        obs_handle = obs.get("handle_pos", obs.get("door_handle_pos", None))
        if obs_eef is not None and obs_handle is not None:
            eef_pos    = np.asarray(obs_eef, dtype=float)
            handle_pos = np.asarray(obs_handle, dtype=float)
            _handle_src = "obs"
        else:
            eef_pos    = self._eef_pos()
            handle_pos = self._handle_pos()
            _handle_src = "fallback_methods"
        dist_handle = float(np.linalg.norm(eef_pos - handle_pos))
        # componenti per i termini densi di REACH (mirror chiusura): xy e dislivello z
        dist_xy     = float(np.linalg.norm(eef_pos[:2] - handle_pos[:2]))
        height_diff = float(eef_pos[2] - handle_pos[2])

        gripper_qpos = obs.get("robot0_gripper_qpos")
        gripper_width = float(np.sum(np.abs(gripper_qpos))) if gripper_qpos is not None else gw
        is_phys_closed = (gripper_width <= handle_diam + 0.025) and (gripper_width >= 0.015)

        # gate del polso (riusa la geometria del grasp; opzionale e robusto)
        wrist_align_ok = True

        prev_phase = self._fsm.state.phase
        fsm_events = self._fsm.update(
            door_angle           = door_angle,
            goal_angle           = self._goal_angle,
            open_tol             = self.cfg.open_tol_rad,
            gripper_action       = float(action[-1]),
            dist_handle          = dist_handle,
            handle_radius        = handle_radius,
            handle_friction      = handle_friction,
            is_physically_closed = is_phys_closed,
            gripper_width        = gripper_width,
            prev_angle           = prev_angle,
            control_freq         = self.cfg.control_freq,
            door_qpos            = door_qpos,
            latch_stiffness      = dr.current_latch_stiffness,
            base_latch_stiffness = dr.base_latch_stiffness or 1.0,
            wrist_align_ok       = wrist_align_ok,
            beta_probs           = None,
        )

        just_succeeded = (prev_phase == PHASE_PULL and self._fsm.state.phase == PHASE_HOLD_OPEN)

        # retreat_pos: fissato all'ingresso in RETREAT (specchio chiusura)
        if self._fsm.state.phase == PHASE_RETREAT and self._fsm.state.retreat_pos is None:
            self._fsm.state.retreat_pos = self._start_eef_pos.copy()
        dist_retreat = (
            float(np.linalg.norm(eef_pos - self._fsm.state.retreat_pos))
            if self._fsm.state.retreat_pos is not None else 1.0
        )

        reward, terminated, truncated, rew_info = self._reward_fn.compute(
            fsm_state      = self._fsm.state,
            phase_consts   = (PHASE_REACH, PHASE_PULL, PHASE_HOLD_OPEN, PHASE_RETREAT),
            door_angle     = door_angle,
            goal_angle     = self._goal_angle,
            door_min       = self._door_min,
            open_tol       = self.cfg.open_tol_rad,
            prev_angle     = prev_angle,
            gripper_action = float(action[-1]),
            grip_thresh    = self._fsm.grip_thresh(handle_friction),
            dist_handle    = dist_handle,
            dist_xy        = dist_xy,
            height_diff    = height_diff,
            dist_retreat   = dist_retreat,
            target_steps   = self._fsm.state.target_hold_steps or 30,
            curriculum_lvl = self.curriculum_level,
            is_physically_closed = is_phys_closed,
            action         = action,
            latch_qpos     = latch_qpos,
            just_succeeded = just_succeeded,
            rs_done        = bool(rs_done),
            step_count     = self._step_count,
            horizon        = self.cfg.horizon,
        )

        self._prev_door_angle = door_angle
        self._prev_gripper_width = float(gripper_width)
        self._prev_is_phys_closed = bool(is_phys_closed)

        # info ricca (stessa interfaccia dei test v2: nomi speculari)
        info: Dict[str, Any] = {}
        info["is_success"]        = self._fsm.state.phase in (PHASE_HOLD_OPEN, PHASE_RETREAT)
        info["door_angle"]        = door_angle
        info["door_qpos"]         = door_qpos
        info["goal_angle"]        = self._goal_angle
        info["open_error"]        = float(abs(self._goal_angle - door_angle))
        info["latch_qpos"]        = latch_qpos
        info["dist_handle"]       = dist_handle
        info["eef_pos"]           = [round(float(x),4) for x in np.asarray(eef_pos).ravel()[:3]]
        info["handle_pos"]        = [round(float(x),4) for x in np.asarray(handle_pos).ravel()[:3]]
        info["vec_eef_to_handle"] = [round(float(x),4) for x in (np.asarray(handle_pos).ravel()[:3] - np.asarray(eef_pos).ravel()[:3])]
        info["handle_src"]        = _handle_src
        info["obs_keys_sample"]   = self._obs_keys[:12]
        info["fsm_phase"]         = self._fsm.state.phase
        info["fsm_phase_name"]    = self._fsm.state.phase_name
        info["hold_duration"]     = self._fsm.state.hold_open_duration
        info["target_hold_steps"] = self._fsm.state.target_hold_steps
        info["curriculum_level"]  = self.curriculum_level
        info["fsm_events"]        = fsm_events
        info["reward_terms"]      = rew_info

        return self._flatten_obs(obs), reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            return self._rs_env.render()
        return None

    def close(self):
        self._rs_env.close()