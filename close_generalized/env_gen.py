#!/usr/bin/env python3
# close_generalized/env_gen.py

import numpy as np
from train_close import RoboSuiteDoorCloseGymnasiumEnv
from scipy.spatial.transform import Rotation as R_scipy

# ─────────────────────────────────────────────────────────────────────────────
# Parametri FSM
# ─────────────────────────────────────────────────────────────────────────────
_GRASP_CONFIRM_STEPS  = 2      # step consecutivi per confermare grasp (era 5: troppo per una policy fresh)
_GRASP_LOSE_STEPS     = 4      # step fuori lose_tol prima di resettare fase 2
_GRIPPER_CLOSE_THRESH = 0.65   # abbassato da 0.85: più accessibile, grad più forte verso chiusura
_GRIPPER_OPEN_THRESH  = -0.85  # forziamo un'apertura molto ampia per evitare collisioni
_APPROACH_HEIGHT_TOL  = 0.005  # tolleranza approccio (quasi nulla per evitare agganci)

# ─────────────────────────────────────────────────────────────────────────────
# Pesi per Reward
# ─────────────────────────────────────────────────────────────────────────────
### Fase 1
_W_REACH_3D       = 5.0   # reward lineare su dist_handle 3D
_W_REACH_XY       = 3.0   # componente orizzontale separata
_W_REACH_Z        = 15.0  # componente verticale fortificata per evitare scivolamenti
_W_LATERAL_ORI    = 1.5   # bonus allineamento Z in zona ravvicinata
_W_GRIPPER_OPEN   = 1.5   # reward per aprire il gripper durante approach
_W_GRIPPER_CLOSE  = 2.5   # aumentato da 1.0: gradiente più ripido verso chiusura completa
_W_PUSH_PENALTY   = 5.0   # penalità pressing dall'alto (eef alto + grip chiuso)
_W_APPROACH_BELOW = 3.0   # penalità approccio dal basso
_W_GRASP_BONUS    = 20.0  # bonus one-shot maggiorato per incentivare transizione a fase 2
### Fase 2
_W_GRASP_LOST     = 6.0     # penalità per grasp perso
_W_PROGRESS_GRASP = 2000.0  # [CARROT] premio immenso per spingere la porta, senza usare bastoni (penalità)
_W_ACTION_PHASE2  = 0.005   # penalità azione leggera in fase 2
_W_ACTION_PHASE1  = 0.0

class GeneralizedDoorEnv(RoboSuiteDoorCloseGymnasiumEnv):

    def __init__(self, cfg, render_mode=None):
        super().__init__(cfg, render_mode)

        self.curriculum_level = 0.0
        self.door_body_id     = self._rs_env.sim.model.body_name2id("Door_main")
        self.base_pos         = np.array(self._rs_env.sim.model.body_pos[self.door_body_id]).copy()
        self.base_quat        = np.array(self._rs_env.sim.model.body_quat[self.door_body_id]).copy()
        self.handle_geom_id   = None

        for i, name in enumerate(self._rs_env.sim.model.geom_names):
            if "handle" in name.lower():
                self.handle_geom_id = i
                break

        if self.handle_geom_id is not None:
            self.base_friction = self._rs_env.sim.model.geom_friction[self.handle_geom_id].copy()

        # FSM State
        self._grasp_phase         = False
        self._grasp_confirm_count = 0
        self._grasp_lose_count    = 0
        self._return_hold         = 0
        self._diag_step           = 0
        self._prev_door_angle     = None

    def set_curriculum_level(self, level: float):
        self.curriculum_level = np.clip(level, 0.0, 1.0)

    def _flatten_obs(self, obs):
        eef_pos    = obs.get("robot0_eef_pos", np.zeros(3))
        handle_pos = obs.get("handle_pos", obs.get("door_handle_pos", eef_pos))
        dist       = float(np.linalg.norm(handle_pos - eef_pos))

        # FSM state as one-hot encoding: [REACH, PUSH, HOLD, RETREAT]
        grasp_phase     = getattr(self, "_grasp_phase", False)
        success_latched = getattr(self, "_success_latched", False)
        ready_retreat   = getattr(self, "_ready_to_retreat", False)

        fsm_reach   = 1.0 if (not grasp_phase and not success_latched) else 0.0
        fsm_push    = 1.0 if (grasp_phase and not success_latched) else 0.0
        fsm_hold    = 1.0 if (success_latched and not ready_retreat) else 0.0
        fsm_retreat = 1.0 if (success_latched and ready_retreat) else 0.0

        # Door hinge angle (0.0 = fully closed, 0.4 = fully open)
        # Essential for the policy to predict the handle's moving arc during PUSH
        # NOTE: obs['hinge_qpos'] is always 0.0 in robosuite's cache; must read live from sim
        try:
            hinge_qpos = float(self._rs_env.sim.data.qpos[self._rs_env.hinge_qpos_addr])
        except Exception:
            hinge_qpos = float(obs.get("hinge_qpos", 0.0))

        custom = np.array([
            dist,                                            # EEF-to-handle distance
            getattr(self, "_current_handle_radius", 0.02),   # handle geometry
            getattr(self, "_current_handle_friction", 0.8),  # handle friction
            fsm_reach, fsm_push, fsm_hold, fsm_retreat,      # FSM one-hot
            hinge_qpos,                                      # door angle
        ], dtype=np.float32)

        base_flat = super()._flatten_obs(obs)
        return np.concatenate([base_flat, custom])

    def _calculate_reward(self, action, obs, rs_done, door_angle, prev_angle, just_succeeded):

        base_reward, terminated, truncated = super()._calculate_reward(
            action, obs, rs_done, door_angle, prev_angle, just_succeeded
        )

        eef_pos    = obs.get("robot0_eef_pos", np.zeros(3))
        handle_pos = obs.get("handle_pos", obs.get("door_handle_pos", eef_pos))

        dist_handle = np.linalg.norm(eef_pos - handle_pos)
        dist_xy     = np.linalg.norm(eef_pos[:2] - handle_pos[:2])
        height_diff = eef_pos[2] - handle_pos[2]

        door_qpos = self._rs_env.sim.data.qpos[self._rs_env.hinge_qpos_addr]
        is_closed = abs(door_qpos) < 0.03

        grip_tol       = getattr(self, "_current_handle_radius", 0.02) + 0.03
        gripper_action = action[-1] if action is not None else 0.0

        gripper_qpos = obs.get("robot0_gripper_qpos")
        if gripper_qpos is not None:
            gripper_width        = np.sum(np.abs(gripper_qpos))
            handle_diameter      = getattr(self, "_current_handle_radius", 0.02) * 2.0
            is_physically_closed = (gripper_width <= handle_diameter + 0.025) and (gripper_width >= 0.015)
        else:
            is_physically_closed = gripper_action > _GRIPPER_CLOSE_THRESH
            gripper_width        = 0.0

        # !!! Allineamento pre-calcolato
        alignment      = 0.0
        flat_alignment = 0.0
        eef_quat       = obs.get("robot0_eef_quat")

        if eef_quat is not None:
            delta_pos     = handle_pos - eef_pos
            norm_delta    = np.linalg.norm(delta_pos)
            if norm_delta > 0:
                dir_to_handle = delta_pos / norm_delta
                rmat           = R_scipy.from_quat(eef_quat).as_matrix()
                eef_z          = rmat[:, 2]
                eef_x          = rmat[:, 0]
                alignment      = abs(np.dot(eef_z, dir_to_handle))
                flat_alignment = abs(eef_x[2])

        reward   = 0.0
        rew_info = {}

        # ══════════════════════════════════════════════════════════════════════
        # FASE 1 — REACH & GRASP
        # ══════════════════════════════════════════════════════════════════════
        if not self._grasp_phase and not self._success_latched:
            rew_info["base"] = base_reward - _W_GRIPPER_CLOSE

            # Segnale di distanza
            rew_info["dist_3d"]  = -_W_REACH_3D * dist_handle
            rew_info["dist_xy"]  = -_W_REACH_XY * dist_xy
            rew_info["dist_z"]   = -_W_REACH_Z  * abs(height_diff)

            # Penalità geometriche di approccio
            if height_diff < -_APPROACH_HEIGHT_TOL:
                rew_info["app_blw"] = -_W_APPROACH_BELOW * abs(height_diff + _APPROACH_HEIGHT_TOL)

            if height_diff > 0.03 and gripper_action > 0.2:
                rew_info["app_top"] = -_W_PUSH_PENALTY * height_diff * gripper_action

            # Allineamento: scalato morbidamente in base alla distanza
            prox_factor = np.exp(-10.0 * dist_handle)
            if dist_handle < grip_tol * 3.0:
                rew_info["align"] = -1.0 * (1.0 - alignment) * prox_factor
                rew_info["flat"]  = -0.5 * flat_alignment    * prox_factor

            # Gestione Gripper
            if dist_handle > 0.025:
                # Lontano: puniamo se chiude (smussato a 1.0 da 5.0)
                if gripper_action > _GRIPPER_OPEN_THRESH:
                    rew_info["grip"] = -1.0 * (gripper_action - _GRIPPER_OPEN_THRESH)
                self._grasp_confirm_count = 0
            else:
                # Vicino: bonus morbido se chiude (ripreso originale, niente cliff)
                if gripper_action > _GRIPPER_OPEN_THRESH:
                    rew_info["grip"] = _W_GRIPPER_CLOSE * ((gripper_action - _GRIPPER_OPEN_THRESH) / (1.0 - _GRIPPER_OPEN_THRESH))

                if gripper_action > _GRIPPER_CLOSE_THRESH and is_physically_closed and dist_handle < 0.025:
                    self._grasp_confirm_count += 1
                else:
                    self._grasp_confirm_count = 0

                if self._grasp_confirm_count >= _GRASP_CONFIRM_STEPS:
                    self._grasp_phase      = True
                    self._grasp_lose_count = 0
                    if not getattr(self, "_has_received_grasp_bonus", False):
                        rew_info["phase_trans"]        = _W_GRASP_BONUS
                        self._has_received_grasp_bonus = True

                    rew_info["base"] = base_reward

        # ══════════════════════════════════════════════════════════════════════
        # FASE 2 — PUSH TO CLOSE
        # ══════════════════════════════════════════════════════════════════════
        elif self._grasp_phase and not self._success_latched:
            rew_info["base"]    = base_reward
            rew_info["dist_3d"] = -5.0 * dist_handle
            rew_info["dist_z"]  = -15.0 * abs(height_diff)

            if self._min_door_angle is None:
                self._min_door_angle = door_angle

            if gripper_action > _GRIPPER_CLOSE_THRESH:
                door_progress = self._min_door_angle - door_angle
                if door_progress > 0:
                    rew_info["door_prog"] = _W_PROGRESS_GRASP * door_progress
                    self._min_door_angle  = door_angle

            if action is not None:
                rew_info["act_pen"] = -_W_ACTION_PHASE2 * np.linalg.norm(action[:-1])

            door_moving         = (prev_angle is not None and prev_angle - door_angle > 0.001)
            effective_lose_tol  = 0.05 if door_moving else 0.04

            gripper_action_lost = gripper_action < _GRIPPER_CLOSE_THRESH
            gripper_lost        = gripper_action_lost or not is_physically_closed
            distance_lost       = dist_handle > effective_lose_tol

            if gripper_lost or distance_lost:
                if distance_lost:
                    rew_info["dist_lost"] = -_W_GRASP_LOST * (dist_handle - effective_lose_tol)
                if gripper_lost:
                    rew_info["grip_lost"] = -5.0 * abs(min(0.0, gripper_action) - _GRIPPER_CLOSE_THRESH)
                self._grasp_phase         = False
                self._grasp_confirm_count = 0
                self._grasp_lose_count    = 0
            else:
                self._grasp_lose_count = 0
                if gripper_action < 1.0:
                    rew_info["grip"] = -5.0 * (1.0 - gripper_action)
                if gripper_action > _GRIPPER_CLOSE_THRESH:
                    rew_info["grip_hold"] = 2.0

            if action is not None and action[2] > 0.05:
                rew_info["lift_pen"] = -2.0 * action[2]

        # ══════════════════════════════════════════════════════════════════════
        # FASE 3 — RELEASE & HOLD
        # ══════════════════════════════════════════════════════════════════════
        elif self._success_latched:
            rew_info["base"] = base_reward
            rew_info["hold"] = 0.0

            if not getattr(self, "_ready_to_retreat", False):
                # Bisogna penalizzare severamente se perde la presa fisica corretta SEMPRE
                if not is_physically_closed:
                    rew_info["hold_slip"] = -5.0
                
                # Bisogna penalizzare severamente se la porta riapre
                if not is_closed:
                    rew_info["hold_bounce"] = -20.0 * abs(door_qpos)
                    self._hold_closed_duration = 0

            if is_closed:
                rew_info["hold"] += 1.0 - abs(door_qpos)
                if abs(door_qpos) < 0.04:
                    control_freq = self.cfg.control_freq
                    target_hold_steps = int(control_freq * 2.0)

                    if not hasattr(self, "_hold_closed_duration"):
                        self._hold_closed_duration = 0

                    if self._hold_closed_duration < target_hold_steps:
                        self._hold_closed_duration += 1
                        self._ready_to_retreat     = False

                        if gripper_action > _GRIPPER_CLOSE_THRESH:
                            rew_info["hold_grip"] = 1.0
                        else:
                            rew_info["hold_grip"] = -2.0 * abs(gripper_action - _GRIPPER_CLOSE_THRESH)

                        if gripper_action < 0.0:
                            rew_info["hold_drop_pen"] = -10.0 * abs(gripper_action)

                        if action is not None:
                            action_norm = np.linalg.norm(action[:-1])
                            if action_norm < 0.05:
                                rew_info["hold_act"] = 1.0
                            else:
                                rew_info["hold_act"] = -5.0 * action_norm

                        rew_info["hold_flat"] = -5.0 * flat_alignment
                    else:
                        if not getattr(self, "_ready_to_retreat", False):
                            self._ready_to_retreat = True
                            self._retreat_pos = eef_pos + np.array([-0.20, 0.0, 0.05], dtype=np.float32)

                        if gripper_action < _GRIPPER_OPEN_THRESH:
                            rew_info["ret_grip"] = 2.0
                        else:
                            rew_info["ret_grip"] = -1.0 * abs(gripper_action + 1.0)

                        if action is not None:
                            rew_info["ret_act"] = -1.0 * np.linalg.norm(action[:-1])

                            # REGOLARIZZAZIONE ROTAZIONALE: nessuna torsione del polso
                            rew_info["ret_rot"] = -10.0 * np.linalg.norm(action[3:6])

                            if gripper_action > -0.8:
                                rew_info["ret_early_move"] = -10.0 * np.linalg.norm(action[:-1])

                            if dist_handle < 0.12:
                                rew_info["ret_lat"] = -5.0 * abs(action[1])
                                if action[2] < 0:
                                    rew_info["ret_down"] = -5.0 * abs(action[2])

                            # BLOCCO TOTALE A FINE RETREAT  # TODO: QUESTO BLOCCO E' SOLO DEL POLSO? OPPURE DI TUTTI I GIUNTI?
                            if hasattr(self, "_retreat_pos"):
                                dist_to_target = float(np.linalg.norm(eef_pos - self._retreat_pos))
                                if dist_to_target < 0.05:
                                    rew_info["ret_freeze"] = -20.0 * np.linalg.norm(action[:-1])

                        # Reward for latch joint returning to rest (0.0 rad)
                        latch_qpos = self._rs_env.sim.data.qpos[self._rs_env.handle_qpos_addr]
                        rew_info["latch_ret"] = -1.0 * abs(latch_qpos)
            else:
                if not getattr(self, "_ready_to_retreat", False):
                    self._hold_closed_duration = 0

        # Calcolo Totale
        for v in rew_info.values():
            reward += v

        # Logging diagnostico ogni 200 step
        self._diag_step += 1
        if self._diag_step % 200 == 0:
            if self._success_latched:
                phase = "4:BACK" if getattr(self, "_ready_to_retreat", False) else "3:HOLD"
            elif self._grasp_phase:
                phase = "2:PUSH"
            else:
                phase = "1:REACH"
            phys_status = "PHYS_OK" if is_physically_closed else "PHYS_OPEN"

            gripper_w_str = f"{gripper_width:.3f}" if gripper_qpos is not None else "N/A"

            print(f"┌─────────┬────────┬────────┬───────┬───────────┬───────┬───────┬───────┬───────┐")
            print(f"│  PHASE  │  DIST  │   dZ   │ GRIP  │   PHYS    │ WIDTH │ ALIGN │ DOOR  │ CONF  │")
            print(f"├─────────┼────────┼────────┼───────┼───────────┼───────┼───────┼───────┼───────┤")
            print(f"│ {phase:<7} │ {dist_handle:>6.3f} │ {height_diff:>+6.3f} │ {gripper_action:>+5.2f} │ {phys_status:<9} │ {gripper_w_str:>5} │ {alignment:>5.2f} │ {door_angle:>5.2f} │  {self._grasp_confirm_count}/{_GRASP_CONFIRM_STEPS}  │")
            print(f"└─────────┴────────┴────────┴───────┴───────────┴───────┴───────┴───────┴───────┘")

            rew_str = " │ ".join([f"{k}: {v:>+5.2f}" for k, v in rew_info.items() if abs(v) > 0.001])
            print(f"  ↳ REWARDS │ {rew_str} │ TOT: {reward:>+6.2f}\n")

        return reward, terminated, truncated


    def reset(self, seed: int = None, options: dict = None):
        self._grasp_phase              = False
        self._grasp_confirm_count      = 0
        self._grasp_lose_count         = 0
        self._return_hold              = 0
        self._hold_closed_duration     = 0
        self._ready_to_retreat         = False
        self._prev_door_angle          = None
        self._min_door_angle           = None
        self._has_received_grasp_bonus = False

        if getattr(self, "handle_geom_id", None) is not None:
            base_radius = 0.02
            base_length = 0.08

            r_scale = np.random.uniform(0.7, 1.4)
            l_scale = np.random.uniform(0.8, 1.2)
            self._current_handle_radius = base_radius * r_scale
            if self._rs_env.sim.model.geom_size[self.handle_geom_id] is not None:
                self._rs_env.sim.model.geom_size[self.handle_geom_id][0] = self._current_handle_radius
                self._rs_env.sim.model.geom_size[self.handle_geom_id][1] = base_length * l_scale

            # Bidirectional friction randomization [Problem 0]
            # Range 0.3–1.2× teaches the policy to adapt grip force to slippery handles
            f_scale = np.random.uniform(0.3, 1.2)
            base_f  = getattr(self, "base_friction", np.array([0.8]))[0]
            new_f   = float(np.clip(base_f * f_scale, 0.05, 2.0))

            self._rs_env.sim.model.geom_friction[self.handle_geom_id][0] = new_f
            self._current_handle_friction = new_f
        else:
            self._current_handle_radius   = 0.02
            self._current_handle_friction = 0.8

        p_var = 0.15 * self.curriculum_level
        r_var = 0.30 * self.curriculum_level

        if self.curriculum_level > 0:
            pos_offset    = np.random.uniform(-p_var, p_var, size=3)
            pos_offset[2] = 0
            yaw           = np.random.uniform(-r_var, r_var)
            q_scipy       = R_scipy.from_euler('z', yaw).as_quat()

            self._rs_env.sim.model.body_pos[self.door_body_id] = self.base_pos + pos_offset

            if hasattr(self.cfg, "human_dist_min") and hasattr(self.cfg, "human_dist_max"):
                dist_sample = np.random.uniform(self.cfg.human_dist_min, self.cfg.human_dist_max)
                self._rs_env.sim.model.body_pos[self.door_body_id][0] = dist_sample + pos_offset[0]

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