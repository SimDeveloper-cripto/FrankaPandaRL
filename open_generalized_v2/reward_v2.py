#!/usr/bin/env python3
# open_generalized_v2/reward_v2.py
#
# PotentialBasedRewardOpen — reward a potenziale gerarchico per l'APERTURA generalizzata,
# SPECULARE a close_generalized_v2/reward_v2.py.
#
# Principio (Ng, Harada & Russell 1999, [3]): lo shaping è F = γ·Φ(s') − Φ(s), che NON
# altera la politica ottima (policy invariance). Il potenziale Φ cresce monotòno lungo le
# fasi REACH→PULL→HOLD_OPEN→RETREAT, così lo shaping "tira" verso il completamento del task
# senza introdurre ottimi spuri (lezione metodologica della v2: motion-quality → env-level,
# competenza del task → reward potential-based).
#
# Inversione chiusura ↔ apertura: il potenziale di fase PULL premia il progresso verso
# l'angolo-OBIETTIVO (apertura), non verso 0 (chiusura).
#
# Riferimenti: [3] Ng 1999 (shaping invariante), [13] ManipForce (contatto), [15] ten Pas.

from __future__ import annotations

import numpy as np
from typing import Dict, Tuple


class PotentialBasedRewardOpen:
    def __init__(self, cfg, gamma: float = 0.95):
        self.cfg             = cfg
        self.gamma           = float(gamma)
        self._prev_phi       = 0.0
        self._max_door_angle = None   # ratchet di APERTURA (sale solo) per door_prog

    def reset(self):
        self._prev_phi       = 0.0
        self._max_door_angle = None

    # ── Potenziali di fase ───────────────────────────────────────────────────────

    def phi_reach(self, dist_handle: float, handle_radius: float, curriculum_lvl: float) -> float:
        sigma = float(np.clip(handle_radius * 3.0, self.cfg.phi_reach_sigma * 0.25, self.cfg.phi_reach_sigma))
        w_eff = self.cfg.phi_reach_weight * (1.0 + self.cfg.curriculum_reward_k * curriculum_lvl)
        return float(w_eff * np.exp(-(dist_handle ** 2) / (2.0 * sigma ** 2)))

    def phi_pull(self, door_angle: float, goal_angle: float, door_min: float,
                 gripper_action: float, grip_thresh: float, curriculum_lvl: float) -> float:
        """
        Progresso di APERTURA verso il goal, in [0,1] (specchio di phi_push della chiusura).
        0 a porta tutta chiusa (door_min), 1 a porta = goal_angle.
        """
        denom    = max(1e-6, goal_angle - door_min)
        progress = float(np.clip((door_angle - door_min) / denom, 0.0, 1.0))
        w_eff    = self.cfg.phi_pull_weight * (1.0 + self.cfg.curriculum_reward_k * curriculum_lvl)
        return float(w_eff * progress)

    def phi_hold(self, hold_duration: int, target_steps: int, door_angle: float,
                 goal_angle: float, open_tol: float) -> float:
        time_frac = float(np.clip(hold_duration / max(1, target_steps), 0.0, 1.0))
        open_frac = float(np.clip(1.0 - abs(goal_angle - door_angle) / max(open_tol * 5.0, 1e-6), 0.0, 1.0))
        return float(self.cfg.phi_hold_weight * time_frac * open_frac)

    def phi_retreat(self, dist_retreat: float) -> float:
        progress = float(np.clip(1.0 - dist_retreat / 0.20, 0.0, 1.0))
        return float(self.cfg.phi_retreat_weight * progress)

    # ── Compute ──────────────────────────────────────────────────────────────────

    def compute(
        self,
        *,
        fsm_state,
        phase_consts,            # (REACH, PULL, HOLD_OPEN, RETREAT)
        door_angle    : float,
        door_qvel     : float = 0.0,
        goal_angle    : float,
        door_min      : float,
        open_tol      : float,
        prev_angle    : float,
        gripper_action: float,
        grip_thresh   : float,
        dist_handle   : float,
        dist_xy       : float = None,
        height_diff   : float = None,
        dist_retreat  : float,
        eef_pos       : np.ndarray = None,
        target_steps  : int,
        curriculum_lvl: float,
        is_physically_closed: bool,
        action        : np.ndarray,
        latch_qpos    : float = 0.0,
        just_succeeded: bool,
        rs_done       : bool,
        step_count    : int,
        horizon       : int,
    ) -> Tuple[float, bool, bool, Dict[str, float]]:
        REACH, PULL, HOLD_OPEN, RETREAT = phase_consts
        rew: Dict[str, float]           = {}
        terminated                      = False

        rew["base"] = -0.10

        # ── potential-based shaping (Ng 1999) ──
        w_reach = self.cfg.phi_reach_weight * (1.0 + self.cfg.curriculum_reward_k * curriculum_lvl)
        w_pull  = self.cfg.phi_pull_weight  * (1.0 + self.cfg.curriculum_reward_k * curriculum_lvl)
        w_hold  = self.cfg.phi_hold_weight

        ph = fsm_state.phase
        if ph == REACH:
            phi_now = 0.0
        elif ph == PULL:
            phi_now = w_reach + self.phi_pull(door_angle, goal_angle, door_min, gripper_action, grip_thresh, curriculum_lvl)
        elif ph == HOLD_OPEN:
            phi_now = w_reach + w_pull + self.phi_hold(fsm_state.hold_open_duration, target_steps, door_angle, goal_angle, open_tol)
        else:  # RETREAT
            phi_now = w_reach + w_pull + w_hold + self.phi_retreat(dist_retreat)
        rew["phi_shape"] = self.gamma * phi_now - self._prev_phi
        self._prev_phi   = phi_now

        # ── termini per-fase (in R genuino, non shaping) ──
        # Rif.: close_generalized_v2/reward_v2.py §1.10.B; competenza-del-task → shaping [3].
        if ph == REACH:
            k = 1.0 + self.cfg.curriculum_reward_k * curriculum_lvl
            rew["dist_3d"] = -self.cfg.w_reach_dist_3d * k * dist_handle
            if dist_xy is not None:
                rew["dist_xy"] = -self.cfg.w_reach_dist_xy * k * dist_xy
            if height_diff is not None:
                rew["dist_z"] = -self.cfg.w_reach_dist_z * k * abs(height_diff)
                if height_diff < -0.005:
                    rew["app_blw"] = -self.cfg.w_reach_app_blw * abs(height_diff + 0.005)
                if height_diff > 0.03:
                    rew["app_top"] = -self.cfg.w_reach_app_top * height_diff
            d_near = self.cfg.fsm_grasp_dist_k_radius * 0.02 + self.cfg.fsm_grasp_dist_k_offset
            if dist_handle > d_near:
                if gripper_action > -0.85:
                    rew["grip"] = -1.0 * (gripper_action - (-0.85))
            else:
                if gripper_action > -0.85:
                    norm_g = (gripper_action - (-0.85)) / (1.0 - (-0.85))
                    rew["grip"] = self.cfg.w_reach_grip_near * norm_g

        if ph == PULL:
            # mantieni la maniglia mentre tiri (mirror dist_3d/dist_z della chiusura)
            rew["dist_3d"] = -self.cfg.w_pull_dist_3d * dist_handle
            if height_diff is not None:
                rew["dist_z"] = -self.cfg.w_pull_dist_z * abs(height_diff)

            prog_angle = door_angle
            if getattr(self.cfg, "pull_progress_cap_at_goal", False):
                prog_angle = min(door_angle, goal_angle)
            if self._max_door_angle is None:
                self._max_door_angle = prog_angle
            if gripper_action > grip_thresh:
                delta = prog_angle - self._max_door_angle
                if delta > 0:
                    rew["door_prog"] = self.cfg.w_pull_progress * delta
                    self._max_door_angle = prog_angle

            # presa genuinamente debole durante il PULL
            if gripper_action < grip_thresh:
                rew["grip"] = -self.cfg.w_pull_grip_weak * (grip_thresh - gripper_action)

            # mantenimento contatto durante l'apertura, scalato sul progresso
            if is_physically_closed:
                denom               = max(1e-6, goal_angle - door_min)
                opening_progress    = float(np.clip((door_angle - door_min) / denom, 0.0, 1.0))
                rew["grip_contact"] = self.cfg.w_grip_contact * opening_progress

        elif ph == HOLD_OPEN:
            open_err   = abs(goal_angle - door_angle)
            is_open_ok = open_err < open_tol

            if is_open_ok:
                rew["hold"] = 1.0
            else:
                rew["hold"] = -1.0 * open_err
            if not is_physically_closed:
                rew["hold_slip"] = -self.cfg.w_hold_slip
            if gripper_action > grip_thresh:
                rew["hold_grip"] = 1.0
            else:
                rew["hold_grip"] = -2.0 * abs(gripper_action - grip_thresh)
            if gripper_action < 0.0:
                rew["hold_drop_pen"] = -self.cfg.w_hold_drop_pen * abs(gripper_action)
            arm_norm = float(np.linalg.norm(action[:-1]))
            rew["hold_act"] = 1.0 if arm_norm < 0.05 else -2.0 * arm_norm
            if dist_handle > 0.06:
                rew["hold_dist"] = -self.cfg.w_hold_dist * (dist_handle - 0.06)

        elif ph == RETREAT:
            open_err     = abs(goal_angle - door_angle)
            door_open_ok = door_angle >= goal_angle - open_tol

            if getattr(fsm_state, "retreat_pos", None) is not None and eef_pos is not None:
                _ep = np.asarray(eef_pos, dtype=float)
                dist_to_target = float(np.linalg.norm(_ep - fsm_state.retreat_pos))
            else:
                dist_to_target = 0.20

            _restoring = bool(getattr(fsm_state, "retreat_restoring", False))
            if not _restoring:
                if gripper_action < -0.85:
                    rew["ret_grip"] = 2.0
                else:
                    rew["ret_grip"] = -1.0 * abs(gripper_action + 1.0)

            # Torsione del polso  [close: ret_rot]
            rew["ret_rot"] = -3.0 * float(np.linalg.norm(action[3:6]))

            if not _restoring and getattr(fsm_state, "retreat_pos", None) is not None and eef_pos is not None:
                if dist_to_target > self.cfg.fsm_retreat_settle_dist:
                    dir_to_target    = fsm_state.retreat_pos - _ep
                    dir_norm         = dir_to_target / (dist_to_target + 1e-6)
                    action_alignment = float(np.dot(np.asarray(action[:3], dtype=float), dir_norm))
                    rew["ret_dir"]   = 3.0 * action_alignment
                    perp             = np.asarray(action[:3], dtype=float) - action_alignment * dir_norm
                    rew["ret_perp"]  = -2.0 * float(np.linalg.norm(perp))
                else:
                    rew["ret_freeze"] = -self.cfg.w_retreat_settle * float(np.linalg.norm(action[:-1]))
                    if door_open_ok and gripper_action < -0.85:
                        rew["ret_release"] = 1.0

            rew["latch_ret"] = -self.cfg.w_latch_ret * abs(latch_qpos)
            rew["hold"]      = 1.0 if open_err < open_tol else 0.0
            if door_angle < goal_angle - open_tol:
                regress             = max(0.0, prev_angle - door_angle)
                rew["door_regress"] = -self.cfg.w_door_regress * regress

        # ── successo / terminazione ──
        if just_succeeded:
            rew["success_bonus"] = self.cfg.success_bonus

        truncated = bool(rs_done) or (step_count >= horizon)

        if getattr(self.cfg, "terminate_on_retreat_complete", True) and ph == RETREAT:
            door_open_ok_t = door_angle >= goal_angle - open_tol
            free_steps     = int(getattr(fsm_state, "retreat_free_steps", 0))
            min_release    = free_steps >= self.cfg.fsm_retreat_target_steps
            latch_home     = abs(latch_qpos) < getattr(self.cfg, "retreat_latch_term_tol", 0.08)
            hardcap        = fsm_state.retreat_steps >= getattr(self.cfg, "retreat_hard_cap", 120)
            exo_exit       = free_steps >= int(getattr(self.cfg, "retreat_exo_exit_steps", 60))

            if (min_release and latch_home) or exo_exit or hardcap:
                terminated = True
                if door_open_ok_t and (min_release and latch_home):
                    rew["success_bonus"] = self.cfg.success_bonus

        reward = float(np.clip(sum(rew.values()), -50.0, 50.0))
        return reward, terminated, truncated, rew