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
        self.cfg = cfg
        self.gamma = float(gamma)
        self._prev_phi = 0.0
        self._max_door_angle = None   # ratchet di APERTURA (sale solo) per door_prog

    def reset(self):
        self._prev_phi = 0.0
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
        denom = max(1e-6, goal_angle - door_min)
        progress = float(np.clip((door_angle - door_min) / denom, 0.0, 1.0))
        w_eff = self.cfg.phi_pull_weight * (1.0 + self.cfg.curriculum_reward_k * curriculum_lvl)
        return float(w_eff * progress)

    def phi_hold(self, hold_duration: int, target_steps: int, door_angle: float,
                 goal_angle: float, open_tol: float) -> float:
        time_frac = float(np.clip(hold_duration / max(1, target_steps), 0.0, 1.0))
        # quanto è vicino al goal (1 al goal, →0 lontano)
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
        target_steps  : int,
        curriculum_lvl: float,
        is_physically_closed: bool,
        action        : np.ndarray,
        just_succeeded: bool,
        rs_done       : bool,
        step_count    : int,
        horizon       : int,
    ) -> Tuple[float, bool, bool, Dict[str, float]]:
        REACH, PULL, HOLD_OPEN, RETREAT = phase_consts
        rew: Dict[str, float] = {}

        # base: piccolo time-penalty (specchio della chiusura)
        rew["base"] = -0.10

        # ── potential-based shaping (Ng 1999) ──
        w_reach = self.cfg.phi_reach_weight * (1.0 + self.cfg.curriculum_reward_k * curriculum_lvl)
        w_pull  = self.cfg.phi_pull_weight  * (1.0 + self.cfg.curriculum_reward_k * curriculum_lvl)
        w_hold  = self.cfg.phi_hold_weight
        ph = fsm_state.phase
        if ph == REACH:
            # §1.9.F (mirror chiusura): Phi_reach = 0 in REACH per AZZERARE la penalità di
            # discount-decay F=(gamma-1)*Phi ~ -0.05*Phi, che vicino alla maniglia diventa
            # una "barriera di potenziale" e impedisce alla policy di RESTARE sulla maniglia
            # il tempo necessario a confermare la presa (5 step). Gli offset cumulativi nelle
            # fasi successive restano → la transizione REACH→PULL dà un bonus di grasp
            # naturale ~ +gamma*w_reach e la perdita presa una penalità ~ -w_reach [Ng 1999].
            phi_now = 0.0
        elif ph == PULL:
            phi_now = w_reach + self.phi_pull(door_angle, goal_angle, door_min,
                                              gripper_action, grip_thresh, curriculum_lvl)
        elif ph == HOLD_OPEN:
            phi_now = w_reach + w_pull + self.phi_hold(
                fsm_state.hold_open_duration, target_steps, door_angle, goal_angle, open_tol)
        else:  # RETREAT
            phi_now = w_reach + w_pull + w_hold + self.phi_retreat(dist_retreat)
        rew["phi_shape"] = self.gamma * phi_now - self._prev_phi
        self._prev_phi = phi_now

        # ── termini per-fase (in R genuino, non shaping) ──
        # FASE REACH — termini DENSI di avvicinamento (mirror della chiusura v2 che
        # funziona). CRUCIALE: con lo shaping a potenziale cumulativo, in REACH il
        # gradiente di Φ è ~nullo; questi termini densi sono l'UNICO segnale che porta
        # il braccio alla maniglia. Senza, la policy resta a mezza distanza (success=0).
        # Rif.: close_generalized_v2/reward_v2.py §1.10.B; competenza-del-task → shaping [3].
        if ph == REACH:
            k = 1.0 + self.cfg.curriculum_reward_k * curriculum_lvl
            rew["dist_3d"] = -self.cfg.w_reach_dist_3d * k * dist_handle
            if dist_xy is not None:
                rew["dist_xy"] = -self.cfg.w_reach_dist_xy * k * dist_xy
            if height_diff is not None:
                rew["dist_z"] = -self.cfg.w_reach_dist_z * k * abs(height_diff)
                # geometria di avvicinamento: non passare sotto, non stare troppo sopra
                if height_diff < -0.005:
                    rew["app_blw"] = -self.cfg.w_reach_app_blw * abs(height_diff + 0.005)
                if height_diff > 0.03:
                    rew["app_top"] = -self.cfg.w_reach_app_top * height_diff
            # gestione gripper calibrata sulla soglia di presa adattiva:
            # lontano → tieni aperto; vicino → premia la chiusura
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

            # ── OBIETTIVO REALE: progresso di APERTURA con ratchet (mirror door_prog) ──
            # delta = nuovo angolo guadagnato verso il goal (door_angle che CRESCE).
            # _max_door_angle sale solo → oscillare avanti/indietro non ri-premia (anti-exploit).
            # È il segnale genuino R che definisce "apri la porta"; lo shaping Ng sopra
            # non ne sposta l'ottimo [Ng 1999]. Rif. close §1.10.C (door_prog).
            if self._max_door_angle is None:
                self._max_door_angle = door_angle
            if gripper_action > grip_thresh:
                delta = door_angle - self._max_door_angle
                if delta > 0:
                    rew["door_prog"] = self.cfg.w_pull_progress * delta
                    self._max_door_angle = door_angle

            # presa genuinamente debole durante il PULL (dolce, come §1.13 chiusura)
            if gripper_action < grip_thresh:
                rew["grip"] = -self.cfg.w_pull_grip_weak * (grip_thresh - gripper_action)

            # mantenimento contatto durante l'apertura, scalato sul progresso (§1.16 specchio)
            if is_physically_closed:
                denom = max(1e-6, goal_angle - door_min)
                opening_progress = float(np.clip((door_angle - door_min) / denom, 0.0, 1.0))
                rew["grip_contact"] = self.cfg.w_grip_contact * opening_progress

        elif ph == HOLD_OPEN:
            rew["hold"] = 1.0
            if gripper_action > grip_thresh:
                rew["hold_grip"] = 1.0
            arm_norm = float(np.linalg.norm(action[:-1]))
            rew["hold_act"] = 1.0 if arm_norm < 0.05 else -2.0 * arm_norm

        elif ph == RETREAT:
            rew["hold"] = 1.0
            # penalizza la RICHIUSURA post-successo (specchio di w_door_regress della chiusura)
            regress = max(0.0, prev_angle - door_angle)   # door_angle che cala = si richiude
            rew["door_regress"] = -self.cfg.w_door_regress * regress

        # ── successo / terminazione ──
        if just_succeeded:
            rew["success_bonus"] = self.cfg.success_bonus

        terminated = False
        truncated = bool(rs_done) or (step_count >= horizon)

        # terminazione pulita: in RETREAT, mantenuto aperto + braccio rientrato
        if ph == RETREAT and fsm_state.retreat_steps >= self.cfg.fsm_retreat_target_steps:
            door_open_ok = door_angle >= goal_angle - open_tol
            if door_open_ok:
                terminated = True

        reward = float(np.clip(sum(rew.values()), -50.0, 50.0))
        return reward, terminated, truncated, rew
