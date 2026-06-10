#!/usr/bin/env python3
# close_generalized_v2/reward_v2.py
#
# PotentialBasedReward — Hierarchical Potential-Based Reward Shaping
#
# Implements proposal:
#   "Reward Potential-Based (Teoricamente Fondato)"
#
# Key literature:
#   [3]  Ng, Russell & Harada (1999) "Policy Invariance Under Reward Transformations"
#        → F(s,a,s') = γΦ(s') − Φ(s) preserves optimal policy
#   [4]  Devlin & Kudenko (2012) "Dynamic Potential-Based Reward Shaping"
#        → Dynamic weights valid if weights converge; used for curriculum co-evolution
#   [16] Krakovna et al. (2020) "Avoiding Side Effects in Complex Environments"
#        → Reward misspecification: motivation for principled shaping
#
# Differences vs close_generalized/env_gen.py (v1):
#   ┌─────────────────────────────────┬──────────────────────────┬─────────────────────────────────────┐
#   │ Aspect                          │ v1 (ad-hoc)              │ v2 (potential-based)                │
#   ├─────────────────────────────────┼──────────────────────────┼─────────────────────────────────────┤
#   │ Reward continuity at transition │ Discontinuous (cliff)    │ Continuous (Φ → 0 at boundaries)    │
#   │ door_prog                       │ 2000 × Δangle (increm.)  │ γΦ_push(s') − Φ_push(s) (monotone)  │
#   │ Grasp bonus                     │ +50 (hard cliff)         │ Smooth via sigmoid in Φ_push        │
#   │ Physics calibration             │ No (fixed weights)       │ Yes (σ_reach = f(handle_radius))    │
#   │ Theoretical guarantee           │ None                     │ Optimal policy preserved [Ng 1999]  │
#   │ Curriculum co-evolution         │ No                       │ Yes [Devlin & Kudenko 2012]         │
#   └─────────────────────────────────┴──────────────────────────┴─────────────────────────────────────┘

from __future__ import annotations

import numpy as np
from typing import Optional

from close_generalized_v2.fsm_v2 import (PHASE_REACH, PHASE_PUSH, PHASE_HOLD, PHASE_RETREAT, FSMState)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-float(x)))


class PotentialBasedReward:
    """
    Computes reward as F(s, a, s') = γΦ(s') − Φ(s) + sparse_terms.

    The potential Φ(s) is defined hierarchically, one component per FSM phase:

        Φ(s) = Φ_reach(s) ×   [phase=REACH]
             + Φ_push(s)  ×   [phase=PUSH]
             + Φ_hold(s)  ×   [phase=HOLD]
             + Φ_retreat(s) × [phase=RETREAT]

    Each Φ_i is non-negative and bounded, so the shaping reward F is bounded
    and the optimal policy of the true MDP is preserved  [Ng et al. 1999].

    Dynamic weights (co-evolve with curriculum_level) are valid because they
    converge as curriculum_level → 1.0  [Devlin & Kudenko 2012].

    References
    ----------
    [3]  Ng, Russell & Harada (1999)
    [4]  Devlin & Kudenko (2012)
    [16] Krakovna et al. (2020) — reward misspecification motivation
    """

    def __init__(self, cfg, gamma: float = 0.95):
        self.cfg   = cfg
        self.gamma = gamma
        self._prev_phi: Optional[float] = None
        # Reward-owned ratchet for the PUSH door-progress reward.
        # MUST be independent of fsm_state.min_door_angle: the FSM updates its own
        # min BEFORE the reward runs (env_v2 calls fsm.update() then reward.compute()),
        # so reading the FSM's min would always give delta=0. See §1.10.C.
        self._min_door_angle: Optional[float] = None

    def reset(self) -> None:
        self._prev_phi       = None
        self._min_door_angle = None

    # ── Potential components ──────────────────────────────────────────────────

    def phi_reach(
        self,
        dist_handle   : float,
        handle_radius : float,
        curriculum_lvl: float,
    ) -> float:
        """
        §3.2 — REACH potential.

            Φ_reach(s) = w_reach_eff × exp(−dist / σ)

        where σ = max(handle_radius × 3, 0.08)  (physics-calibrated scale).
        w_reach_eff = w_reach × (1 + k_curr × curriculum_lvl)   [Devlin 2012]

        Properties:
        - Φ_reach ∈ [0, w_reach_eff]
        - Φ_reach → 0 as dist → ∞  (no reward far from handle)
        - Φ_reach → w_reach_eff as dist → 0  (maximum at grasp)
        - Auto-normalises to handle size: large handle → larger σ → softer gradient
        """
        sigma   = float(np.clip(handle_radius * 3.0, self.cfg.phi_reach_sigma * 0.25, self.cfg.phi_reach_sigma))
        w_eff   = self.cfg.phi_reach_weight * (
            1.0 + self.cfg.curriculum_reward_k * curriculum_lvl
        )
        return w_eff * float(np.exp(-dist_handle / sigma))

    def phi_push(
        self,
        door_angle     : float,
        door_max       : float,
        gripper_action : float,
        grip_thresh    : float,
        curriculum_lvl : float,
    ) -> float:
        """
        §3.2 — PUSH potential.

            Φ_push(s) = w_push × (door_max − door_angle) / door_max × grip_factor

        where grip_factor = sigmoid(10 × (gripper_action − grip_thresh)).

        Properties:
        - Monotonically increasing as door closes (door_angle decreases)
        - Continuously smoothed via sigmoid for grip factor (no cliff like v1's +50 bonus)
        - Naturally calibrated to door_max (independent of scale)

        v1 equivalent: 2000 × Δangle  (incremental, prone to oscillation)
        """
        closure   = float(np.clip((door_max - door_angle) / (door_max + 1e-8), 0.0, 1.0))
        grip_f    = _sigmoid(10.0 * (gripper_action - grip_thresh))
        w_eff     = self.cfg.phi_push_weight * (
            1.0 + self.cfg.curriculum_reward_k * curriculum_lvl
        )
        return w_eff * closure * grip_f

    def phi_hold(
        self,
        hold_duration  : int,
        target_steps   : int,
        door_qpos      : float,
        tol_closed     : float = 0.04,
    ) -> float:
        """
        §3.2 — HOLD potential.

            Φ_hold(s) = w_hold × (duration / target) × (1 − |door_qpos| / tol)

        Properties:
        - Grows linearly with hold progress
        - Penalises implicitly if door re-opens (|door_qpos| increases → Φ decreases)
        - Bounded in [0, w_hold]
        """
        time_frac = float(np.clip(hold_duration / max(1, target_steps), 0.0, 1.0))
        door_frac = float(np.clip(1.0 - abs(door_qpos) / tol_closed, 0.0, 1.0))
        return self.cfg.phi_hold_weight * time_frac * door_frac

    def phi_retreat(
        self,
        dist_to_target: float,
        max_dist      : float = 0.20,
    ) -> float:
        """
        §3.2 — RETREAT potential.

            Φ_retreat(s) = w_retreat × (1 − dist_to_target / max_dist)

        Grows as EEF approaches retreat target.
        """
        progress = float(np.clip(1.0 - dist_to_target / max(max_dist, 1e-6), 0.0, 1.0))
        return self.cfg.phi_retreat_weight * progress

    # ── Main Reward Computation ────────────────────────────────────────────────

    def compute(
        self,
        *,
        fsm_state       : FSMState,
        base_reward     : float,
        door_angle      : float,
        door_max        : float,
        door_qpos       : float,
        dist_handle     : float,
        dist_xy         : float,
        height_diff     : float,
        handle_radius   : float,
        handle_friction : float,
        grip_thresh     : float,
        gripper_action  : float,
        gripper_width   : float,
        is_physically_closed: bool,
        gripper_qpos    : Optional[np.ndarray],
        alignment       : float,
        flat_alignment  : float,
        joint_vel       : Optional[np.ndarray],
        action          : np.ndarray,
        prev_eef_action : np.ndarray,
        eef_pos         : np.ndarray,
        latch_qpos      : float,
        door_qvel       : float,
        curriculum_lvl  : float,
        just_grasped    : bool = False,
        just_succeeded  : bool = False,
        just_hold_done  : bool = False,
        grasp_lost      : bool = False,
        terminated      : bool = False,
        truncated       : bool = False,
    ) -> tuple[float, bool, bool]:
        """
        Returns
        -------
        (reward, terminated, truncated)
        """
        rew_info: dict[str, float] = {}
        phase = fsm_state.phase

        # Jerk penalty: regularises action variation
        jerk = float(np.linalg.norm(action[:-1] - prev_eef_action))
        rew_info["smoothness"] = -self.cfg.w_smoothness * jerk

        # ── Base reward from parent env
        rew_info["base"] = base_reward


        if self.cfg.use_potential_reward:
            target_steps = fsm_state.target_hold_steps or 60
            dist_retreat = float(np.linalg.norm(eef_pos - fsm_state.retreat_pos)) \
                if fsm_state.retreat_pos is not None else 0.20

            # Cumulative base weights to guarantee continuity across transitions
            w_reach_eff = self.cfg.phi_reach_weight * (1.0 + self.cfg.curriculum_reward_k * curriculum_lvl)
            w_push_eff  = self.cfg.phi_push_weight  * (1.0 + self.cfg.curriculum_reward_k * curriculum_lvl)
            w_hold_eff  = self.cfg.phi_hold_weight

            if phase == PHASE_REACH:
                phi_now = 0.0
            elif phase == PHASE_PUSH:
                phi_now = w_reach_eff + self.phi_push(door_angle, door_max, gripper_action, grip_thresh, curriculum_lvl)
            elif phase == PHASE_HOLD:
                phi_now = w_reach_eff + w_push_eff + self.phi_hold(fsm_state.hold_closed_duration, target_steps, door_qpos)
            else:  # RETREAT
                phi_now = w_reach_eff + w_push_eff + w_hold_eff + self.phi_retreat(dist_retreat)

            if self._prev_phi is not None:
                # Ng et al. (1999) shaping, kept EXACT at the MDP's gamma so the
                # optimal policy of the true reward R is provably preserved.
                #
                # The previous v2 made this term lethal NOT because of gamma, but
                # because the cumulative offsets made Phi huge (~25..100). The
                # discount term (gamma-1)*Phi then imposed a standing penalty of
                # -0.05*Phi per step (up to -5/step in the FROZEN hold phase),
                # punishing the agent for *being* in PUSH/HOLD/RETREAT. See §1.10.A.
                #
                # Fix: the potentials are now small (O(1-5), see config_v2), so the
                # drift is <= -0.5/step and is dwarfed by the genuine reward terms.
                # The task objective lives in R (dense reach + ratcheted door
                # progress + sparse hold), NOT in the shaping. This is exactly the
                # role Ng et al. intend for Phi: guidance, not objective.
                shaping = self.gamma * phi_now - self._prev_phi
                rew_info["phi_shape"] = float(np.clip(shaping, -10.0, 10.0))

            self._prev_phi = phi_now

        # ══════════════════════════════════════════════════════════════════════
        # PHASE 1 — REACH: approach and grasp the handle
        # ══════════════════════════════════════════════════════════════════════
        if phase == PHASE_REACH:
            # Curriculum-scaled distance weights  [§3.6, Devlin & Kudenko 2012]
            k = 1.0 + self.cfg.curriculum_reward_k * curriculum_lvl

            # ── Direct dense distance rewards ─────────────────────────────────────────
            # In the cumulative-potential design Phi_reach == 0, so REACH receives
            # NO shaping gradient at all. These dense terms are therefore the ONLY
            # approach signal and must be as strong as the working v1 reward
            # (env_gen.py), otherwise the arm stalls at mid-distance. The previous
            # v2 had weakened them (-2 dist, no xy) on the false assumption that the
            # potential would help here — it cannot. See §1.10.B.
            rew_info["dist_3d"] = -5.0  * k * dist_handle
            rew_info["dist_xy"] = -3.0  * k * (dist_xy if dist_xy is not None else dist_handle)
            rew_info["dist_z"]  = -15.0 * k * abs(height_diff)

            # Approach geometry penalties
            if height_diff < -0.005:
                rew_info["app_blw"] = -3.0 * abs(height_diff + 0.005)
            if height_diff > 0.03:
                # Penalise being above handle (condition was gated on gripper_action > 0.2
                # which is NEVER true in REACH — gripper is always open/negative).
                rew_info["app_top"] = -1.5 * height_diff

            # Multi-approach alignment  [§3.3, ten Pas 2017] — handled separately
            # (alignment value already max-pooled over K candidates by MultiApproachGrasp)
            prox_factor       = float(np.exp(-10.0 * dist_handle))
            rew_info["align"] = -self.cfg.w_multi_align * (1.0 - alignment) * prox_factor
            rew_info["flat"]  = -0.5 * flat_alignment * prox_factor

            # Gripper Management (physics-calibrated based on adaptive grasp threshold)
            d_thresh = self.cfg.fsm_grasp_dist_k_radius * handle_radius + self.cfg.fsm_grasp_dist_k_offset
            if dist_handle > d_thresh:
                if gripper_action > -0.85:
                    rew_info["grip"] = -1.0 * (gripper_action - (-0.85))
            else:
                if gripper_action > -0.85:
                    norm_g = (gripper_action - (-0.85)) / (1.0 - (-0.85))
                    rew_info["grip"] = 2.5 * norm_g

            # REACH → PUSH bonus
            if just_grasped and not fsm_state.has_grasp_bonus:
                # Keep a small explicit bonus for interpretability
                rew_info["phase_trans"]   = 10.0
                fsm_state.has_grasp_bonus = True

            # Grasp lost penalty
            if grasp_lost:
                rew_info["grasp_lost_pen"] = -5.0

        # ══════════════════════════════════════════════════════════════════════
        # PHASE 2 — PUSH: maintain grip, push door closed
        # ══════════════════════════════════════════════════════════════════════
        elif phase == PHASE_PUSH:
            # Distance maintenance (keep gripper on handle)
            rew_info["dist_3d"] = -5.0 * dist_handle
            rew_info["dist_z"]  = -15.0 * abs(height_diff)

            # ── Real task objective: ratcheted door-progress ──────────────────────────
            # This is the v1-proven closing signal (env_gen.py). It is the GENUINE
            # reward R that defines "close the door"; adding Ng-shaping on top leaves
            # its optimum unchanged [Ng et al. 1999], so it is kept ON regardless of
            # use_potential_reward.
            #
            # Two bugs are fixed here vs the previous v2 (§1.10.C):
            #   1. It was gated behind `if not use_potential_reward` → with shaping ON
            #      (the default) the door had NO closing reward, only the tiny phi_push.
            #   2. It read fsm_state.min_door_angle, which the FSM already lowered to
            #      the current angle THIS step → delta was always 0. We keep our own
            #      ratchet so delta reflects true new progress.
            #
            # Non-exploitable: _min_door_angle only ever decreases, so oscillating the
            # door back and forth cannot re-earn reward.
            if self._min_door_angle is None:
                self._min_door_angle = door_angle
            if gripper_action > grip_thresh:
                delta = self._min_door_angle - door_angle
                if delta > 0:
                    rew_info["door_prog"]  = 2000.0 * delta
                    self._min_door_angle   = door_angle

            # Lift penalty (same as v1)
            if action[2] > 0.05:
                rew_info["lift_pen"] = -2.0 * action[2]

            # Action regularisation (same as v1, small)
            rew_info["act_pen"] = -0.005 * float(np.linalg.norm(action[:-1]))

            # Grasp loss penalties
            if grasp_lost:
                rew_info["dist_lost"] = -6.0 * max(0.0, dist_handle - 0.05)
                rew_info["grip_lost"] = -5.0 * abs(min(0.0, gripper_action) - grip_thresh)
            elif gripper_action < grip_thresh:
                # §1.13 — Penalizza SOLO una presa genuinamente debole (sotto la soglia di
                # presa adattiva), in modo dolce. La versione precedente
                # `grip = -5·(1 − gripper_action)` pretendeva il gripper a +1.0 ESATTO:
                # tassava anche una presa valida (es. 0.8 con soglia 0.75 → −1.0/step) e
                # diventava −10/step se il gripper si apriva. Questo trasformava PUSH in
                # un campo minato a valore atteso negativo, spingendo la policy a
                # "accamparsi" in REACH (vedi §1.13). Ora una presa ≥ soglia non paga nulla.
                rew_info["grip"] = -2.0 * (grip_thresh - gripper_action)

            # §1.16 — GRIP IN CHIUSURA: premia il MANTENIMENTO del contatto fisico
            # (is_physically_closed = gripper_width nella banda di buona presa) MENTRE la
            # porta si chiude, scalato sul progresso di chiusura. È un reward POSITIVO e
            # limitato che premia lo STATO desiderato, NON lo "stringere di più": su
            # maniglie sottili stringere oltre farebbe scendere gripper_width sotto la
            # soglia di contatto e perderebbe la presa (§3.1). La scala sul progresso
            # (closing_progress ≈ 0 a porta aperta → 1 a porta chiusa) evita un nuovo
            # incentivo ad "accamparsi" tenendo ferma una porta aperta: tenere una porta
            # aperta paga ~0; il segnale cresce solo mentre/dopo aver chiuso. Bounded e
            # non-negativo ⇒ nessuna "valle" negativa (§1.13); vive in R (non nello
            # shaping Φ) quindi l'invarianza di Ng resta intatta. Indipendente dal
            # curriculum ⇒ effetto identico a livello 0 e 1.
            if is_physically_closed and door_max > 1e-6:
                closing_progress = float(np.clip(1.0 - door_angle / door_max, 0.0, 1.0))
                rew_info["grip_contact"] = self.cfg.w_grip_contact * closing_progress

        # ══════════════════════════════════════════════════════════════════════
        # PHASE 3 — HOLD: maintain door closed
        # ══════════════════════════════════════════════════════════════════════
        elif phase == PHASE_HOLD:
            is_closed = abs(door_qpos) < 0.03

            if is_closed:
                rew_info["hold"] = 1.0 - abs(door_qpos)
            else:
                # Bounce penalty (same weight as v1)
                rew_info["hold_bounce"] = -20.0 * abs(door_qpos)

            # Anti-bounce velocity damping  [v1: -25.0, same weight]
            if abs(door_qvel) > 0.01:
                rew_info["hold_veldamp"] = -25.0 * abs(door_qvel)

            # Physical grip check (same as v1)
            if not is_physically_closed:
                rew_info["hold_slip"] = -5.0

            # Gripper command (same as v1)
            if gripper_action > grip_thresh:
                rew_info["hold_grip"] = 1.0
            else:
                rew_info["hold_grip"] = -2.0 * abs(gripper_action - grip_thresh)

            if gripper_action < 0.0:
                rew_info["hold_drop_pen"] = -10.0 * abs(gripper_action)

            # Joint freeze (same as v1)
            if joint_vel is not None:
                rew_info["hold_jnt_freeze"] = -1.0 * float(np.linalg.norm(joint_vel))

            # Arm action norm (same as v1; action[:-1] is zeroed by env override)
            action_norm = float(np.linalg.norm(action[:-1]))
            if action_norm < 0.05:
                rew_info["hold_act"] = 1.0
            else:
                rew_info["hold_act"] = -2.0 * action_norm

            # Wrist torsion penalty (same as v1)
            rew_info["hold_flat"] = -2.0 * flat_alignment

            # Handle distance (same as v1)
            if dist_handle > 0.06:
                rew_info["hold_dist"] = -3.0 * (dist_handle - 0.06)

        # ══════════════════════════════════════════════════════════════════════
        # PHASE 4 — RETREAT: withdraw from handle
        # ══════════════════════════════════════════════════════════════════════
        elif phase == PHASE_RETREAT:
            if fsm_state.retreat_pos is not None:
                dist_to_target = float(np.linalg.norm(eef_pos - fsm_state.retreat_pos))
            else:
                dist_to_target = 0.20

            # Gripper: open to release handle  [v1: same]
            if gripper_action < -0.85:
                rew_info["ret_grip"] = 2.0
            else:
                rew_info["ret_grip"] = -1.0 * abs(gripper_action + 1.0)

            # Wrist rotation penalty  [v1: same]
            rew_info["ret_rot"] = -3.0 * float(np.linalg.norm(action[3:6]))

            # Lateral / downward penalties near handle  [v1: same]
            if dist_handle < 0.12:
                rew_info["ret_lat"] = -5.0 * abs(action[1])
                if action[2] < 0:
                    rew_info["ret_down"] = -5.0 * abs(action[2])

            # Directional reward toward retreat target, poi SETTLE (immobilizza + rilascia).
            # §1.15 — la zona di settle è allargata (fsm_retreat_settle_dist) rispetto al
            # vecchio freeze a 0.02 m, che non veniva quasi mai raggiunto → il braccio
            # continuava a inseguire il target via ret_dir e non si fermava mai. Ora, appena
            # vicino alla posa di retreat, il braccio si FERMA (penalità su tutte le DOF
            # tranne il gripper) e riceve un bonus di rilascio solo a porta chiusa + gripper
            # aperto. La terminazione §1.14 resta invariata → la len d'episodio non cambia.
            if dist_to_target > self.cfg.fsm_retreat_settle_dist:
                dir_to_target    = fsm_state.retreat_pos - eef_pos
                dir_norm         = dir_to_target / (dist_to_target + 1e-6)
                action_alignment = float(np.dot(action[:3], dir_norm))

                rew_info["ret_dir"]  = 3.0 * action_alignment
                perp                 = action[:3] - action_alignment * dir_norm
                rew_info["ret_perp"] = -2.0 * float(np.linalg.norm(perp))
            else:
                # SETTLE: immobilizza il braccio (tutte le DOF tranne il gripper).
                rew_info["ret_freeze"] = -self.cfg.w_retreat_settle * float(
                    np.linalg.norm(action[:-1])
                )
                # Rilascio pulito: bonus SOLO a porta chiusa + gripper aperto, così non
                # c'è incentivo ad "accamparsi" immobile a porta ancora aperta.
                if abs(door_qpos) < 0.03 and gripper_action < -0.85:
                    rew_info["ret_release"] = 1.0

            # Progressive joint freeze  [v1: same]
            if joint_vel is not None:
                fw = float(np.clip(1.0 - dist_to_target / 0.15, 0.1, 1.0))
                rew_info["ret_jnt_prog"] = -5.0 * fw * float(np.linalg.norm(joint_vel))

            # Latch monitor  [v1: same]
            rew_info["latch_ret"] = -1.0 * abs(latch_qpos)

            # Door stability monitor in retreat
            rew_info["hold"] = 1.0 - abs(door_qpos) if abs(door_qpos) < 0.03 else 0.0

            # §1.14 — Stato TERMINALE al completamento del task.
            # Quando il RETREAT è sostenuto (la porta è rimasta chiusa abbastanza a lungo)
            # e la porta è chiusa e il latch è neutro, il task È finito: terminare qui
            # (1) accorcia l'episodio (~step 120 invece di 500) e (2) toglie sia
            # l'incentivo sia l'occasione di continuare a muovere il braccio per
            # raccogliere ret_dir/ret_grip. Bonus una tantum a preservare il valore.
            if (self.cfg.terminate_on_retreat_complete
                    and fsm_state.retreat_steps >= self.cfg.fsm_retreat_target_steps
                    and abs(door_qpos)  < 0.03
                    and abs(latch_qpos) < 0.08):
                rew_info["success_bonus"] = self.cfg.success_bonus
                terminated = True

        # ── Total reward ──────────────────────────────────────────────────────
        reward = float(sum(rew_info.values()))

        # ── Termination override  [v1: same logic] ────────────────────────────
        if terminated:
            latch_neutral = abs(latch_qpos) < 0.08
            door_closed   = abs(door_qpos)  < 0.03
            if not latch_neutral or not door_closed:
                terminated  = False
                reward     -= 500.0
                reward      = float(np.clip(reward, -100.0, 100.0))

        if not terminated:
            reward = float(np.clip(reward, -100.0, 100.0))

        return reward, terminated, truncated, rew_info