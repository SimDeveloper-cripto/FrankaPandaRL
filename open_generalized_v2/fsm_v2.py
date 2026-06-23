#!/usr/bin/env python3
# open_generalized_v2/fsm_v2.py
#
# AdaptiveFSMOpen — Macchina a stati a soglie adattive per l'APERTURA generalizzata.
#
# E' la differenza architetturale principale rispetto alla v1 (come per la chiusura):
# la transizione tra fasi non usa soglie fisse ma soglie CONTEST-SENSITIVE, funzione
# della fisica corrente (frizione/raggio maniglia, rigidità latch) fornita dal domain
# randomizer. Questo è ciò che permette la "generalizzazione della generalizzazione".
#
# Fasi (SPECULARI alla chiusura REACH→PUSH→HOLD→RETREAT):
#   REACH      0  — avvicinati e afferra la maniglia
#   PULL       1  — tira/spingi la porta verso l'angolo-obiettivo (apertura)
#   HOLD_OPEN  2  — mantieni la porta aperta al goal per un tempo adattivo
#   RETREAT    3  — rilascia e allontanati (transizioni gestite dall'environment)
#
# Inversione chiave chiusura ↔ apertura:
#   chiusura:  PUSH→HOLD   quando  door_angle <= success_angle      (porta chiusa)
#   apertura:  PULL→HOLD   quando  door_angle >= goal_angle - tol    (porta aperta al goal)
#
# Riferimenti:
#   [1] Sutton, Precup & Singh (1999) — fasi come opzioni con terminazione.
#   [2] Konidaris & Barto (2009)      — precondizioni di opzione.
#   [13] ManipForce (2015)            — soglie adattive di presa/forza.

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional, List

PHASE_REACH     = 0
PHASE_PULL      = 1
PHASE_HOLD_OPEN = 2
PHASE_RETREAT   = 3

PHASE_NAMES = {0: "REACH", 1: "PULL", 2: "HOLD_OPEN", 3: "RETREAT"}


@dataclass
class FSMStateOpen:
    phase                : int = PHASE_REACH
    grasp_confirm_count  : int = 0
    hold_open_duration   : int = 0
    reach_steps          : int = 0
    pull_steps           : int = 0
    hold_steps_total     : int = 0
    retreat_steps        : int = 0
    return_hold          : int = 0
    retreat_pos          : Optional[np.ndarray] = None
    target_hold_steps    : Optional[int] = None

    def reset(self):
        self.phase               = PHASE_REACH
        self.grasp_confirm_count = 0
        self.hold_open_duration  = 0
        self.reach_steps         = 0
        self.pull_steps          = 0
        self.hold_steps_total    = 0
        self.retreat_steps       = 0
        self.return_hold         = 0
        self.retreat_pos         = None
        self.target_hold_steps   = None

    @property
    def phase_name(self) -> str:
        return PHASE_NAMES.get(self.phase, "?")


class AdaptiveFSMOpen:
    """
    FSM a soglie adattive per l'apertura. API speculare a AdaptiveFSM (chiusura):
      - grip_thresh(friction)            : soglia di chiusura gripper adattiva
      - grasp_dist_thresh(handle_radius) : distanza di presa adattiva
      - compute_target_hold_steps(...)   : timer HOLD_OPEN adattivo
      - update(...)                      : avanza la FSM, ritorna lista di eventi (log)
    """

    _GRASP_CONFIRM_STEPS = 5   # step consecutivi di presa valida per confermare REACH→PULL

    def __init__(self, cfg):
        self.cfg = cfg
        self.state = FSMStateOpen()

    # ── Soglie adattive (§3.1) ──────────────────────────────────────────────────

    def grip_thresh(self, handle_friction: float) -> float:
        """Soglia di chiusura gripper adattiva alla frizione (ManipForce 2015, §3.1)."""
        f_min = self.cfg.fsm_friction_min
        f_max = self.cfg.fsm_friction_max
        norm_f = float(np.clip((handle_friction - f_min) / (f_max - f_min + 1e-8), 0.0, 1.0))
        return float(np.clip(
            self.cfg.fsm_grip_thresh_base - self.cfg.fsm_grip_thresh_k_fric * norm_f,
            0.50, 0.90,
        ))

    def grasp_dist_thresh(self, handle_radius: float) -> float:
        """Distanza di presa adattiva al raggio maniglia (§3.1)."""
        return float(
            self.cfg.fsm_grasp_dist_base
            + self.cfg.fsm_grasp_dist_k_radius * handle_radius
            + self.cfg.fsm_grasp_dist_offset
        )

    def compute_target_hold_steps(self, control_freq, latch_stiffness, base_latch_stiffness):
        """
        Timer di HOLD_OPEN adattivo alla rigidità del latch (§3.1).
        Latch più rigido tende a richiudere → serve mantenere l'apertura un po' di più.
        """
        base = float(base_latch_stiffness) if base_latch_stiffness else 1.0
        stiff = float(latch_stiffness) if latch_stiffness else base
        # frazione [0,1]: 0 = molla debole, 1 = molla forte (max 2x base)
        norm = float(np.clip((stiff - base) / (base + 1e-8), 0.0, 1.0))
        extra = self.cfg.fsm_hold_k_stiff * norm * control_freq * 0.5
        return int(self.cfg.fsm_hold_base_steps + extra)

    # ── Update ───────────────────────────────────────────────────────────────────

    def update(
        self,
        *,
        door_angle       : float,
        goal_angle       : float,
        open_tol         : float,
        gripper_action   : float,
        dist_handle      : float,
        handle_radius    : float,
        handle_friction  : float,
        is_physically_closed: bool,
        gripper_width    : float,
        prev_angle       : float,
        control_freq     : int,
        door_qpos        : float,
        latch_stiffness  : float,
        base_latch_stiffness: float,
        wrist_align_ok   : bool = True,
        beta_probs       : Optional[dict] = None,
    ) -> List[str]:
        """
        Avanza la FSM di uno step. Ritorna una lista di stringhe-evento per il logging.
        SPECULARE alla chiusura, con l'unica inversione nella condizione PULL→HOLD_OPEN.
        """
        s = self.state
        events: List[str] = []

        g_thresh = self.grip_thresh(handle_friction)         # §3.1
        d_thresh = self.grasp_dist_thresh(handle_radius)     # §3.1

        if s.target_hold_steps is None:
            s.target_hold_steps = self.compute_target_hold_steps(
                control_freq, latch_stiffness, base_latch_stiffness
            )

        if   s.phase == PHASE_REACH    : s.reach_steps      += 1
        elif s.phase == PHASE_PULL     : s.pull_steps       += 1
        elif s.phase == PHASE_HOLD_OPEN: s.hold_steps_total += 1
        elif s.phase == PHASE_RETREAT  : s.retreat_steps    += 1

        # ── REACH → PULL : presa confermata (soglie adattive + gate polso opz.) ──
        if s.phase == PHASE_REACH:
            grasp_cond = (
                gripper_action > g_thresh
                and is_physically_closed
                and dist_handle < d_thresh
                and wrist_align_ok
            )
            if grasp_cond:
                s.grasp_confirm_count += 1
            else:
                s.grasp_confirm_count = 0

            if s.grasp_confirm_count >= self._GRASP_CONFIRM_STEPS:
                s.phase = PHASE_PULL
                events.append(f"REACH→PULL (d={dist_handle:.3f}, g={gripper_action:.2f})")

        # ── PULL → HOLD_OPEN : porta aperta al goal (INVERSIONE vs chiusura) ──────
        elif s.phase == PHASE_PULL:
            # apertura: door_angle deve aver RAGGIUNTO il goal (entro tolleranza)
            opened_enough = (door_angle >= goal_angle - open_tol) and (gripper_action > 0.80)
            if opened_enough:
                s.phase = PHASE_HOLD_OPEN
                s.hold_open_duration = 0
                events.append(f"PULL→HOLD_OPEN (angle={door_angle:.3f}, goal={goal_angle:.3f})")
            elif not is_physically_closed:
                # presa persa durante il PULL → torna a REACH
                s.phase = PHASE_REACH
                s.grasp_confirm_count = 0
                events.append(f"PULL→REACH (grip lost, width={gripper_width:.3f})")

        # ── HOLD_OPEN → RETREAT : mantenuto aperto per il tempo adattivo ──────────
        elif s.phase == PHASE_HOLD_OPEN:
            open_ok = door_angle >= goal_angle - open_tol
            if open_ok:
                s.hold_open_duration += 1
            else:
                # è ri-richiusa sotto il goal: scala il timer (specchio del bounce chiusura)
                penalty = int(abs(goal_angle - door_angle) / max(open_tol, 1e-6) * 5)
                s.hold_open_duration = max(0, s.hold_open_duration - penalty)

            hold_done = s.hold_open_duration >= s.target_hold_steps
            if hold_done:
                s.phase = PHASE_RETREAT
                events.append(
                    f"HOLD_OPEN→RETREAT (dur={s.hold_open_duration}, "
                    f"target={s.target_hold_steps})"
                )

        # PHASE_RETREAT: transizioni/terminazione gestite dall'environment (§1.17/§1.21)
        return events

    def reset(self):
        self.state.reset()