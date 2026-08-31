#!/usr/bin/env python3

"""
Documento di riferimento: tabella_reward_machine_unificata.md
  §3  «Le quattro fasi» + le quattro correzioni alle transizioni
  §8  «I due controllori» — dichiarati come stati mascherati
  §6    struttura a quattro fasi e soglie adattive: INVARIATE

Riferimenti dei sorgenti originali, conservati:
  [Sutton, Precup & Singh 1999]  opzioni: ogni fase è un'opzione con la sua
                                 condizione di terminazione
  [Konidaris & Barto 2009]       soglie adattive al raggio della maniglia
  [ManipForce 2015]              soglia di presa adattiva all'attrito
"""
from enum import IntEnum
from dataclasses import dataclass

import numpy as np


class Fase(IntEnum):
    REACH = 0      # raggiungi e afferra la maniglia
    MOVE = 1       # porta la porta al bersaglio  (era PUSH / PULL)
    HOLD = 2       # mantieni                     (era HOLD / HOLD_OPEN)
    RELEASE = 3    # rilascia e allontanati       (era RETREAT)
    FINE = 4

class Controllore(IntEnum):
    NESSUNO = 0
    C0_RIPORTO_LEVA = 1    # il braccio accompagna la leva; presa ancora chiusa
    C1_ESCAPE = 2          # l'ambiente comanda il braccio verso il punto di ritiro

MASCHERA = {
    Controllore.NESSUNO: frozenset(),
    Controllore.C0_RIPORTO_LEVA: frozenset({"release", "retreat"}),
    Controllore.C1_ESCAPE: frozenset(),
}


@dataclass
class StatoFSM:
    fase: Fase = Fase.REACH
    controllore: Controllore = Controllore.NESSUNO

    passi_presa: int = 0          # conferma della presa (REACH -> MOVE)
    passi_persa: int = 0          # perdita della presa  (MOVE -> REACH)
    timer_hold: int = 0           # sale se A, scende se non A
    passi_cala: int = 0           # guardia di stallo in HOLD: frame fuori tolleranza
    passi_a_bersaglio: int = 0    # frame consecutivi con A vera prima di HOLD
    passi_release: int = 0
    uscita_pulita: bool = False   # §7 terminata per condizione, non per tetto duro

    prev_fase: Fase = Fase.REACH

    @property
    def one_hot(self) -> np.ndarray:
        v = np.zeros(4, dtype=np.float32)
        v[min(int(self.fase), 3)] = 1.0
        return v

class UnifiedFSM:
    def __init__(self, cfg):
        self.cfg = cfg
        self.thr = cfg.thr
        self.s   = StatoFSM()

    # ── soglie adattive: invariate (§6) ──────────────────────────────────
    def soglia_distanza(self, raggio: float) -> float:
        """1.5·raggio + 0.005 — [Konidaris & Barto 2009]."""
        return self.thr.grasp_dist_k_radius * raggio + self.thr.grasp_dist_k_offset

    def soglia_presa(self, attrito: float) -> float:
        """0.75 − 0.10·attrito normalizzato — [ManipForce 2015].
        Più attrito → presa stabile con meno chiusura → soglia più bassa.
        """
        f = float(np.clip((attrito - self.thr.friction_min) /
                          (self.thr.friction_max - self.thr.friction_min + 1e-8), 0.0, 1.0))
        return float(self.thr.grip_thresh_base - self.thr.grip_thresh_k_fric * f)

    def reset(self) -> StatoFSM:
        self.s = StatoFSM()
        return self.s

    def presa_persa(self, door, contatto: bool, dist_mano: float,
                    grip_cmd: float, soglia: float, door_qvel: float) -> bool:
        t   = self.thr
        tol = float(np.clip(t.lose_dist_base + t.lose_dist_k_speed * abs(door_qvel),
                            t.lose_dist_base, t.lose_dist_max))

        al_bersaglio = abs(door.e) <= t.near_target_tol
        return bool(dist_mano > tol
                    or grip_cmd < soglia - t.grip_release_margin
                    or (not contatto and not al_bersaglio))

    # ── Transizione ───────────────────────────────────────────────────
    def step(self, door, presa_ok: bool, contatto: bool, dist_mano: float, latch: float, hold_target: int, grip_cmd: float, soglia: float,
             door_qvel: float, dist_ritiro: float = 1.0) -> StatoFSM:

        s = self.s
        s.prev_fase = s.fase

        if s.fase in (Fase.REACH, Fase.MOVE):
            s.passi_a_bersaglio = s.passi_a_bersaglio + 1 if door.A else 0
            if s.passi_a_bersaglio >= hold_target:
                s.fase, s.passi_release = Fase.RELEASE, 0

        if s.fase == Fase.REACH:
            s.passi_presa = s.passi_presa + 1 if presa_ok else 0
            if s.passi_presa >= self.thr.grasp_confirm_steps:
                s.fase, s.passi_presa = Fase.MOVE, 0

        elif s.fase == Fase.MOVE:
            persa = self.presa_persa(door, contatto, dist_mano, grip_cmd, soglia, door_qvel)
            s.passi_persa = s.passi_persa + 1 if persa else 0

            if door.A and grip_cmd >= soglia and contatto:
                s.fase, s.timer_hold, s.passi_cala = Fase.HOLD, 0, 0
            elif s.passi_persa >= self.thr.grasp_lose_steps:
                s.fase, s.passi_persa = Fase.REACH, 0

        elif s.fase == Fase.HOLD:
            if door.A:
                s.timer_hold += 1
            else:
                s.timer_hold = max(0, s.timer_hold - 1)
                s.passi_cala += 1
            if s.timer_hold >= hold_target:
                s.fase = Fase.RELEASE
            elif s.passi_cala >= self.thr.stall_guard_steps:
                s.fase = Fase.RELEASE

        elif s.fase == Fase.RELEASE:
            s.passi_release += 1
            self._aggiorna_controllore(latch, dist_ritiro)
            pronto = (s.passi_release >= self.thr.retreat_target_steps and door.A and abs(latch) <= self.thr.latch_term_tol)
            if pronto or s.passi_release >= self.thr.retreat_hard_cap:
                s.uscita_pulita = bool(pronto)
                s.fase          = Fase.FINE

        return s

    def _aggiorna_controllore(self, latch: float, dist_ritiro: float) -> None:
        t = self.cfg.task
        s = self.s

        if t.riporto_leva and abs(latch) > t.leva_consegna:
            s.controllore = Controllore.C0_RIPORTO_LEVA
        elif t.escape and dist_ritiro > self.thr.retreat_settle_dist:
            s.controllore = Controllore.C1_ESCAPE
        else:
            s.controllore = Controllore.NESSUNO

    def termini_mascherati(self) -> frozenset:
        return MASCHERA[self.s.controllore]