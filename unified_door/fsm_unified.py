#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fsm_unified.py — la macchina a stati, quattro fasi.

Documento di riferimento: tabella_reward_machine_unificata.md
  §3  «Le quattro fasi» + le tre correzioni alle transizioni
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
    HOLD = 2       # mantieni                      (era HOLD / HOLD_OPEN)
    RELEASE = 3    # rilascia e allontanati        (era RETREAT)
    FINE = 4


class Controllore(IntEnum):
    """§8 — non sono termini di ricompensa: sono override sull'azione.

    Entrano nella macchina SOLO come stati dichiarati in cui un sottoinsieme
    dei termini è mascherato. Renderlo esplicito è ciò che impedisce alla
    stessa fase di avere due funzioni di ricompensa diverse a seconda di che
    cosa sta facendo l'ambiente.
    """
    NESSUNO = 0
    C0_RIPORTO_LEVA = 1    # il braccio accompagna la leva; presa ancora chiusa
    C1_ESCAPE = 2          # l'ambiente comanda il braccio verso il punto di ritiro


# §8 — LA MASCHERA, dichiarata in un punto solo e COMPLETA.
#
# Nel progetto originale la maschera era nascosta in una condizione del codice
# ed era incompleta: copriva ret_grip, ret_dir, ret_perp ma NON ret_rot, che
# restava attivo e costava −14.44 su 27 step per una rotazione che la policy
# non aveva scelto. Qui è una costante, quindi non può risultare incompleta
# per distrazione.
MASCHERA = {
    Controllore.NESSUNO: frozenset(),
    Controllore.C0_RIPORTO_LEVA: frozenset({"release", "retreat"}),
    Controllore.C1_ESCAPE: frozenset(),   # nessuna: la policy è comunque premiata
}


@dataclass
class StatoFSM:
    fase: Fase = Fase.REACH
    controllore: Controllore = Controllore.NESSUNO

    passi_presa: int = 0          # conferma della presa (REACH -> MOVE)
    passi_persa: int = 0          # perdita della presa  (MOVE -> REACH)
    timer_hold: int = 0           # sale se A, scende se non A
    passi_cala: int = 0           # guardia di stallo in HOLD: decrementi consecutivi
    passi_release: int = 0

    prev_fase: Fase = Fase.REACH

    @property
    def one_hot(self) -> np.ndarray:
        """4 valori nell'osservazione (§6). Invariata rispetto ai due progetti."""
        v = np.zeros(4, dtype=np.float32)
        v[min(int(self.fase), 3)] = 1.0
        return v


class UnifiedFSM:
    """§3 — quattro fasi. Le condizioni di transizione sono le stesse dei due
    progetti, con TRE correzioni dichiarate nel documento:

      1. MOVE -> HOLD diventa BILATERALE.
         Oggi la chiusura chiede θ ≤ soglia e l'apertura θ ≥ goal − tol. La prima
         è di fatto bilaterale perché la porta non può andare sotto zero; la
         seconda no, e lascia passare qualsiasi sovra-apertura. Qui la condizione
         è `A`, cioè |e| ≤ tol, uguale per i due compiti.

      2. Si aggiunge una GUARDIA DI STALLO in HOLD.
         Con il gate bilaterale un episodio che supera il bersaglio vedrebbe il
         timer scendere senza mai risalire, restando bloccato fino all'orizzonte;
         la guardia forza l'uscita dopo N decrementi consecutivi.

      3. L'ISTERESI della presa, ripresa dal sorgente della chiusura: vedi
         `presa_persa`.

      La stessa guardia era stata provata anche in MOVE ed e' stata TOLTA: la
      misura (chiusura, 600k passi) mostra che produce solo un ping-pong
      REACH<->MOVE al periodo della guardia — REACH 26 % / MOVE 74 %, cioe' 20
      passi dentro e 7 fuori — e ogni espulsione fa scendere la scala di Φ di
      −4.6, che nel log si legge come `phi` −0.82 per passo in REACH, il termine
      dominante della fase. Costo netto misurato: `ep_rew_mean` da −400 a −464.

      Il documento avverte (§7): la prima e la seconda vanno introdotte
      INSIEME, perché la prima senza la seconda peggiora la situazione.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.thr = cfg.thr
        self.s = StatoFSM()

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
        """§3 — ISTERESI: si afferra stretto, si perde largo.

        La condizione con cui si CONFERMA la presa (`presa_ok`, tre condizioni
        strette) non è quella con cui la si PERDE. È la forma del sorgente della
        chiusura, che la chiama «Schmitt trigger + hysteresis»:

          · la mano può allontanarsi fino a 0.05 m, e fino a 0.12 m se la porta
            si sta muovendo — perché muovere la porta muove la maniglia;
          · il comando del gripper può scendere fino a soglia − 0.20;
          · le dita fisicamente aperte fanno perdere la presa **tranne che al
            bersaglio**, dove la porta si accompagna a mano aperta invece che
            trascinata (`near_latch` nel sorgente della chiusura).

        Senza l'isteresi le due condizioni coincidono: qualunque movimento del
        braccio rischia di far cadere MOVE, e con esso il gradino di Φ. La
        condotta ottima diventa afferrare e NON muoversi più — misurata nei
        training: MOVE 88 % dei passi, `progress` assente, HOLD mai raggiunta.
        """
        t = self.thr
        tol = float(np.clip(t.lose_dist_base + t.lose_dist_k_speed * abs(door_qvel),
                            t.lose_dist_base, t.lose_dist_max))
        al_bersaglio = abs(door.e) <= t.near_target_tol
        return bool(dist_mano > tol
                    or grip_cmd < soglia - t.grip_release_margin
                    or (not contatto and not al_bersaglio))

    # ── la transizione ───────────────────────────────────────────────────
    def step(self, door, presa_ok: bool, contatto: bool, dist_mano: float,
             latch: float, hold_target: int, grip_cmd: float, soglia: float,
             door_qvel: float, dist_ritiro: float = 1.0) -> StatoFSM:
        """`door` è un DoorState: da lì viene `A`, unica per i due compiti."""
        s = self.s
        s.prev_fase = s.fase

        if s.fase == Fase.REACH:
            s.passi_presa = s.passi_presa + 1 if presa_ok else 0
            if s.passi_presa >= self.thr.grasp_confirm_steps:
                s.fase, s.passi_presa = Fase.MOVE, 0

        elif s.fase == Fase.MOVE:
            persa = self.presa_persa(door, contatto, dist_mano, grip_cmd, soglia, door_qvel)
            s.passi_persa = s.passi_persa + 1 if persa else 0
            if s.passi_persa >= self.thr.grasp_lose_steps:
                s.fase, s.passi_persa = Fase.REACH, 0          # ritorno di fase
            elif door.A and presa_ok:                          # (1) BILATERALE
                s.fase, s.timer_hold, s.passi_cala = Fase.HOLD, 0, 0

        elif s.fase == Fase.HOLD:
            if door.A:
                s.timer_hold += 1
                s.passi_cala = 0
            else:
                s.timer_hold = max(0, s.timer_hold - 1)
                s.passi_cala += 1
            if s.timer_hold >= hold_target:
                s.fase = Fase.RELEASE
            elif s.passi_cala >= self.thr.stall_guard_steps:    # (2) GUARDIA
                s.fase = Fase.RELEASE

        elif s.fase == Fase.RELEASE:
            s.passi_release += 1
            self._aggiorna_controllore(latch, dist_ritiro)
            pronto = (s.passi_release >= self.thr.retreat_target_steps
                      and door.A and abs(latch) <= self.thr.latch_term_tol)
            if pronto or s.passi_release >= self.thr.retreat_hard_cap:
                s.fase = Fase.FINE

        return s

    def _aggiorna_controllore(self, latch: float, dist_ritiro: float) -> None:
        """§8 — quale controllore è attivo. Solo l'apertura li accende."""
        t = self.cfg.task
        s = self.s
        if t.riporto_leva and abs(latch) > self.thr.latch_term_tol:
            s.controllore = Controllore.C0_RIPORTO_LEVA
        elif t.escape and dist_ritiro > self.thr.retreat_settle_dist:
            s.controllore = Controllore.C1_ESCAPE
        else:
            s.controllore = Controllore.NESSUNO

    def termini_mascherati(self) -> frozenset:
        """I termini che NON vanno pagati in questo step, perché il braccio non
        è guidato dalla policy. Un punto solo, quindi non può essere incompleta."""
        return MASCHERA[self.s.controllore]
