#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fsm_unified.py — la macchina a stati, quattro fasi.

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
    passi_cala: int = 0           # guardia di stallo in HOLD: frame fuori tolleranza
    passi_a_bersaglio: int = 0    # frame consecutivi con A vera prima di HOLD
    passi_release: int = 0
    uscita_pulita: bool = False   # §7 terminata per condizione, non per tetto duro

    prev_fase: Fase = Fase.REACH

    @property
    def one_hot(self) -> np.ndarray:
        """4 valori nell'osservazione (§6). Invariata rispetto ai due progetti."""
        v = np.zeros(4, dtype=np.float32)
        v[min(int(self.fase), 3)] = 1.0
        return v


class UnifiedFSM:
    """§3 — quattro fasi. Le condizioni di transizione sono le stesse dei due
    progetti, con QUATTRO correzioni dichiarate nel documento:

      1. MOVE -> HOLD diventa BILATERALE.
         Oggi la chiusura chiede θ ≤ soglia e l'apertura θ ≥ goal − tol. La prima
         è di fatto bilaterale perché la porta non può andare sotto zero; la
         seconda no, e lascia passare qualsiasi sovra-apertura. Qui la condizione
         è `A`, cioè |e| ≤ tol, uguale per i due compiti.
         Il secondo membro del gate e' il COMANDO del gripper sopra la soglia
         adattiva, esattamente come nei due sorgenti (`gripper_action > 0.80`
         nella chiusura, `> g_thresh` nell'apertura dopo la correzione §1.30).
         NON la vicinanza: chiedendo anche quella si apre una finestra morta fra
         0.035 e 0.05 m in cui la mano e' troppo lontana per confermare la presa
         e troppo vicina per perderla, e con la porta gia' al bersaglio l'episodio
         non esce piu' da MOVE. Misurato: 1 valutazione su 20 nella chiusura e
         3 su 20 nell'apertura finivano a 600 passi con la porta ESATTAMENTE al
         bersaglio (errore −0.0060 e −0.0155) e nessuna terminazione.

      2. Si aggiunge una GUARDIA DI STALLO in HOLD.
         Con il gate bilaterale un episodio che supera il bersaglio vedrebbe il
         timer scendere senza mai risalire, restando bloccato fino all'orizzonte;
         la guardia forza l'uscita dopo N frame fuori tolleranza. Il conteggio
         e' CUMULATIVO, non consecutivo: azzerandolo a ogni frame dentro
         tolleranza una porta che oscilla dentro/fuori terrebbe il timer a zero
         e la guardia a zero insieme, e HOLD non finirebbe mai. Cosi' invece
         l'uscita e' garantita: o il timer arriva a T_hold, o la guardia arriva
         a N.

      3. L'ISTERESI della presa, ripresa dal sorgente della chiusura: vedi
         `presa_persa`.

      4. LA PORTA E' IL COMPITO. Se `A` regge per T_hold frame consecutivi
         senza essere passati da HOLD si va in RELEASE: la porta e' stata tenuta
         al bersaglio per il tempo richiesto, comunque fosse messa la mano.
         Senza questa regola la macchina ha stati da cui FINE e' IRRAGGIUNGIBILE
         (128 combinazioni su 576 con la porta ferma al bersaglio); mandando
         invece in HOLD si crea una trappola, perche' HOLD presuppone la presa.
         Vedi la nota estesa dentro `step`.

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

        # (4) LA PORTA E' IL COMPITO: se `A` regge per T_hold frame consecutivi
        # senza che si sia mai entrati in HOLD, la porta E' STATA TENUTA al
        # bersaglio per il tempo richiesto — con o senza la mano sopra — e si
        # passa a RELEASE. Serve a togliere gli stati da cui FINE e'
        # IRRAGGIUNGIBILE (misurato: 128 combinazioni su 576 con la porta ferma
        # al bersaglio).
        #
        # Perche' RELEASE e non HOLD. HOLD presuppone la presa: congela il
        # braccio e paga `hold_grip`, che senza dita chiuse vale −5. Mandandoci
        # l'agente a mano libera si crea una TRAPPOLA da cui non puo' uscire:
        # misurato, −6.21 per passo per 62 passi, cioe' −385 per episodio, e
        # l'addestramento impara la lezione sbagliata — «chiudere la porta senza
        # una presa salda e' una catastrofe» — con REACH all'87 % dei passi e
        # `progress` sceso da +26 a +7.5. RELEASE invece non presuppone niente:
        # paga l'apertura della mano e l'allontanamento, e termina comunque.
        #
        # Non e' una scorciatoia: costa gli stessi T_hold frame della via
        # normale, ma li spende in REACH o MOVE, dove il bilancio e' negativo
        # (−1.45 e −0.50 per passo) invece che positivo (+0.24 in HOLD).
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
            # (1) BILATERALE. Il secondo membro e' il comando del gripper sopra
            # la soglia adattiva, come nei sorgenti, PIU' le dita fisicamente
            # chiuse: e' cio' che rende `hold_grip` una misura sensata in HOLD.
            if door.A and grip_cmd >= soglia and contatto:
                s.fase, s.timer_hold, s.passi_cala = Fase.HOLD, 0, 0
            elif s.passi_persa >= self.thr.grasp_lose_steps:
                s.fase, s.passi_persa = Fase.REACH, 0          # ritorno di fase

        elif s.fase == Fase.HOLD:
            if door.A:
                s.timer_hold += 1
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
                # §7 — il clean success dell'originale richiede una terminazione
                # PULITA: per condizione soddisfatta, non per tetto duro.
                s.uscita_pulita = bool(pronto)
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
