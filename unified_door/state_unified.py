#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
state_unified.py — le quattro grandezze derivate e la soglia di stato porta.

Documento di riferimento: tabella_reward_machine_unificata.md
  §4  «La chiusura è l'apertura con bersaglio zero»
  §4  «Quattro grandezze derivate»
  §4  «La soglia di stato porta»

È il cuore dell'unificazione, ed è volutamente il file più corto del progetto:
tutto ciò che distingue i due compiti passa da qui.
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class DoorState:
    """§4 — le quattro grandezze, calcolate e non configurate.

    | grandezza | definizione                                   |
    |-----------|-----------------------------------------------|
    | σ         | sign(θ* − θ₀)   verso del compito              |
    | e         | θ − θ*          errore con segno               |
    | A         | |e| ≤ tol       soglia di stato porta          |
    | p         | cricchetto sul lavoro svolto, saturato        |
    """
    theta_star: float          # θ*  bersaglio, fissato al reset
    theta_zero: float          # θ₀  angolo iniziale, fissato al reset
    tol: float                 # tolleranza

    theta: float = 0.0         # θ   angolo corrente
    leva: float = 0.0          # angolo corrente della leva
    leva_rilascio: float = 0.0 # |leva| che libera il chiavistello, 0 = nessun blocco
    _p: float = 0.0            # riferimento del cricchetto
    _ancorato: bool = False    # il riferimento e' stato posato (ingresso in MOVE)

    # ── §4 -------------------------------------------------------------
    @property
    def sigma(self) -> float:
        """Verso del compito: −1 chiude, +1 apre. Calcolato una volta al reset."""
        s = np.sign(self.theta_star - self.theta_zero)
        return float(s) if s != 0.0 else 1.0

    @property
    def e(self) -> float:
        """Errore con segno rispetto al bersaglio."""
        return self.theta - self.theta_star

    @property
    def A(self) -> bool:
        """§4 — LA SOGLIA DI STATO PORTA.

        «la porta è dove deve essere?» È una sola condizione, e la macchina la
        usa in cinque punti (vedi `USI_DI_A`). È anche la risposta alla domanda
        «come fa il robot a capire se la porta è aperta o chiusa»: non glielo si
        dice, lo calcola — e `tol` tiene conto della randomizzazione del dominio.
        """
        return abs(self.e) <= self.tol

    @property
    def escursione(self) -> float:
        """Il LAVORO TOTALE del compito, in radianti: leva + cerniera.

        §8 — dove il chiavistello blocca la porta, girare la leva e' il primo
        tratto della stessa corsa. Misurato sulle traiettorie del modello
        originale: la cerniera esce dal gioco solo a |leva| = 1.03 rad. Sommare i
        due tratti e' cio' che rende `progress` e Φ_move capaci di vedere anche
        quel pezzo di compito.
        Vale |θ* − θ₀| da sola quando `leva_rilascio` e' 0 (chiusura).
        """
        return abs(self.theta_star - self.theta_zero) + self.leva_rilascio

    @property
    def _lavoro(self) -> float:
        """Radianti di corsa gia' coperti, leva + cerniera, senza cricchetto."""
        return (min(abs(self.leva), self.leva_rilascio)
                + float(np.clip(self.sigma * (self.theta - self.theta_zero),
                                0.0, abs(self.theta_star - self.theta_zero))))

    @property
    def avanzamento(self) -> float:
        """Frazione dell'escursione coperta ADESSO, in [0, 1].

        Differisce da `p` perche' NON e' a cricchetto: se la porta torna
        indietro, questa scende. Serve allo shaping: Φ dev'essere funzione
        dello stato corrente, come nei due originali (`1 − door_angle/door_max`
        nella chiusura, `(door_angle − door_min)/(goal − door_min)`
        nell'apertura). Usando il cricchetto dentro Φ il potenziale diventa
        monotono e il gradiente sull'angolo della porta SPARISCE: spingere e
        lasciar tornare indietro valgono uguale, e la porta che si chiude da
        sola per gravita' consuma il potenziale senza che l'agente impari nulla.
        Il cricchetto resta dove serve — nel termine di ricompensa `progress` —
        perche' li' l'obiettivo e' non farsi ripagare due volte lo stesso tratto.
        Un potenziale invece e' gia' non-mungibile per costruzione: oscillare la
        porta produce una somma telescopica ~0 [Ng, Russell & Harada 1999].
        """
        if self.escursione < 1e-9:
            return 0.0
        return float(np.clip(self._lavoro / self.escursione, 0.0, 1.0))

    @property
    def p(self) -> float:
        """Progresso utile, a cricchetto e SATURATO al bersaglio (§7)."""
        return self._p

    @property
    def target_ramp(self) -> float:
        """§1 — clip(1 − |e|/tol, 0, 1). Vale 1 al bersaglio, 0 al bordo."""
        return float(np.clip(1.0 - abs(self.e) / max(self.tol, 1e-9), 0.0, 1.0))

    # ── aggiornamento ------------------------------------------------------
    def reset(self, theta_zero: float, theta_star: float, tol: float,
              leva_rilascio: float = 0.0) -> None:
        self.theta_zero = float(theta_zero)
        self.theta_star = float(theta_star)
        self.tol = float(tol)
        self.leva_rilascio = float(leva_rilascio)
        self.theta = float(theta_zero)
        self.leva = 0.0
        self._p = 0.0
        self._ancorato = False

    def step(self, theta: float, leva: float = 0.0) -> None:
        """Aggiorna lo stato letto dal simulatore. Non paga niente."""
        self.theta = float(theta)
        self.leva = float(leva)

    def ancora(self) -> None:
        """§7 — posa il riferimento del cricchetto all'ingresso in MOVE.

        E' la riga `if self._min_door_angle is None: self._min_door_angle =
        door_angle` dei due sorgenti, che vive nel ramo della fase di moto: il
        progresso si misura da DOVE SI E' AFFERRATO, non dalla posa iniziale.
        Senza, il tratto che la porta percorre da sola durante REACH — la
        gravita' nella chiusura — verrebbe consumato dal cricchetto e non
        pagato a nessuno. Idempotente: agisce solo la prima volta.
        """
        if not self._ancorato:
            self._ancorato = True
            self._p = self._lavoro

    def incassa(self) -> float:
        """§7 — paga il progresso NUOVO e fa avanzare il riferimento.

        Va chiamata SOLO quando il progresso e' effettivamente pagabile. Il
        riferimento avanza dentro lo stesso `if` del pagamento, esattamente
        come nei sorgenti: cosi' il tratto percorso mentre la presa e'
        comandata sotto soglia non viene distrutto, ma resta da incassare.

        Due proprieta', entrambe dichiarate nel documento:
          · CRICCHETTO: il riferimento non torna indietro, quindi oscillare la
            porta non permette di farsi ripagare lo stesso tratto;
          · SATURAZIONE al bersaglio: oltre il goal `_lavoro` non cresce piu',
            quindi l'overshoot non paga.
        """
        svolto = self._lavoro
        delta = max(0.0, svolto - self._p)
        self._p = max(self._p, svolto)
        return delta


# §4 — i cinque punti in cui A è usata. Il test `test_unified.py` verifica
# che siano esattamente questi cinque e che nessun altro riformuli la condizione.
USI_DI_A = (
    "transizione MOVE -> HOLD (bilaterale, uguale per i due compiti)",
    "timer di mantenimento: sale se A, scende se non A",
    "termine `target`: vale 1 al bersaglio e cala allontanandosi",
    "condizione di terminazione",
    "bonus di successo",
)
