#!/usr/bin/env python3

from dataclasses import dataclass
import numpy as np

@dataclass
class DoorState:
    """§4 — Le quattro grandezze, calcolate e non configurate.

    | grandezza | definizione                                   |
    |-----------|-----------------------------------------------|
    | σ         | sign(θ* − θ₀)   verso del compito             |
    | e         | θ − θ*          errore con segno              |
    | A         | |e| ≤ tol       soglia di stato porta         |
    | p         | cricchetto sul lavoro svolto, saturato        |
    """

    theta_star: float          # θ*  bersaglio, fissato al reset
    theta_zero: float          # θ₀  angolo iniziale, fissato al reset
    tol:        float          # tolleranza

    theta: float = 0.0         # θ   angolo corrente
    leva: float = 0.0          # angolo corrente della leva
    leva_rilascio: float = 0.0 # |leva| che libera il chiavistello (§7)
    quota_leva: float = 0.30   # §7 quota del budget `progress` che spetta alla leva
    verso_leva: float = +1.0   # §4 verso che ABBASSA la maniglia (vedi `_lavoro`)
    _p: float = 0.0            # riferimento del cricchetto
    _ancorato: bool = False    # il riferimento e' stato posato (ingresso in MOVE)

    @property
    def sigma(self) -> float:
        s = np.sign(self.theta_star - self.theta_zero)
        return float(s) if s != 0.0 else 1.0

    @property
    def e(self) -> float:
        return self.theta - self.theta_star

    @property
    def A(self) -> bool:
        """§4 — LA SOGLIA DI STATO PORTA"""
        return abs(self.e) <= self.tol

    @property
    def escursione(self) -> float:
        """Il LAVORO TOTALE del compito, in radianti: leva + cerniera.
        Vale |θ* − θ₀| da sola quando `leva_rilascio` e' 0 (chiusura).
        """
        return abs(self.theta_star - self.theta_zero) + self._leva_equivalente

    @property
    def _lavoro(self) -> float:
        corsa      = abs(self.theta_star - self.theta_zero)
        k          = (self._leva_equivalente / self.leva_rilascio) if self.leva_rilascio > 1e-9 else 0.0
        leva_utile = max(self.verso_leva * self.leva, 0.0)
        return (k * min(leva_utile, self.leva_rilascio) + float(np.clip(self.sigma * (self.theta - self.theta_zero), 0.0, corsa)))

    @property
    def _leva_equivalente(self) -> float:
        q = float(self.quota_leva)
        return abs(self.theta_star - self.theta_zero) * q / max(1.0 - q, 1e-6)

    @property
    def avanzamento(self) -> float:
        if self.escursione < 1e-9:
            return 0.0
        return float(np.clip(self._lavoro / self.escursione, 0.0, 1.0))

    @property
    def p(self) -> float:
        return self._p

    @property
    def target_ramp(self) -> float:
        return float(np.clip(1.0 - abs(self.e) / max(self.tol, 1e-9), 0.0, 1.0))

    # ── Aggiornamento ------------------------------------------------------
    def reset(self, theta_zero: float, theta_star: float, tol: float, leva_rilascio: float = 0.0, quota_leva: float = 0.30, verso_leva: float = +1.0) -> None:
        self.theta_zero    = float(theta_zero)
        self.theta_star    = float(theta_star)
        self.tol           = float(tol)
        self.leva_rilascio = float(leva_rilascio)
        self.quota_leva    = float(quota_leva)
        self.verso_leva    = float(verso_leva)
        self.theta         = float(theta_zero)
        self.leva          = 0.0
        self._p            = 0.0
        self._ancorato     = False

    def step(self, theta: float, leva: float = 0.0) -> None:
        self.theta = float(theta)
        self.leva  = float(leva)

    def ancora(self) -> None:
        if not self._ancorato:
            self._ancorato = True
            self._p = self._lavoro

    def incassa(self) -> float:
        svolto  = self._lavoro
        delta   = max(0.0, svolto - self._p)
        self._p = max(self._p, svolto)
        return delta

USI_DI_A = (
    "transizione MOVE -> HOLD (bilaterale, uguale per i due compiti)",
    "timer di mantenimento: sale se A, scende se non A",
    "termine `target`: vale 1 al bersaglio e cala allontanandosi",
    "condizione di terminazione",
    "bonus di successo",
)