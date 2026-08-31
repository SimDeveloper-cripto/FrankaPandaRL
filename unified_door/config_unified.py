#!/usr/bin/env python3

from dataclasses import dataclass, field
from typing import Tuple


# ═════════════════════════════════════════════════════════════════════════════
# §6 — I NOVE PARAMETRI CHE DEFINISCONO IL COMPITO
# ═════════════════════════════════════════════════════════════════════════════
@dataclass
class TaskSpec:
    nome: str

    theta_star_frac: Tuple[float, float]   # θ*  bersaglio, frazione della corsa
    theta_zero_frac: Tuple[float, float]   # θ₀  posa iniziale, frazione della corsa
    tol: float                             # tol tolleranza attorno al bersaglio [rad]
    t_hold_s: float                        # T_hold durata del mantenimento [s]

    # --- geometria ---------------------------------------------------------
    d_ret: float                      # distanza del punto di ritiro [m]
    z_ret: float                      # offset verticale del punto di ritiro [m]
    orienta_normale: bool             # gira la normale verso il robot

    # --- interruttori di controllori (override sull'azione, non reward) -----
    riporto_leva: bool                # C0 riporto attivo della leva
    escape: bool                      # C1 sfilamento guidato

    leva_consegna: float

    def campiona_bersaglio(self, corsa_min: float, corsa_max: float, rng) -> float:
        return self._campiona(self.theta_star_frac, corsa_min, corsa_max, rng)

    def campiona_posa_iniziale(self, corsa_min: float, corsa_max: float, rng) -> float:
        return self._campiona(self.theta_zero_frac, corsa_min, corsa_max, rng)

    @staticmethod
    def _campiona(frac, corsa_min, corsa_max, rng) -> float:
        lo, hi   = frac
        ampiezza = corsa_max - corsa_min
        f = lo if lo == hi else float(rng.uniform(lo, hi))
        return float(corsa_min + f * ampiezza)

CHIUSURA_V2_CURR1 = TaskSpec(
    nome = "chiusura_v2_curr1",
    theta_star_frac = (0.015, 0.015),      # bersaglio: porta contro il telaio
    theta_zero_frac = (0.70, 1.00),        # parte APERTA
    tol = 0.03, t_hold_s = 2.0,
    d_ret = 0.25, z_ret = 0.0, orienta_normale = True,
    riporto_leva = True, escape = True, leva_consegna = 0.40,
)

APERTURA_V2_CURR1 = TaskSpec(
    nome = "apertura_v2_curr1",
    theta_star_frac = (0.85, 1.00),        # bersaglio campionato a ogni reset
    theta_zero_frac = (0.0, 0.0),          # parte CHIUSA
    tol = 0.05, t_hold_s = 1.0,
    d_ret = 0.25, z_ret = 0.0, orienta_normale = True,
    riporto_leva = True, escape = True, leva_consegna = 0.15,
)

TASKS = {"close": CHIUSURA_V2_CURR1, "open": APERTURA_V2_CURR1}

# ═════════════════════════════════════════════════════════════════════════════
# §6 — QUELLO CHE NON SI TOCCA: iperparametri di SAC
# ═════════════════════════════════════════════════════════════════════════════
@dataclass
class SacHyper:
    net_arch: Tuple[int, int]   = (512, 512)
    learning_rate: float        = 3e-4
    buffer_size: int            = 1_000_000
    batch_size: int             = 256
    gamma: float                = 0.95
    tau: float                  = 0.005
    gradient_steps: int         = 1
    learning_starts: int        = 20_000
    ent_coef: str               = "auto"
    target_update_interval: int = 1
    use_sde: bool               = False
    control_freq: int           = 30
    n_envs: int                 = 8
    horizon: int                = 600
    target_entropy: float       = -3.0

# ═════════════════════════════════════════════════════════════════════════════
# §6 — soglie adattive: invariate [Konidaris & Barto 2009; ManipForce 2015]
# ═════════════════════════════════════════════════════════════════════════════
@dataclass
class AdaptiveThresholds:
    grasp_dist_k_radius: float = 1.5     # 1.5·raggio + 0.005
    grasp_dist_k_offset: float = 0.005
    grip_thresh_base: float    = 0.75    # 0.75 − 0.10·attrito normalizzato
    grip_thresh_k_fric: float  = 0.10
    friction_min: float        = 0.05
    friction_max: float        = 2.00
    hold_k_stiffness: float    = 0.5     # timer allungato con molla più rigida

    leva_rilascio: float = 1.23
    quota_leva: float    = 0.30

    verso_leva: float = +1.0

    assestamento_max_passi: int    = 90
    assestamento_vel_soglia: float = 0.05

    sgombero_max_passi: int        = 60
    sgombero_dist_obiettivo: float = 0.30

    sgombero_alzata: float         = 0.04
    sgombero_alzata_max_passi: int = 20

    assestamento_pausa_s: float    = 3.0
    grasp_confirm_steps: int       = 5         # REACH -> MOVE
    grasp_lose_steps: int          = 3         # MOVE  -> REACH

    lose_dist_base: float      = 0.05      # tolleranza di perdita a porta ferma
    lose_dist_k_speed: float   = 0.5       # + 0.5·|dθ/dt|
    lose_dist_max: float       = 0.12      # tetto
    grip_release_margin: float = 0.20      # si perde sotto soglia − 0.20
    near_target_tol: float     = 0.05      # al bersaglio le dita possono aprirsi
    retreat_settle_dist: float = 0.06

    ritiro_spostamento_min: float = 0.06
    retreat_target_steps: int     = 30
    retreat_hard_cap: int         = 120
    latch_term_tol: float         = 0.15   # |latch| ≤ tol_leva per terminare
    stall_guard_steps: int        = 20
    riporto_guadagno: float       = 2.0    # ampiezza = 2.0·|leva|, con tetto


    riporto_mag_max: float  = 1.0
    riporto_rampa: int      = 4      # avvio morbido: niente strappo alla porta
    riporto_rot_gain: float = 0.5    # rotazione del polso lungo l'arco
    gabbia_margine: float   = 0.015  # dita a diametro + margine: la barra ruota
    grip_lock_margin: float = 0.10   # [=] morsa in MOVE/HOLD: soglia + margine


# ═════════════════════════════════════════════════════════════════════════════
# §1 — I PESI DEI 16 TERMINI
# ═════════════════════════════════════════════════════════════════════════════
@dataclass
class Weights:
    """Ogni peso è annotato con la sua provenienza nei due sorgenti.

    Legenda della colonna di commento:
      [=]  identico nei due progetti: adottato senza decisioni
      [C]  divergente: adottato il valore della CHIUSURA (regola di §9)
      [A]  divergente: adottato il valore dell'APERTURA (prescritto dal documento)
      [N]  nuovo: non esiste in nessuno dei due, introdotto dal progetto
    """
    # 1 time --------------------------------------------------------------
    time_cost: float = 0.10          # [A] open rew["base"]=-0.10; close ereditava train_close.py

    # 2 smooth ------------------------------------------------------------
    smooth: float = 0.1              # [C] close 0.3; assente nell'apertura.
    smooth_act: float = 0.005        # [C] close act_pen

    # 3 phi — shaping potenziale [Ng, Russell & Harada 1999] --------------
    phi_reach: float = 2.0           # [C] close 2.0 · open 25.0
    phi_move: float = 3.0            # [C] close phi_push 3.0 · open phi_pull 5.0
    phi_hold: float = 2.0            # [C] close 2.0 · open 5.0
    phi_release: float = 2.0         # [C] close 2.0 · open 5.0
    phi_clip: float = 10.0           # [C] close clip ±10 · open nessun taglio
    curriculum_k: float = 0.5        # [C] close 0.5 · open 0.0

    # 4 approach ----------------------------------------------------------
    app_3d: float = 5.0              # [=] close -5.0 · open w_reach_dist_3d=5.0
    app_xy: float = 3.0              # [=] close -3.0 · open w_reach_dist_xy=3.0
    app_z: float = 15.0              # [=] close -15.0 · open w_reach_dist_z=15.0

    # 5 approach_geom -----------------------------------------------------
    geom_below: float = 3.0          # [=] close -3.0 · open w_reach_app_blw=3.0
    geom_above: float = 1.5          # [=] close -1.5 · open w_reach_app_top=1.5
    geom_band: Tuple[float, float] = (-0.005, 0.03)   # [=] banda del §1

    # 6 wrist -------------------------------------------------------------
    wrist_reach: float = 1.5         # [C] close w_multi_align=1.5; assente nell'apertura
    wrist_flat: float = 0.5          # [C] close -0.5
    wrist_hold: float = 2.0          # [C] close hold_flat -2.0

    # 7 grip --------------------------------------------------------------
    grip_far: float = 1.0            # [=] -1.0·(g+0.85)
    grip_near: float = 2.5           # [=] close 2.5 · open w_reach_grip_near=2.5
    grip_weak_move: float = 2.0      # [=] close -2.0 · open w_pull_grip_weak=2.0

    # 8 progress — normalizzato, vedi budget_progress ---------------------
    budget_progress: float = 700.0   # [C] close 2000·0.35 ≈ 700 · open 300·0.37 ≈ 111

    # 9 contact -----------------------------------------------------------
    contact: float = 0.5             # [=] w_grip_contact=0.5

    target: float = 1.0              # [N] rampa clip(1−|e|/tol,0,1)
    target_bounce: float = 20.0      # [C] `hold_bounce`: −20·|e| fuori tolleranza, in HOLD
    damp: float = 25.0               # [C] `hold_veldamp`: −25·|dθ/dt|, solo in HOLD
    damp_floor: float = 0.01         # sotto 0.01 rad/s non interviene

    # 12 still ------------------------------------------------------------
    still_bonus: float = 1.0         # [=] +1 se fermo
    still_act: float = 2.0           # [=] -2.0·‖a‖
    still_jnt: float = 1.0           # [=] close hold_jnt_freeze -1.0
    still_release: float = 20.0      # [C] peso dell'immobilita' in RELEASE
    still_eps: float = 0.05          # [=] soglia di immobilità

    # 13 hold_grip --------------------------------------------------------
    hg_bonus: float = 1.0            # [=] +1 presa salda
    hg_slip: float = 5.0             # [=] close -5.0 · open w_hold_slip=5.0
    hg_loose: float = 2.0            # [=] -2.0·|g − soglia|
    hg_drop: float = 10.0            # [=] close -10.0 · open w_hold_drop_pen=10.0
    hg_dist: float = 3.0             # [=] close -3.0 · open w_hold_dist=3.0
    hg_dist_max: float = 0.06        # [=] 6 cm

    # 14 release ----------------------------------------------------------
    rel_open: float = 2.0            # [=] ret_grip +2.0
    rel_closed: float = 1.0          # [=] -1.0·|g+1|
    rel_clean: float = 1.0           # [=] ret_release +1.0

    # 15 retreat ----------------------------------------------------------
    ret_dir: float = 3.0             # [=] +3.0·allineamento
    ret_perp: float = 2.0            # [=] -2.0·‖perpendicolare‖
    ret_rot: float = 3.0             # [=] -3.0·‖rotazione‖

    # 16 latch_home -------------------------------------------------------
    latch_home: float = 1.0          # [=] close -1.0 · open w_latch_ret=1.0

    # 17 success ----------------------------------------------------------
    transizione: float = 10.0        # [C] close phase_trans=10.0: una tantum, uscita da REACH
    success: float = 600.0           # [C] 500 (terminazione in `base`) + 100 (`success_bonus`)

    # taglio finale sul reward dello step
    clip_step: float = 100.0         # [C] close ±100 · open ±50

    def w_progress(self, escursione: float) -> float:
        return self.budget_progress / max(abs(escursione), 1e-6)

@dataclass
class DomainRand:
    grasp_n_candidates: int         = 3      # [ten Pas et al. 2017]
    rand_latch_stiffness: bool      = True
    rand_latch_stiffness_min: float = 0.5
    rand_latch_stiffness_max: float = 2.0
    rand_hinge_damping: bool        = True
    rand_hinge_damping_min: float   = 0.3
    rand_hinge_damping_max: float   = 1.5
    rand_door_mass: bool            = True
    rand_door_mass_min: float       = 0.5
    rand_door_mass_max: float       = 2.0


@dataclass
class UnifiedConfig:
    task: TaskSpec
    sac: SacHyper           = field(default_factory=SacHyper)
    thr: AdaptiveThresholds = field(default_factory=AdaptiveThresholds)
    w: Weights              = field(default_factory=Weights)
    rand: DomainRand        = field(default_factory=DomainRand)

    curriculum_level: float = 1.0
    action_alpha: float     = 0.95

    def __getattr__(self, nome: str):
        try:
            return getattr(object.__getattribute__(self, "rand"), nome)
        except AttributeError:
            raise AttributeError(f"{type(self).__name__} non ha {nome!r}") from None

    @staticmethod
    def per(nome: str) -> "UnifiedConfig":
        if nome not in TASKS:
            raise ValueError(f"compito sconosciuto: {nome!r} (attesi: {sorted(TASKS)})")
        cfg = UnifiedConfig(task=TASKS[nome])
        cfg.sac.target_entropy = -3.0 if nome == "close" else 1.0
        return cfg

    def hold_steps(self, latch_stiffness: float = 1.0, base_stiffness: float = 1.0) -> int:
        import numpy as np

        base = self.task.t_hold_s * self.sac.control_freq
        norm = float(np.clip((latch_stiffness - base_stiffness) / (base_stiffness + 1e-8), 0.0, 1.0))
        return int(base + self.thr.hold_k_stiffness * norm * self.sac.control_freq * 0.5)