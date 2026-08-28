#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
config_unified.py — UNA sola configurazione per i due compiti.

Documento di riferimento: tabella_reward_machine_unificata.md
  §1  i pesi dei diciassette termini              -> Weights
  §6  i nove parametri che distinguono i compiti   -> TaskSpec
  §6  SAC e soglie adattive, invariati             -> SacHyper, AdaptiveThresholds

REGOLA DI RICONCILIAZIONE DEI PESI (§6)
---------------------------------------
I pesi dei due progetti coincidono quasi ovunque: l'apertura li ha spostati in
configurazione, la chiusura li tiene come letterali, ma il VALORE è lo stesso.
Divergono in quattro punti soltanto, elencati nel §6: dove divergono si adotta
il valore della CHIUSURA, cioe' del compito che risulta risolto.
"""
from dataclasses import dataclass, field
from typing import Tuple


# ═════════════════════════════════════════════════════════════════════════════
# §6 — I NOVE PARAMETRI CHE DEFINISCONO IL COMPITO
# ═════════════════════════════════════════════════════════════════════════════
@dataclass
class TaskSpec:
    """I nove valori del §6. Nessuno di essi è un termine di ricompensa.

    Quattro sono di SPECIFICA (che cosa vuoi ottenere), tre sono GEOMETRICI
    (com'è fatta la scena), due sono INTERRUTTORI di controllori.
    """
    nome: str

    # --- specifica ---------------------------------------------------------
    # θ* e θ₀ sono espressi come FRAZIONI della corsa della cerniera, come nei
    # due progetti originali (`close_fraction`, `goal_frac_*`, `init_open_*`).
    # Così i due compiti sono davvero speculari: la chiusura parte dalla porta
    # aperta al 70–100 % e punta all'1.5 %, l'apertura parte da 0 % e punta
    # all'85–100 %. La conversione in radianti la fa l'ambiente, che conosce la
    # corsa reale del giunto.
    theta_star_frac: Tuple[float, float]   # θ*  bersaglio, frazione della corsa
    theta_zero_frac: Tuple[float, float]   # θ₀  posa iniziale, frazione della corsa
    tol: float                        # tol tolleranza attorno al bersaglio [rad]
    t_hold_s: float                   # T_hold durata del mantenimento [s]

    # --- geometria ---------------------------------------------------------
    d_ret: float                      # distanza del punto di ritiro [m]
    z_ret: float                      # offset verticale del punto di ritiro [m]
    orienta_normale: bool             # gira la normale verso il robot

    # --- interruttori di controllori (override sull'azione, non reward) -----
    riporto_leva: bool                # C0 riporto attivo della leva
    escape: bool                      # C1 sfilamento guidato

    def campiona_bersaglio(self, corsa_min: float, corsa_max: float, rng) -> float:
        """θ* in radianti. Fisso nella chiusura, campionato nell'apertura."""
        return self._campiona(self.theta_star_frac, corsa_min, corsa_max, rng)

    def campiona_posa_iniziale(self, corsa_min: float, corsa_max: float, rng) -> float:
        """θ₀ in radianti: da dove parte la porta a ogni episodio."""
        return self._campiona(self.theta_zero_frac, corsa_min, corsa_max, rng)

    @staticmethod
    def _campiona(frac, corsa_min, corsa_max, rng) -> float:
        lo, hi = frac
        ampiezza = corsa_max - corsa_min
        f = lo if lo == hi else float(rng.uniform(lo, hi))
        return float(corsa_min + f * ampiezza)


# I DUE COMPITI. Sono gli unici due oggetti che cambiano fra i due esperimenti.
# Le frazioni sono quelle dei due config_v2.py originali, non scelte da me:
#   chiusura  init_open_min/max_fraction = 0.7 / 1.0   ·  close_fraction = 0.015
#   apertura  porta chiusa al reset                    ·  goal_frac_min/max = 0.85 / 1.0
# §5 bis — LA CHIUSURA PRENDE I CONTROLLORI DELL'APERTURA.
#
# La reward machine non cambia: cambiano i parametri del compito, e i valori
# nuovi sono quelli che l'apertura ha gia'. Ogni riga ha la sua misura.
#
#   riporto_leva  False -> True
#     Quattro dei cinque fallimenti della chiusura sono «porta chiusa alla
#     perfezione, maniglia ancora girata» (leva +1.674, +1.284, +1.572, +0.724).
#     L'apertura ha questo controllore e non perde un episodio sulla leva.
#     Misurato con il modello attuale: si accende in 48 episodi su 60, cioe'
#     dove serve, e resta spento a leva gia' a riposo.
#
#   escape        False -> True
#     Guida il braccio fuori invece di lasciarlo vagare: la rettilineita' del
#     ritiro passa da 0.654 a 0.793 gia' con il modello attuale, mai addestrato
#     con i controllori accesi.
#
#   orienta_normale  False -> True
#     Va insieme a `escape`. Senza, la direzione di fuga punta all'opposto di
#     dove il braccio va davvero: coseno −0.486, negativo in 27 episodi su 30.
#     Orientata verso il robot diventa +0.486. Accendere `escape` da solo fa
#     combattere il braccio contro il controllore: RELEASE da 30 a 49 passi e
#     2 true success persi su 60.
#
#   d_ret 0.13 -> 0.25 e z_ret +0.04 -> 0.0
#     Con `escape` acceso `d_ret` non e' piu' solo ricompensa: e' il bersaglio
#     del controllore. C1 si spegne a `retreat_settle_dist` = 0.06 dal
#     bersaglio, quindi con 0.13 la fuga ha appena 0.07 m di corsa. Misurato su
#     un addestramento con 0.13: spostamento del ritiro 0.084 m contro 0.190 m,
#     e clean success da 0.970 a 0.688, perche' la soglia del clean e' 0.06 m.
#     Con 0.25 — il valore dell'apertura, che ritira 0.164 m negli stessi passi
#     — la fuga ha la corsa che le serve.
CHIUSURA_V2_CURR1 = TaskSpec(
    nome="chiusura_v2_curr1",
    theta_star_frac=(0.015, 0.015),      # bersaglio: porta contro il telaio
    theta_zero_frac=(0.70, 1.00),        # parte APERTA
    tol=0.03, t_hold_s=2.0,
    d_ret=0.25, z_ret=0.0, orienta_normale=True,
    riporto_leva=True, escape=True,
)

APERTURA_V2_CURR1 = TaskSpec(
    nome="apertura_v2_curr1",
    theta_star_frac=(0.85, 1.00),        # bersaglio campionato a ogni reset
    theta_zero_frac=(0.0, 0.0),          # parte CHIUSA
    tol=0.05, t_hold_s=1.0,
    d_ret=0.25, z_ret=0.0, orienta_normale=True,
    riporto_leva=True, escape=True,
)

TASKS = {"close": CHIUSURA_V2_CURR1, "open": APERTURA_V2_CURR1}


# ═════════════════════════════════════════════════════════════════════════════
# §6 — QUELLO CHE NON SI TOCCA: iperparametri di SAC
# ═════════════════════════════════════════════════════════════════════════════
@dataclass
class SacHyper:
    """Letti dai best_model.zip consegnati. INVARIATI rispetto agli originali.

    `target_entropy` e' l'unico valore che resta per-compito (−3.0 chiusura,
    +1.0 apertura), esattamente come nei due progetti: non appartiene alla
    reward machine. La tesi (§6.2.2) misura che con −3.0 la chiusura converge a
    `ent_coef` ≈ 3·10⁻⁴ e risolve il compito, quindi non c'e' motivo di toccarlo.
    """
    net_arch: Tuple[int, int] = (512, 512)
    learning_rate: float = 3e-4
    buffer_size: int = 1_000_000
    batch_size: int = 256
    gamma: float = 0.95
    tau: float = 0.005
    gradient_steps: int = 1
    learning_starts: int = 20_000
    ent_coef: str = "auto"
    target_update_interval: int = 1
    use_sde: bool = False
    control_freq: int = 30
    n_envs: int = 8
    horizon: int = 600
    target_entropy: float = -3.0        # close −3.0 · open +1.0 (§6)


# ═════════════════════════════════════════════════════════════════════════════
# §6 — soglie adattive: invariate [Konidaris & Barto 2009; ManipForce 2015]
# ═════════════════════════════════════════════════════════════════════════════
@dataclass
class AdaptiveThresholds:
    grasp_dist_k_radius: float = 1.5     # 1.5·raggio + 0.005
    grasp_dist_k_offset: float = 0.005
    grip_thresh_base: float = 0.75       # 0.75 − 0.10·attrito normalizzato
    grip_thresh_k_fric: float = 0.10
    friction_min: float = 0.05
    friction_max: float = 2.00
    hold_k_stiffness: float = 0.5        # timer allungato con molla più rigida
    # §7 — LA SOGLIA DI SBLOCCO DEL CHIAVISTELLO, misurata sul modello MuJoCo.
    # Non e' un parametro di compito: e' una proprieta' GEOMETRICA della porta,
    # identica nei due versi e su tutti i modelli campionati. Con la leva a
    # riposo la porta si ferma a 0.182 rad chiudendo e a 0.015 aprendo; oltre
    # |leva| = 1.223 (chiusura) e 1.228 (apertura) il chiavistello libera il
    # montante e la corsa e' completa. Bisezione su 5 modelli randomizzati:
    # sempre gli stessi due valori.
    leva_rilascio: float = 1.23
    quota_leva: float = 0.30             # §7 quota del budget `progress` alla leva
    # §4 — IL VERSO DELLA MANIGLIA. Come `leva_rilascio`, e' geometria della
    # porta e non parametro di compito: la maniglia si ABBASSA, in entrambi i
    # compiti, come nel mondo reale. Sul modello MuJoCo il giunto
    # `Door_latch_joint` (asse Y, corsa ±1.57) porta la maniglia a z = 1.000 con
    # leva +1.5 e a z = 1.150 con leva −1.5, contro z = 1.075 a riposo: il verso
    # che abbassa e' il POSITIVO.
    verso_leva: float = +1.0
    # §9 — CODA DEL `--play`. Non sono soglie di compito: sono i due numeri che
    # decidono quanto a lungo mostrare il braccio che si ferma DOPO che
    # l'episodio e' finito. Misurati: alla terminazione i giunti vanno a 3.57
    # rad/s (chiusura) e 2.86 rad/s (apertura), e servono in media 46 e 37 passi
    # perche' scendano sotto 0.05. Con 90 si copre anche la coda piu' lunga
    # osservata. Non entrano in nessuna metrica.
    assestamento_max_passi: int = 90
    assestamento_vel_soglia: float = 0.05
    grasp_confirm_steps: int = 5         # REACH -> MOVE
    grasp_lose_steps: int = 3            # MOVE  -> REACH
    # §3 ISTERESI DELLA PRESA (close_generalized_v2/fsm_v2.py, ramo PHASE_PUSH,
    # «Schmitt trigger + hysteresis»). Si afferra a 0.035 m, si perde solo oltre
    # questa tolleranza, che cresce con la velocita' della porta.
    lose_dist_base: float = 0.05         # tolleranza di perdita a porta ferma
    lose_dist_k_speed: float = 0.5       # + 0.5·|dθ/dt|
    lose_dist_max: float = 0.12          # tetto
    grip_release_margin: float = 0.20    # si perde sotto soglia − 0.20
    near_target_tol: float = 0.05        # al bersaglio le dita possono aprirsi
    retreat_settle_dist: float = 0.06
    # §7 — soglia del ritiro, presa dal progetto originale
    # (`scratch/test_open_task_v2/_common.py`, STUCK_MOVE_THRESH): la mano deve
    # essersi allontanata di almeno 6 cm dalla posa in cui ha lasciato la
    # maniglia. NON e' «distanza da una posa bersaglio».
    ritiro_spostamento_min: float = 0.06
    retreat_target_steps: int = 30
    retreat_hard_cap: int = 120
    latch_term_tol: float = 0.15         # |latch| ≤ tol_leva per terminare
    stall_guard_steps: int = 20
    # §8 C0 — riporto della leva, costanti del sorgente dell'apertura
    riporto_guadagno: float = 2.0        # ampiezza = 2.0·|leva|, con tetto
    riporto_mag_max: float = 0.6
    riporto_rampa: int = 4               # avvio morbido: niente strappo alla porta
    riporto_rot_gain: float = 0.5        # rotazione del polso lungo l'arco
    gabbia_margine: float = 0.015        # dita a diametro + margine: la barra ruota
    grip_lock_margin: float = 0.10   # [=] morsa in MOVE/HOLD: soglia + margine          # §3 guardia di stallo (NUOVA)


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
    smooth: float = 0.1              # [C] close 0.3; assente nell'apertura. Il config
    # originale annota: «was 1.0 — high smoothness penalty was teaching "don't move"».
    # L'apertura ha una fase esplorativa molto piu' lunga e NON ricompensata — girare
    # la leva di ~1.5 rad non muove la cerniera, quindi non paga nulla — e li' 0.3
    # tassava esattamente il movimento da esplorare: nel training di apertura `smooth`
    # e' il termine dominante di MOVE (-0.40/passo su 95 % dei passi, porta ferma).
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

    # 10-11 IL MECCANISMO DI ARRESTO (tesi §6.4.1, classe F) -----------------
    # Nella chiusura l'arresto sul bersaglio lo produce la FISICA: il pannello
    # si appoggia al telaio. Nell'apertura il bersaglio e' un punto interno e la
    # cerniera non ha molla di richiamo, quindi lo stesso arresto va scritto
    # nella ricompensa: rampa dentro la tolleranza, rimbalzo fuori, e
    # smorzamento della velocita' residua.
    target: float = 1.0              # [N] rampa clip(1−|e|/tol,0,1) — §1
    target_bounce: float = 20.0      # [C] `hold_bounce`: −20·|e| fuori tolleranza, in HOLD
    damp: float = 25.0               # [C] `hold_veldamp`: −25·|dθ/dt|, solo in HOLD
    damp_floor: float = 0.01         # sotto 0.01 rad/s non interviene

                                     # la porta e' ferma e il termine non interviene

    # 12 still ------------------------------------------------------------
    still_bonus: float = 1.0         # [=] +1 se fermo
    still_act: float = 2.0           # [=] -2.0·‖a‖
    still_jnt: float = 1.0           # [=] close hold_jnt_freeze -1.0
    still_release: float = 20.0       # [C] peso dell'immobilita' in RELEASE
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
    success: float = 600.0       # [C] 500 (terminazione in `base`) + 100 (`success_bonus`)

    # taglio finale sul reward dello step
    clip_step: float = 100.0         # [C] close ±100 · open ±50

    def w_progress(self, escursione: float) -> float:
        """§7 — il peso del progresso è NORMALIZZATO sull'escursione del compito.

        Oggi i due progetti pagano budget diversi per lo stesso gesto:
        chiusura 2000 × 0.35 ≈ 700, apertura 300 × 0.37 ≈ 111. È la ragione per
        cui i ritorni non sono confrontabili (§7 e §5, punto 2).
        Qui il budget totale è fissato una volta sola e il peso ne discende:
        percorrere tutta l'escursione paga `budget_progress` in entrambi i casi.
        """
        return self.budget_progress / max(abs(escursione), 1e-6)


@dataclass
class DomainRand:
    """§6 — randomizzazione del dominio: INVARIATA [Tobin et al. 2017].

    I valori sono copiati dai due `config_v2.py`, dove sono identici. Servono
    perché `ExtendedDomainRandomizer` e `MultiApproachGrasp` — moduli che il §6
    dichiara invariati e che quindi NON vanno riscritti — li leggono dal cfg.
    """
    grasp_n_candidates: int = 3           # [ten Pas et al. 2017]
    rand_latch_stiffness: bool = True
    rand_latch_stiffness_min: float = 0.5
    rand_latch_stiffness_max: float = 2.0
    rand_hinge_damping: bool = True
    rand_hinge_damping_min: float = 0.3
    rand_hinge_damping_max: float = 1.5
    rand_door_mass: bool = True
    rand_door_mass_min: float = 0.5
    rand_door_mass_max: float = 2.0


@dataclass
class UnifiedConfig:
    """La configurazione completa di un esperimento.

    Espone anche gli attributi di `DomainRand` al primo livello (`__getattr__`),
    perché i moduli invarianti li cercano come `cfg.grasp_n_candidates`.
    """
    task: TaskSpec
    sac: SacHyper = field(default_factory=SacHyper)
    thr: AdaptiveThresholds = field(default_factory=AdaptiveThresholds)
    w: Weights = field(default_factory=Weights)
    rand: DomainRand = field(default_factory=DomainRand)
    curriculum_level: float = 1.0        # curr 1 per entrambi i compiti
    # §6 (tesi §6.2.1): filtro EMA sull'azione, invariato dagli originali
    action_alpha: float = 0.95

    def __getattr__(self, nome: str):
        """Ponte verso i moduli invarianti, che leggono cfg.<parametro>."""
        try:
            return getattr(object.__getattribute__(self, "rand"), nome)
        except AttributeError:
            raise AttributeError(f"{type(self).__name__} non ha {nome!r}") from None

    @staticmethod
    def per(nome: str) -> "UnifiedConfig":
        """`nome` ∈ {"close", "open"} — l'unica scelta da fare."""
        if nome not in TASKS:
            raise ValueError(f"compito sconosciuto: {nome!r} (attesi: {sorted(TASKS)})")
        cfg = UnifiedConfig(task=TASKS[nome])
        cfg.sac.target_entropy = -3.0 if nome == "close" else 1.0
        return cfg

    def hold_steps(self, latch_stiffness: float = 1.0, base_stiffness: float = 1.0) -> int:
        """T_hold in step, allungato con la rigidità della molla (§6, invariato)."""
        import numpy as np
        base = self.task.t_hold_s * self.sac.control_freq
        norm = float(np.clip((latch_stiffness - base_stiffness) / (base_stiffness + 1e-8), 0.0, 1.0))
        return int(base + self.thr.hold_k_stiffness * norm * self.sac.control_freq * 0.5)
