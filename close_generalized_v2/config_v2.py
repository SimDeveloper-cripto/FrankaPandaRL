#!/usr/bin/env python3
# close_generalized_v2/config_v2.py

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Tuple, Optional


@dataclass
class TrainConfigV2:
    # ── Identity ──────────────────────────────────────────────────────────────
    seed   : int = 42
    run_dir: str = "runs/close_gen_v2"
    tb_dir : str = "runs/tb_v2"

    # ── Robosuite / Environment ────────────────────────────────────────────────
    env_name    : str = "Door"
    robot       : str = "Panda"
    horizon     : int = 600
    control_freq: int = 30

    reward_shaping      : bool  = True
    reward_scale        : float = 1.0
    use_object_obs      : bool  = True
    use_camera_obs      : bool  = False
    terminate_on_success: bool  = False

    # ── Vectorization ─────────────────────────────────────────────────────────
    num_envs    : int  = 8
    vecnormalize: bool = True

    # ── SAC Hyperparameters ────────────────────────────────────────────────────
    total_steps    : int   = 800_000 # 1_500_000
    learning_rate  : float = 3e-4
    buffer_size    : int   = 1_000_000
    batch_size     : int   = 256
    gamma          : float = 0.95
    tau            : float = 0.005
    train_freq     : int   = 1
    gradient_steps : int   = 1      # was 2 — reduce to slow down Q overfit during early exploration
    learning_starts: int   = 20_000 # was 10k — more random steps before policy locks in
    ent_coef       : str   = "auto"

    # §1.13 — Pavimento di entropia per SAC. Con il default target_entropy = −dim_azione
    # (≈ −7 per OSC_POSE), ent_coef collassa a ~7e-5 → esplorazione nulla. Se ciò avviene
    # PRIMA che la policy scopra la sequenza di chiusura, resta intrappolata nell'ottimo
    # locale di "accampamento" in REACH. Un target meno negativo mantiene l'esplorazione
    # viva attraverso la finestra di scoperta, senza impedire la convergenza finale
    # (SAC continua ad auto-tarare ent_coef verso questo target). Tunabile: per azione a
    # 4 dimensioni usare ~ −2.0.
    target_entropy : float = -3.0

    # ── Network Architecture ───────────────────────────────────────────────────
    # Slightly deeper than v1 (512, 512) to handle ~47-dim observation space
    policy_net_arch: Tuple[int, int] = (512, 512)

    # ── Evaluation ────────────────────────────────────────────────────────────
    eval_freq      : int = 10_000
    n_eval_episodes: int = 20
    checkpoint_freq: int = 200_000

    # ── Door Closing Parameters ────────────────────────────────────────────────
    close_fraction        : float = 0.015
    init_open_min_fraction: float = 0.70
    init_open_max_fraction: float = 1.00

    # ── Base Reward Weights (from original) ───────────────────────────────────
    w_progress   : float = 0.0
    w_delta      : float = 2.0
    w_action     : float = 0.0
    time_penalty : float = 0.1   # was 0.5 — reduced: -0.50/step (-300/ep) drowned shaping signal
    success_bonus: float = 100.0  # §1.14 — one-time bonus al completamento del task.
    # Dimensionato per preservare il valore dello stato di retreat-completo: con γ=0.95
    # il valore SCONTATO del "mungere" la reward di RETREAT (~+3/step) è ≈ 3/(1−γ) ≈ 60,
    # NON +2000 (quello è il ritorno NON scontato su 400 step). Un bonus ≳ 60 rende
    # "termina" preferibile a "continua", senza cliff destabilizzanti.

    # §1.14 — Terminazione al completamento del RETREAT.
    # Senza stato terminale, l'episodio gira fino all'orizzonte (ep_len=500) e la policy
    # continua a muovere il braccio per raccogliere ret_dir/ret_grip. Terminare quando il
    # retreat è sostenuto + porta chiusa + latch neutro risolve durata episodio E
    # movimento residuo del braccio.
    terminate_on_retreat_complete: bool = True
    fsm_retreat_target_steps     : int  = 30   # step sostenuti in RETREAT prima di terminare

    # §1.15 — RETREAT: immobilizzazione del braccio + rilascio pulito della maniglia.
    # La zona di "settle" è allargata rispetto al vecchio freeze a 0.02 m (soglia quasi
    # mai raggiunta → il braccio continuava a inseguire il target via ret_dir). Appena
    # entro `fsm_retreat_settle_dist` dalla posa di retreat, il braccio si FERMA
    # (penalità su tutte le DOF tranne il gripper) e riceve un bonus di rilascio solo a
    # porta chiusa + gripper aperto. La terminazione §1.14 resta invariata, quindi la
    # lunghezza d'episodio NON cambia.
    fsm_retreat_settle_dist: float = 0.06   # [m] entro cui immobilizzare invece di inseguire
    w_retreat_settle       : float = 20.0   # forza dell'immobilizzazione (= vecchio freeze)

    # §1.16 — GRIP IN CHIUSURA: peso del reward (positivo, limitato) per il mantenimento
    # del contatto fisico durante PUSH, scalato sul progresso di chiusura. Premia lo STATO
    # di buona presa (gripper_width nella banda is_physically_closed), non lo stringere di
    # più (su maniglie sottili stringere oltre perde il contatto, §3.1). Bounded e
    # non-negativo ⇒ §1.13-safe; nessun effetto a porta aperta (closing_progress≈0).
    w_grip_contact         : float = 0.5

    # §1.15 — Pinning del livello di curriculum (due modalità di training senza toccare
    # il codice esistente):
    #   None  → curriculum ADATTIVO 0→1 (comportamento attuale, via AdaptiveCurriculumV2)
    #   0.0   → POSA FISSA (riproduce il run attuale): nessuna randomizzazione di posa,
    #           fisica sempre randomizzata. Curriculum adattivo disattivato.
    #   1.0   → POSA VARIABILE piena dall'inizio (pos ±15 cm, yaw ±17°).
    # L'env ri-fissa il livello a ogni reset, quindi non può driftare.
    fixed_curriculum_level: Optional[float] = None

    # ── Return Stage ──────────────────────────────────────────────────────────
    enable_return_stage: bool  = True
    w_return_pos       : float = 2.0
    w_door_regress     : float = 4.0
    return_hold_steps  : int   = 10
    return_pos_tol     : float = 0.05

    # ── Action Smoothing ──────────────────────────────────────────────────────
    action_smooth_alpha: float = 0.95  # was 0.8 — reduced jerk, making smoothness penalty less punishing

    # ── Original Handle Randomization ─────────────────────────────────────────
    limit_handle_friction: bool  = True
    handle_friction_max  : float = 0.8
    human_dist_min       : float = 0.50
    human_dist_max       : float = 0.60

    # ══════════════════════════════════════════════════════════════════════════
    # §3.1 — Adaptive FSM Thresholds
    # Ref: Konidaris & Barto (2009) "Skill Chaining"
    # ══════════════════════════════════════════════════════════════════════════

    # Grasp distance: dist_thresh = k_r * handle_radius + k_offset
    fsm_grasp_dist_k_radius: float = 1.5   # multiplier on handle radius
    fsm_grasp_dist_k_offset: float = 0.005 # fixed offset [m]

    # Grip threshold: grip_thresh = base - k_f * norm_friction
    fsm_grip_thresh_base  : float = 0.75
    fsm_grip_thresh_k_fric: float = 0.10   # subtracted proportionally to friction

    # §1.11 — Schmitt trigger + hysteresis on grasp loss (anti-chatter).
    # Release the grasp at a LOWER threshold than required to enter PUSH, and only
    # after several consecutive bad frames. Kills the REACH<->PUSH chatter seen at 400k.
    fsm_grip_release_margin: float = 0.20  # release_thresh = grip_thresh - this
    fsm_grasp_lose_steps   : int   = 3     # consecutive bad frames before declaring loss

    # Friction normalization range (matches domain randomizer below)
    fsm_friction_min: float = 0.05
    fsm_friction_max: float = 2.00

    # HOLD timer: hold_steps = base + k_stiff * (stiff_max - current_stiff)
    fsm_hold_time_base    : float = 2.0   # [s] base hold duration
    fsm_hold_k_stiffness  : float = 0.5   # extra seconds per unit of (stiff_max - stiff)

    # Retreat direction: perpendicular to door vs fixed global axis
    fsm_retreat_dist  : float = 0.13      # [m] retreat distance along door normal
    fsm_retreat_z_off : float = 0.04      # [m] vertical offset

    # ══════════════════════════════════════════════════════════════════════════
    # §3.2 — Potential-Based Reward Shaping
    # Ref: Ng, Russell & Harada (1999) "Policy Invariance Under Reward Transformations"
    #      Devlin & Kudenko (2012) "Dynamic Potential-Based Reward Shaping"
    # ══════════════════════════════════════════════════════════════════════════

    use_potential_reward: bool = True

    # ── Potential magnitudes ──────────────────────────────────────────────────

    # CRITICAL: these are now SMALL on purpose. Potential-based shaping is an
    # auxiliary GUIDANCE term, not the objective. With the cumulative design,
    # Phi accumulates across phases (Phi_reach + Phi_push + Phi_hold + ...), and the
    # discounted shaping F = (gamma * Phi') - Phi leaves a standing drift of
    # (gamma-1) * Phi = -0.05 * Phi, per step. The previous values (25/50/5) made
    # Phi ~ 75-100 -> a -3.75..-5/step penalty that punished the agent for staying
    # in PUSH/HOLD/RETREAT (lethal, since the arm is frozen in HOLD). See §1.10.A.
    #
    # With these O(1-5) values: Phi_HOLD ~ 5-9 -> drift <= -0.5/step, fully dwarfed
    # by the genuine reward (dense reach + ratcheted door progress + +1/step hold
    # bonuses).
    # gamma stays at 0.95, so Ng et al. (1999) policy invariance is EXACT.
    # The grasp transition still gets a clean ~+gamma * w_reach bonus from the Phi jump.

    phi_reach_weight: float = 2.0    # was 25.0 — sets the REACH->PUSH shaping bonus / ladder base
    phi_reach_sigma : float = 0.40   # (unused for shaping; Phi_reach=0 in REACH by design)

    # PUSH potential: Phi_push = w * (door_max - angle)/door_max * grip_factor
    phi_push_weight : float = 3.0    # was 50.0 — closing is driven by door_prog (real R), not this

    # HOLD potential: Phi_hold = w * (duration/target) * (1 - |door_qpos|/tol)
    phi_hold_weight : float = 2.0    # was 5.0

    # RETREAT: direction-aligned reward (mostly explicit, not purely potential)
    phi_retreat_weight: float = 2.0  # was 3.0

    # Jerk / smoothness regularisation (active in all phases)
    w_smoothness: float = 0.3   # was 1.0 — high smoothness penalty was teaching "don't move"

    # ══════════════════════════════════════════════════════════════════════════
    # §3.3 — Multi-Approach Grasp
    # Ref: ten Pas et al. (2017) "Grasp Pose Detection"
    #      ManipForce (2015) "Force-based manipulation primitives"
    # ══════════════════════════════════════════════════════════════════════════

    # Number of candidate grasp approach directions
    grasp_n_candidates: int = 3    # top-down, lateral-left, lateral-right

    # Weight for multi-approach alignment reward in REACH
    w_multi_align: float = 1.5

    # ══════════════════════════════════════════════════════════════════════════
    # §3.4 — Extended Physics Randomization
    # Ref: Tobin et al. (2017) "Domain Randomization"
    #      Zhao et al. (2020) "Sim-to-Real Transfer"
    #      Mehta et al. (2020) "Active Domain Randomization"
    # ══════════════════════════════════════════════════════════════════════════

    # Latch joint stiffness: scale ∈ [scale_min, scale_max] × base
    rand_latch_stiffness     : bool  = True
    rand_latch_stiffness_min : float = 0.5
    rand_latch_stiffness_max : float = 2.0

    # Hinge joint damping: scale ∈ [scale_min, scale_max] × base
    rand_hinge_damping    : bool  = True
    rand_hinge_damping_min: float = 0.3
    rand_hinge_damping_max: float = 1.5

    # Door body mass: scale ∈ [scale_min, scale_max] × base
    rand_door_mass    : bool  = True
    rand_door_mass_min: float = 0.5
    rand_door_mass_max: float = 2.0

    # ══════════════════════════════════════════════════════════════════════════
    # §3.5 — Beta-Network (Learned Termination Functions)
    # Ref: Sutton, Precup & Singh (1999) "Between MDPs and Semi-MDPs"
    #      Konidaris & Barto (2009) "Skill Chaining"
    # ══════════════════════════════════════════════════════════════════════════

    use_beta_net          : bool  = False  # Off by default; enable in Phase 4
    beta_net_hidden       : int   = 64
    beta_net_lr           : float = 1e-4
    beta_net_reg          : float = 1e-3   # L2 regularisation on β output

    # ══════════════════════════════════════════════════════════════════════════
    # §3.6 — Curriculum Reward Co-Evolution
    # Ref: Devlin & Kudenko (2012) "Dynamic Potential-Based Reward Shaping"
    #      Portelas et al. (2020) "Automatic Curriculum Learning"
    # ══════════════════════════════════════════════════════════════════════════

    # k_curr: at curriculum_level = 1.0, reward weights increase by k_curr × 100%
    curriculum_reward_k    : float = 0.5

    # Phase-time efficiency criterion for curriculum advancement
    # (max steps allowed per phase to be considered "efficient")
    curriculum_max_reach_steps: int = 25
    curriculum_max_push_steps : int = 25
    curriculum_check_freq     : int = 25_000
    curriculum_advance_delta  : float = 0.10

    # §1.12 — Curriculum gate (success-driven, windowed). See AdaptiveCurriculumV2.
    # Advance when RECENT success is reliable; grasp_floor is a low anti-collapse guard,
    # NOT a performance bar (post-§1.11 a perfect episode does ~1 grasp).
    curriculum_success_thresh : float = 0.85
    curriculum_grasp_floor    : float = 0.20

    # ── Diagnostic ────────────────────────────────────────────────────────────
    debug_print_every: int = 200  # :)