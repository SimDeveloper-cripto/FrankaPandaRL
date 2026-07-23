#!/usr/bin/env python3
# open_generalized_v2/config_v2.py
#
# TrainConfigV2Open — configurazione del task di APERTURA GENERALIZZATA (v2),
# costruita SPECULARE alla v2 di chiusura (close_generalized_v2/config_v2.py).
#
# Differenza concettuale chiusura ↔ apertura:
#   - chiusura: l'obiettivo è portare la porta a door_angle ≈ 0 (chiusa) e far
#     stabilizzare il latch; le fasi sono REACH → PUSH → HOLD(chiuso) → RETREAT.
#   - apertura: l'obiettivo è portare la porta a door_angle ≈ goal_angle (aperta al
#     valore richiesto) e mantenerla aperta; le fasi sono REACH → PULL → HOLD_OPEN →
#     RETREAT.  Il "successo fisico" è |door_angle − goal_angle| ≤ tolleranza.
#
# Tutto è pensato per il SOLO curriculum 1 (posa variabile, soglie adattive):
#   fixed_curriculum_level = 1.0  → posizione/yaw porta randomizzati + fisica randomizzata.
#
# Letteratura (citazioni, vedi docs/UPDATE_OPEN_V2.md):
#   [1]  Sutton, Precup & Singh (1999) — opzioni/semi-MDP, fasi come opzioni.
#   [2]  Konidaris & Barto (2009)      — skill chaining, precondizioni di opzione.
#   [3]  Ng, Harada & Russell (1999)   — potential-based reward shaping (invarianza).
#   [8]  Tobin et al. (2017)           — domain randomization.
#   [13] ManipForce (2015)             — soglie adattive di presa/forza al contatto.
#   [15] ten Pas et al. (2017)         — grasp come posa 6-D, qualità di presa.
#   [17] Zhao et al. (2020)            — physics randomization per sim-to-real.

from __future__ import annotations

from typing import Tuple
from dataclasses import dataclass


@dataclass
class TrainConfigV2Open:
    # ── Generale ──────────────────────────────────────────────────────────────
    seed   : int = 42
    run_dir: str = "runs/open_gen_v2"
    tb_dir : str = "runs/tb_open_v2"

    env_name    : str = "Door"
    robot       : str = "Panda"
    horizon     : int = 600
    control_freq: int = 30

    use_object_obs      : bool = True
    use_camera_obs      : bool = False
    terminate_on_success: bool = False

    num_envs    : int  = 8
    vecnormalize: bool = True

    # ── SAC ───────────────────────────────────────────────────────────────────
    total_steps    : int   = 1_500_000   # curriculum 1: posa variabile, serve più budget
    learning_rate  : float = 3e-4
    buffer_size    : int   = 1_000_000
    batch_size     : int   = 256
    gamma          : float = 0.95
    tau            : float = 0.005
    train_freq     : int   = 1
    gradient_steps : int   = 1      # allineato alla chiusura v2: rallenta overfit del Q in early exploration
    learning_starts: int   = 20_000   # allineato alla chiusura v2: più esplorazione prima del lock-in
    ent_coef       : str   = "auto"
    target_entropy : float = 1.0         # era -3.0 (chiusura): alzato per CONTRASTARE l'entropy
                                          # collapse (§1.9.C). Nell'apertura la maniglia è di lato
                                          # (~0.25m in Y): con target basso la policy si cristallizza
                                          # sullo "stare ferma" prima di trovarla. Target più alto =
                                          # più stocasticità mantenuta = esplorazione più lunga.
    policy_net_arch: Tuple[int, int] = (512, 512)

    eval_freq      : int = 50_000
    n_eval_episodes: int = 20
    checkpoint_freq: int = 200_000

    # ── Curriculum (SOLO livello 1, fisso) ──────────────────────────────────────
    # 1.0 = posa variabile (posizione ±15 cm, yaw ±17°) + fisica randomizzata.
    fixed_curriculum_level: float = 1.0
    curriculum_reward_k   : float = 0.0   # nessuna modulazione extra: livello fisso

    # ── Obiettivo di APERTURA ────────────────────────────────────────────────────
    # cap dell'angolo di apertura (rad) rispetto a door_min, come nel task v1 di apertura.
    door_open_cap_rad : float = 0.400
    # frazione [0,1] del range effettivo [door_min, door_min+cap] da raggiungere.
    # curriculum 1: target alto e variabile attorno all'apertura piena.
    goal_frac_min     : float = 0.85
    goal_frac_max     : float = 1.00
    # tolleranza di "porta aperta al goal": |door_angle - goal_angle| <= open_tol_rad
    # §1.29 — ALLARGATA da 0.03 a 0.05 (intervento GEOMETRICO, non di reward). Razionale fisico:
    # il goal è vicino al cap, fuori equilibrio; la diagnosi su 20 episodi reali mostra che la
    # porta RAGGIUNGE sempre il goal (open_error minimo ≈ 0.000) ma poi DERIVA indietro di
    # 0.024–0.050 rad per effetto della molla, prima che la FSM consolidi HOLD_OPEN. Con tol=0.03
    # la coda deterministica (alta frizione/massa, pose scomode) si ferma appena SOTTO soglia →
    # eval ~75% mentre il rollout stocastico è 1.0. Coerenza con la chiusura: lì la finestra 0.03
    # coincide con l'EQUILIBRIO (door≈0, deriva nulla); qui la finestra deve essere larga almeno
    # quanto la deriva fisica reale. 0.05 rad (~2.9°) resta un "aperto al valore richiesto" stretto.
    open_tol_rad      : float = 0.05

    # ── FSM a soglie adattive (§3.1) — SPECULARE alla chiusura ───────────────────
    # Soglia di chiusura del gripper per confermare la presa (REACH→PULL), adattiva
    # alla frizione: friz. alta → presa stabile con meno chiusura → soglia più bassa.
    fsm_grip_thresh_base   : float = 0.75
    fsm_grip_thresh_k_fric : float = 0.10
    fsm_friction_min       : float = 0.24   # 0.8 * 0.30 (min scala frizione)
    fsm_friction_max       : float = 0.96   # 0.8 * 1.20 (max scala frizione)
    # Distanza di presa adattiva al raggio maniglia (REACH→PULL).
    fsm_grasp_dist_base    : float = 0.045
    fsm_grasp_dist_k_radius: float = 1.5
    fsm_grasp_dist_offset  : float = 0.005
    # Timer di HOLD_OPEN adattivo alla rigidità del latch.
    fsm_hold_base_steps    : int   = 30
    fsm_hold_k_stiff       : float = 1.0
    # RETREAT: step sostenuti (post-rilascio) prima di terminare (gate min_release).
    # §1.52 — 30→40: con lo sfilamento sbloccato (guardia qvel tolta dall'escape, env §1.52)
    # il braccio arretra ~0.10 m; 40 step post-rilascio (~1.3 s) danno il TEMPO di completare
    # e vedere l'allontanamento prima della chiusura pulita. exo_exit=60 resta la rete di
    # sicurezza (l'episodio chiude comunque), quindi nessun rischio di deadlock.
    fsm_retreat_target_steps: int  = 40
    fsm_retreat_settle_dist : float = 0.06
    w_retreat_settle        : float = 20.0
    # §1.32 — RITIRO ATTIVO speculare alla chiusura (parametri IDENTICI a close config):
    # target di ritiro lungo la NORMALE della porta + guida direzionale nel reward +
    # terminazione su latch neutro (raggiungibile ora che il latch-restore è rimosso).
    terminate_on_retreat_complete: bool = True
    fsm_retreat_dist  : float = 0.25   # §1.41 [m] MISURATO (era 0.13, copiato dalla chiusura).
                                        # L'arretramento UTILE prima del freeze di settle e'
                                        # (fsm_retreat_dist - fsm_retreat_settle_dist): con 0.13
                                        # erano solo 7 cm. Misura nell'env reale su 4 seed della
                                        # distanza necessaria a LIBERARE la leva incastrata fra
                                        # le dita aperte: 0.026 / 0.025 / 0.128 / 0.080 m -> il
                                        # fabbisogno VARIA con la randomizzazione (lunghezza e
                                        # raggio maniglia, posa di presa) fino a 12.8 cm. Con 7 cm
                                        # gli episodi facili si liberavano e quelli difficili si
                                        # INCASTRAVANO (latch fermo a ~0.8-0.9, asintotico).
                                        # 0.25 -> arretramento utile 0.19 m = +48% sul caso
                                        # peggiore misurato. Raggiungibilita' verificata: il
                                        # braccio arretra fino a 0.48 m.
                                        # La chiusura usa 0.13 perche' li' la porta e' a
                                        # fine-corsa e la geometria della leva e' diversa: la
                                        # soglia segue la FISICA del task, non la simmetria.
    fsm_retreat_z_off : float = 0.0    # §1.45 [m] — FUGA PURA LUNGO LA NORMALE. Tre run,
                                        # tre esperimenti, una sola geometria compatibile:
                                        #  +0.04 (=chiusura, run §1.42): il dito sale DENTRO
                                        #    l'arco di ritorno della leva → leva appoggiata sul
                                        #    dito (latch asintotico ~0.5), incastro a hard-cap.
                                        #  -0.06 (§1.44, diag): il dito PREME la barra contro il
                                        #    suo fine-corsa (ep.3/5: latch spinto a +1.5711=π/2
                                        #    esatto, braccio inchiodato 7 mm/70 step) oppure,
                                        #    quando si sfila, il pattino resta premuto sulla
                                        #    barra per tutta l'uscita e TRASCINA la porta
                                        #    (ep.1: door 0.400→0.217 con qvel -0.4 costante).
                                        # La barra ruota in un piano PARALLELO al pannello:
                                        # l'unica uscita che non la tocca mai e' perpendicolare
                                        # a quel piano = normale della porta, z=0. Nella
                                        # chiusura +0.04 resta corretto (leva gia' a riposo).
    # §1.43 — ESCAPE GUIDATO post-rilascio (env-level, vedi env_v2.step, ramo RETREAT).
    # Dopo il rilascio delle dita, il braccio è GUIDATO lungo la normale della porta finché
    # (a) si è allontanato di retreat_escape_dist dal punto di presa, oppure (b) la leva è
    # tornata neutra (|latch| ≤ retreat_latch_neutral_tol). Motivo (misurato, §1.41 + traccia
    # §1.42): per liberare la leva servono fino a 0.128 m ma la policy si fermava a ~0.10 m →
    # leva appoggiata sul dito, porta spinta dalla molla, terminazione solo a hard-cap.
    # 0.15 m = caso peggiore misurato (0.128) + margine. Deterministico: nessun retraining
    # necessario per validare (rilanciare diagnose_phase sulla policy esistente).
    retreat_escape_enabled : bool  = True
    retreat_escape_dist    : float = 0.15
    retreat_escape_gain    : float = 5.0   # guadagno direzione→azione (come la mano guidata)
    # §1.56 — RITIRO VERSO LA POSA DI PARTENZA (SOLO play/diagnostica, MAI training).
    # retreat_to_start_enabled = default che play/diagnostica passano a env.set_retreat_to_start().
    # In training il flag runtime dell'env resta False → target = normale (baseline 100%).
    retreat_to_start_enabled   : bool  = True    # attivo in play/diag; ininfluente in training
    retreat_to_start_clearance : float = 0.10    # [m] offset del target lungo la normale (anti-incastro)
    # §1.44 — USCITA ESOGENA GARANTITA (la ricetta §1.36/§1.37 capita fino in fondo, che
    # faceva il 100% VERO). Il gate §1.42 sul latch rendeva la durata del RETREAT
    # controllabile dalla policy (incastro = episodio infinito = farming del reddito
    # per-step). L'episodio ora termina COMUNQUE a retreat_exo_exit_steps dal rilascio:
    # la leva col z_off corretto torna in 3-8 step, quindi l'uscita normale resta
    # (min_release=30 AND latch a casa); questa e' la RETE DI SICUREZZA non estendibile
    # che azzera il valore del sabotaggio. Bonus solo a leva tornata (§1.43).
    retreat_exo_exit_steps : int   = 60
    # §1.46 — RIPORTO ATTIVO DELLA LEVA (env-level, deterministico). Diag §1.45: con
    # z_off=0 la porta resta al goal (5/5) ma lo sfilamento sotto CARICO è friction-limited
    # (molla a θ≈1.35 preme la barra sul dito; il braccio si arresta esponenzialmente a
    # ~3.7 cm; leva a casa 0/5). Fix: a inizio RETREAT, PRESA ANCORA CHIUSA, l'eef è
    # guidato lungo la tangente dell'arco della leva (v = -axis×r da MuJoCo xaxis/xanchor)
    # finché |latch| ≤ retreat_restore_tol (max retreat_restore_max_steps): la molla si
    # SCARICA prima dello stacco → rilascio ed escape senza carico. È il §1.22 fatto nel
    # modo giusto: ATTIVO (il §1.22 congelava il braccio e aspettava → la leva non può
    # tornare mentre è impugnata da un braccio fermo → deadlock → fu abbandonato).
    retreat_restore_enabled  : bool  = True
    retreat_restore_tol      : float = 0.35
    retreat_restore_max_steps: int   = 40   # §1.47: era 20 — rate di riporto MISURATO
                                             # ~0.034 rad/step: da fondo corsa (1.57) a
                                             # tol (0.35) servono ~36 step; con 20 il
                                             # riporto si fermava a meta' (leva a 0.75-0.9
                                             # ancora carica) e lo sfilamento restava
                                             # friction-limited negli episodi lenti.
    retreat_restore_gain     : float = 2.0
    # §1.48 — fallback MORSA→GABBIA del riporto. Con maniglie GROSSE (r≈0.025) la barra
    # non può ruotare dentro la pinza chiusa (coppia d'attrito ∝ raggio; polso bloccato):
    # il riporto non progredisce (ep.4 §1.47: 0.001 rad/step vs 0.034 degli altri). Se dopo
    # retreat_restore_cage_after step il latch è sceso meno di retreat_restore_cage_progress,
    # le dita si SEMIAPRONO a diam+margine: barra libera di ruotare nella gabbia, il dito
    # la accompagna lungo l'arco spingendola. Gli episodi che progrediscono in morsa
    # (4/5 nel diag §1.47) non cambiano comportamento.
    retreat_restore_cage_after   : int   = 12
    retreat_restore_cage_progress: float = 0.10
    retreat_restore_cage_margin  : float = 0.015
    # §1.49 — guadagno della rotazione del polso durante il riporto (moto rigido attorno
    # all'asse del latch: traslazione tangente + rotazione coerente; con le rotazioni
    # azzerate l'OSC teneva l'orientazione rigida e nelle pose scomode piantava il braccio).
    retreat_restore_rot_gain     : float = 0.5
    # §1.50 — la gabbia è la modalità PREDEFINITA del riporto (non più solo fallback):
    # in morsa le dita scivolano dalla barra e la pinza si chiude a pugno nel vuoto
    # (width 0.0012 misurata nel diag §1.49) trascinando la porta di ±0.13 rad.
    retreat_restore_cage_always  : bool  = True
    # §1.51 — ANTI-BOUNCE + rilascio più deciso (env-level, deterministico). Post-training
    # 4/5 PULITA con RIMBALZO ≤0.03, ma resta un avanti-indietro della porta all'ingresso
    # in RETREAT (spike door_qvel −0.54 misurato) e un rilascio un po' lento. Rimedi:
    #  • guardia door_qvel: se la porta si muove, il braccio MOLLA (scala l'azione fino a
    #    floor) invece di inseguire → spezza il ciclo di rimbalzo.
    #  • rampa del riporto: i primi step del riporto partono morbidi (niente strappo).
    retreat_door_qvel_ref   : float = 0.15   # |dθ/dt| [rad/s] oltre cui il braccio molla
    retreat_door_qvel_floor : float = 0.25   # scala minima del comando (non fermarsi mai del tutto)
    retreat_restore_ramp    : int   = 4      # step di avvio morbido del riporto (0→pieno)
    retreat_hard_cap  : int   = 120    # §1.46: 90→120 per far spazio ai ≤20 step di
                                        # riporto leva. §1.44: era 200 — con l'incastro sistematico (§1.42/
                                        # §1.43) il cap NON era piu' una guardia ma la DURATA
                                        # DI FATTO del RETREAT: 200 step di reddito per-step
                                        # (~+6/step con l'escape pagato) = jackpot ep_rew~500,
                                        # con la policy che imparava a SABOTARE il rilascio
                                        # per restarci. 90 = il valore che il commento §1.42
                                        # gia' indicava come "solo guardia anti-deadlock".
                                        # (§1.42): guardia anti-deadlock. La terminazione
                                        # ora aspetta la LEVA tornata (come la chiusura):
                                        # tempo di ritorno leva misurato 24-40 step.
                                        # In caso patologico il cap chiude comunque a porta aperta.
                                        # il cap 90 resta solo guardia anti-deadlock.

    # ── Reward potential-based (§3.2, Ng 1999) — SPECULARE ───────────────────────
    phi_reach_weight  : float = 25.0    # allineato alla chiusura v2 (bonus di grasp forte alla transizione REACH→PULL)
    phi_reach_sigma   : float = 0.40    # allineato alla chiusura v2 (§1.9.A)
    phi_pull_weight   : float = 5.0     # specchio di phi_push (apertura invece di chiusura)
    phi_hold_weight   : float = 5.0
    phi_retreat_weight: float = 5.0

    # termine di mantenimento contatto durante PULL (specchio di §1.16 grip_contact)
    w_grip_contact    : float = 0.5
    # ── Progresso DENSO di apertura nel PULL (mirror di door_prog della chiusura) ─
    # CRUCIALE: come per il REACH, il solo Φ_pull ha gradiente troppo debole; questo
    # termine ratchet è il segnale GENUINO che "apre la porta" verso il goal. Senza,
    # la policy afferra ma non tira fino al goal (success plateau ~15%). Rif. close §1.10.C.
    w_pull_progress   : float = 300.0    # come phi_push_weight*... della chiusura (door_prog)
    # §1.31 — saturazione del progresso al goal (specchio della chiusura, dove door_prog
    # spinge verso 0 = fine-corsa e satura). Default OFF: preserva la baseline al 100% (§1.30).
    # Attivare per l'A/B che mira a ridurre la deriva post-rilascio sui goal bassi (vedi §10.9).
    pull_progress_cap_at_goal : bool = False
    w_pull_dist_3d    : float = 5.0      # non perdere la maniglia mentre tira
    w_pull_dist_z     : float = 15.0
    w_pull_grip_weak  : float = 2.0      # presa sotto soglia durante il PULL (dolce)
    # ── Termini DENSI di REACH (mirror della chiusura v2 che funziona) ───────────
    # CRUCIALE: con shaping a potenziale cumulativo, Φ_reach NON dà gradiente in REACH;
    # questi termini densi sono l'UNICO segnale che porta il braccio alla maniglia.
    # Senza, la policy resta a mezza distanza (success_rate=0). Rif. close reward §1.10.B.
    w_reach_dist_3d  : float = 5.0
    w_reach_dist_xy  : float = 3.0
    w_reach_dist_z   : float = 15.0
    w_reach_app_blw  : float = 3.0     # penalità "sotto" la maniglia
    w_reach_app_top  : float = 1.5     # penalità "troppo sopra" la maniglia
    w_reach_grip_near: float = 2.5     # premio chiusura quando vicino alla maniglia
    fsm_grasp_dist_k_offset : float = 0.005   # usato per la soglia di "vicinanza" gripper

    # ── Override deterministici env-level (zero reward) — SPECULARE ──────────────
    # §1.17 — rilascio pulito nel RETREAT.
    retreat_clean_release : bool  = True
    retreat_clear_margin  : float = 0.012  # §1.51: era 0.02 — sfilamento avviato ~2-3 step
                                            # prima (dita libere = width > diam+margine): il
                                            # braccio si allontana più in fretta appena la
                                            # barra è geometricamente libera.
    # §1.22 — accompagnamento della LEVA/maniglia alla posizione di partenza PRIMA del
    # rilascio (env-level, ZERO reward). A porta aperta e ferma, durante la presa la leva
    # è tenuta ruotata; prima di staccarsi il braccio mantiene la presa e lascia che la
    # molla di richiamo riporti la leva a latch≈0 (specchio della chiusura, che NON termina
    # finché |latch_qpos| non è sotto soglia). È motion-quality a successo già raggiunto →
    # deterministico, nessun retraining, nessun rischio per il reward che funziona.
    # §1.22/§1.26 → §1.32: DISATTIVATO. L'accompagnamento leva teneva la PRESA CHIUSA fino a
    # retreat_latch_max_steps. Ma la leva non può tornare neutra MENTRE è impugnata → il ramo
    # scadeva sempre al cap (20 step) con braccio CONGELATO → restavano ~3 step di rilascio + 8
    # di rampa e la terminazione a contatore (30) scattava → il braccio non si ritirava MAI
    # (RETREAT=31 esatti in 20/20 episodi del diagnostico). La suite della CHIUSURA (T5) prova
    # che è l'approccio sbagliato: lì il latch è >0.15 al 100% delle transizioni HOLD→RETREAT,
    # si rilascia SUBITO e la molla neutralizza la leva DURANTE il ritiro (termina a latch<0.08).
    # Il deadlock storico di §1.25 (ep_len~580) era causato da QUESTO override (allora senza
    # cap), non dal gate sul latch.
    retreat_latch_restore     : bool  = False   # era True: causa-radice del mancato ritiro
    retreat_latch_neutral_tol : float = 0.05    # |latch_qpos| sotto cui la leva è "a posto"
    # §1.26 — CAP temporale dell'accompagnamento leva: la CHIUSURA può attendere la leva
    # perché lì il latch torna a 0 da solo; nell'APERTURA la leva può NON neutralizzarsi
    # (porta spalancata) e il braccio resterebbe aggrappato all'infinito (ep_len alto, il
    # robot non si ritira — vedi screenshot). Quindi: accompagna la leva al MASSIMO per
    # questi step; superati, procedi comunque a rilascio+ritiro. Allineato a retreat_target.
    retreat_latch_max_steps   : int   = 20
    # §1.18 — grip-lock in PULL/HOLD_OPEN (blocca aperture accidentali).
    grip_lock_enabled     : bool  = True
    grip_lock_margin      : float = 0.15   # §1.53 — CONTATTO PIÙ SOLIDO. Il grip-lock (env,
                                            # zero reward) si attiva SOLO quando le dita sono
                                            # GIÀ fisicamente attorno alla maniglia
                                            # (_prev_is_phys_closed): alzare il floor da
                                            # grip_thresh+0.10 a +0.15 fa premere le dita più
                                            # a fondo sulla barra che stanno già tenendo →
                                            # presa più salda in PULL/HOLD_OPEN. NON viola la
                                            # cautela §1.16 (over-close su maniglie sottili):
                                            # le dita non possono chiudersi OLTRE il diametro
                                            # della barra (la barra le blocca), quindi la
                                            # larghezza resta ≈diam > 0.015 = dentro la banda
                                            # di contatto. Direzionale (solo verso la chiusura),
                                            # auto-disattivante se la presa si perde. Reversibile
                                            # rimettendo 0.10.
    # §1.21 — rampa di avvio del ritiro (avvio morbido fermo→policy).
    retreat_rampup_enabled: bool  = True
    retreat_rampup_steps  : int   = 8

    # ── Ritorno/terminazione (specchio di return_* della chiusura) ───────────────
    return_pos_tol   : float = 0.05
    return_hold_steps: int   = 10
    w_return_pos     : float = 2.0
    w_door_regress   : float = 4.0       # in apertura: penalizza la RICHIUSURA post-successo

    success_bonus    : float = 5.0
    action_smooth_alpha: float = 0.8
    # §1.28 — STABILIZZAZIONE sul goal in HOLD_OPEN/RETREAT (copia FEDELE del blocco HOLD
    # della chiusura v2 che fa 100% true success). Nella chiusura la porta arriva a door≈0 e
    # ci RESTA ferma (door_end ±0.004) grazie a questi termini; nell'apertura mancavano del
    # tutto (solo hold=1.0 piatto) → la porta arrivava al goal ma rimbalzava a ridosso di
    # goal-tol (eval deterministico ~75% vs rollout ~100%, "bounce della porta"). Stessi pesi.
    w_hold_bounce    : float = 20.0      # §1.29 DEPRECATO: combatte la molla fuori equilibrio (vedi reward HOLD_OPEN)
    w_hold_veldamp   : float = 10.0      # §1.29 DEPRECATO: idem — non più applicato
    w_hold_slip      : float = 5.0       # presa fisica persa in HOLD (chiusura: -5)
    w_hold_drop_pen  : float = 10.0      # gripper che si apre in HOLD (chiusura: -10)
    w_hold_dist      : float = 3.0       # maniglia che si allontana in HOLD (chiusura: -3)
    # §1.25 — LATCH MONITOR nel RETREAT (mirror di latch_ret della chiusura). Insegna ad
    # accompagnare la maniglia alla posizione di partenza PRIMA di staccarsi. Attivo SOLO
    # in RETREAT → non interferisce col task (apertura). Peso moderato come la chiusura (1.0).
    w_latch_ret          : float = 1.0
    retreat_latch_term_tol: float = 0.15   # §1.51: era 0.08. Post-training ep.4 (maniglia a
                                            # bassa rigidità) la molla, appena rilasciata,
                                            # SUPERA lo zero e oscilla, chiudendo a latch
                                            # −0.127 → ESOGENA per un soffio. La misura MuJoCo
                                            # (nota sotto) dà residuo di equilibrio 0.05–0.20
                                            # sul range di stiffness randomizzato: 0.15 è
                                            # DENTRO la banda fisica → la leva è "a casa" anche
                                            # con l'overshoot, l'episodio chiude PULITO. Le
                                            # PULITE già a 0.05–0.08 non cambiano. NON è un
                                            # allargamento arbitrario: è la fisica del giunto.
                                            # §1.33: gate storicamente non di terminazione (misura MuJoCo:
                                            # residuo leva ≈0.1/stiffness ∈ [0.05,0.20] a porta
                                            # aperta → irraggiungibile per stiffness ≲1.0).
                                            # Tenuto per diagnostica/retrocompatibilità.

    # ── Domain randomization estesa (§3.4) — sempre attiva ───────────────────────
    rand_latch_stiffness    : bool  = True
    rand_latch_stiffness_min: float = 0.5
    rand_latch_stiffness_max: float = 2.0
    rand_hinge_damping      : bool  = True
    rand_hinge_damping_min  : float = 0.3
    rand_hinge_damping_max  : float = 1.5
    rand_door_mass          : bool  = True
    rand_door_mass_min      : float = 0.5
    rand_door_mass_max      : float = 2.0

    # ── Grasp multi-approach (§3.3) ──────────────────────────────────────────────
    grasp_n_candidates: int = 3

    # ── beta_net (§3.5) — disabilitato di default (capitolo futuro) ──────────────
    use_beta_net   : bool  = False
    beta_net_hidden: int   = 64
    beta_net_lr    : float = 1e-4
    beta_net_reg   : float = 1e-4

    debug_print_every: int = 200