---

### 7.1 — Architettura dei Moduli e Responsabilità

Il modulo è composto da 8 file. Ogni file ha **una sola responsabilità**:
nessuna logica è duplicata tra moduli diversi.

```
close_generalized_v2/
│
├── config_v2.py          ← TrainConfigV2
│                            Unica fonte di verità per tutti i parametri.
│                            Non ha logica: è un dataclass di soli valori.
│                            Viene passato come argomento (cfg) a ogni modulo.
│                            Modificare qui per cambiare qualsiasi iperparametro.
│
├── fsm_v2.py             ← AdaptiveFSM + FSMState
│                            Decide QUANDO transitare tra fasi.
│                            Calcola le soglie adattive (§3.1).
│                            NON calcola il reward e NON tocca MuJoCo.
│
├── reward_v2.py          ← PotentialBasedReward
│                            Calcola QUANTO reward assegnare a ogni step.
│                            Legge la fase corrente da FSMState.
│                            NON decide le transizioni e NON tocca MuJoCo.
│
├── grasp_strategy.py     ← MultiApproachGrasp
│                            Calcola le K=3 direzioni di approccio (§3.3).
│                            Calcola il valore di alignment per ciascuna.
│                            Fornisce le features per l'osservazione.
│
├── domain_rand_v2.py     ← ExtendedDomainRandomizer
│                            Modifica il modello MuJoCo in-place a ogni reset.
│                            È l'unico modulo che scrive su sim.model.
│                            Espone i valori randomizzati come attributi correnti.
│
├── beta_net.py           ← BetaNetwork (disabilitata di default)
│                            Gate probabilistico AGGIUNTIVO per le transizioni FSM.
│                            Contiene tre MLP indipendenti (una per fase).
│                            Restituisce sempre {1.0, 1.0, 1.0} se disabilitata.
│
├── env_v2.py             ← AdvancedGeneralizedDoorEnv
│                            Orchestratore: assembla e chiama tutti gli altri moduli.
│                            Eredita da RoboSuiteDoorCloseGymnasiumEnv (v1 base).
│                            È il solo punto di contatto con SB3 e Robosuite.
│
└── train_gen_v2.py       ← main()
                             Crea env, model SAC, callbacks e lancia il training.
                             Contiene i 4 callback: SuccessRate, GraspDiagnostic,
                             AdaptiveCurriculum, CustomEval.
```

**Dipendenze tra file (chi importa chi):**
```
train_gen_v2.py
    └── importa: config_v2, env_v2

env_v2.py
    └── importa: config_v2, fsm_v2, reward_v2,
                 grasp_strategy, domain_rand_v2, beta_net

reward_v2.py
    └── importa: fsm_v2 (solo FSMState, PHASE_*)

Tutti gli altri
    └── importano solo: config_v2 (TrainConfigV2)
```

Questa struttura fa sì che ogni modulo possa essere testato in isolamento:
per testare `reward_v2.py`, non serve MuJoCo — basta creare un `FSMState` fittizio.

---

### 7.2 — Flusso di un Episodio (reset)

Il reset è il momento in cui la fisica dell'episodio viene stabilita.
Tutti i valori randomizzati per quell'episodio vengono fissati qui
e rimangono costanti per tutti gli step successivi.

**Confronto v1 vs v2 nel reset:**

| Operazione | v1 | v2 |
|-----------|----|---------|
| Randomizzazione handle | ✅ | ✅ (esteso) |
| Randomizzazione posizione/yaw porta | ✅ | ✅ (identico) |
| Randomizzazione stiffness latch | ❌ | ✅ |
| Randomizzazione damping cerniera | ❌ | ✅ |
| Randomizzazione massa porta | ❌ | ✅ |
| Calcolo timer HOLD adattivo | ❌ (fisso 60) | ✅ (calcolato da stiffness) |
| Reset Φ precedente (reward) | ❌ | ✅ (`_prev_phi = None`) |
| Reset FSM state | ✅ | ✅ (con calcolo hold_steps) |

**Flusso dettagliato del reset in v2:**

```
env.reset(seed, options)
    │
    ├─► [1] domain_rand.randomize_episode(curriculum_level)
    │          ├─► _randomize_handle()
    │          │       sim.model.geom_size[handle_geom][0]    = base_r × U(0.7, 1.4)
    │          │       sim.model.geom_friction[handle_geom]   = base_f × U(0.3, 1.2)
    │          │       self.current_handle_radius   = nuovo valore
    │          │       self.current_handle_friction = nuovo valore
    │          │
    │          ├─► _randomize_latch_stiffness()
    │          │       sim.model.jnt_stiffness[latch_joint]   = base_s × U(0.5, 2.0)
    │          │       self.current_latch_stiffness = nuovo valore
    │          │
    │          ├─► _randomize_hinge_damping()
    │          │       sim.model.dof_damping[hinge_dof]       = base_d × U(0.3, 1.5)
    │          │       self.current_hinge_damping = nuovo valore
    │          │
    │          └─► _randomize_door_mass()
    │                  sim.model.body_mass[door_body]         = base_m × U(0.5, 2.0)
    │                  self.current_door_mass = nuovo valore
    │
    ├─► [2] fsm.reset(latch_stiffness=domain_rand.current_latch_stiffness,
    │                  base_latch_stiffness=domain_rand.base_latch_stiffness,
    │                  control_freq=cfg.control_freq)
    │          ├─► FSMState.reset()
    │          │       phase = PHASE_REACH
    │          │       grasp_confirm_count = 0
    │          │       hold_closed_duration = 0
    │          │       retreat_pos = None
    │          │       target_hold_steps = None  ← temporaneamente
    │          │       ... (tutti i campi a 0/None)
    │          │
    │          └─► compute_target_hold_steps()
    │                  base_steps   = 30 × 2.0 = 60
    │                  stiff_norm   = current / base
    │                  extra_steps  = k × max(0, 1 − stiff_norm) × 30
    │                  target_hold_steps = base + extra  ← salvato in FSMState
    │
    ├─► [3] reward_fn.reset()
    │          self._prev_phi = None  ← il potenziale precedente non esiste ancora
    │
    ├─► [4] Randomizzazione posizione/yaw porta (stessa di v1)
    │          pos_offset = U(-p_var, p_var, size=3); pos_offset[2] = 0
    │          yaw        = U(-r_var, r_var)
    │          sim.model.body_pos[door_body_id]  = base_pos + pos_offset
    │          sim.model.body_quat[door_body_id] = q_new  (composizione quaternioni)
    │
    ├─► [5] super().reset() → Robosuite resetta il simulatore fisico
    │
    └─► [6] _flatten_obs(obs)  ← costruisce l'osservazione iniziale
                ├─► base_flat  = super()._flatten_obs(obs)       (~32 dim)
                │       joint_pos (7), joint_vel (7), eef_pos (3),
                │       eef_quat (4), gripper_qpos (2), gripper_vel (2),
                │       door/handle raw obs da Robosuite (~7)
                │
                ├─► custom     = [dist, radius, friction,          (8 dim)
                │                 fsm_onehot[4], hinge_qpos]
                │       dist      = ||eef_pos − handle_pos||
                │       radius    = domain_rand.current_handle_radius
                │       friction  = domain_rand.current_handle_friction
                │       fsm_onehot = fsm.state.one_hot  → [1,0,0,0] (REACH)
                │       hinge_qpos = sim.data.qpos[hinge_qpos_addr]
                │
                ├─► grasp_feats = grasp_strategy.obs_features(     (4 dim)
                │       eef_quat, handle_pos, eef_pos, door_quat)
                │       → [best_align, align_top, align_latL, align_latR]
                │
                └─► physics_feats = domain_rand.obs_features()     (3 dim)
                        → [norm_stiffness, norm_damping, norm_mass]

        obs_v2 = concat([base_flat, custom, grasp_feats, physics_feats])
        → totale ~47 dim
```

---

### 7.3 — Flusso di uno Step (con confronto v1 vs v2)

**Confronto pipeline step v1 vs v2:**

| Passo | v1 | v2 | Differenza |
|------|----|----|-----------|
| [1] Smoothing azione | ✅ alpha=0.8 | ✅ alpha=0.8 | Identico |
| [2] FSM override azione | ✅ solo HOLD freeze | ✅ HOLD + RETREAT freeze | v2 aggiunge freeze in RETREAT |
| [3] Step MuJoCo | ✅ | ✅ | Identico |
| [4] Variabili fisiche | ✅ door_angle, dist | ✅ + door_qvel, door_speed | v2 aggiunge velocità porta |
| [5] Beta-net | ❌ assente | ✅ (disabilitata di default) | Nuova in v2 |
| [6] FSM update | ✅ soglie costanti | ✅ **soglie adattive** | v2 calcola thresh da physics |
| [7] Grasp alignment | ❌ 1 direzione fissa | ✅ **K=3 direzioni** | Nuovo modulo `grasp_strategy` |
| [8] Reward | ✅ ~15 termini fissi | ✅ **potential-based** | Riscrittura completa |
| [9] Osservazione | ~40 dim | **~47 dim** (+7 dim) | +grasp_feats, +physics_feats |
| [10] Info dict | success, angle | success, angle, **phase, hold_dur, align** | v2 più ricco per debugging |

---

**Flusso dettagliato v2:**

```
SAC.predict(obs) → action  (7 dim EEF + 1 dim gripper)
    │
    ▼
env.step(action)
    │
    ├─► [1] Action clipping e smoothing (alpha = 0.8)
    │       action = clip(action, -1, 1)
    │       action = 0.8 * action + 0.2 * prev_action   ← riduce jerk
    │
    ├─► [2] FSM action overrides
    │       ├─ PHASE_HOLD    → action[:-1] = 0.0  (braccio congelato, solo gripper attivo)
    │       └─ PHASE_RETREAT → action zeroed se dist_to_retreat_pos < tol
    │                          [v1: retreat_pos fisso in -X;  v2: retreat_pos perp. porta]
    │
    ├─► [3] super().step(action) → obs, rs_done
    │       (MuJoCo simula un passo fisico, Robosuite restituisce obs raw)
    │
    ├─► [4] Calcolo variabili fisiche
    │       ├─► door_angle  = _get_door_angle()
    │       ├─► door_qpos   = sim.data.qpos[hinge_qpos_addr]
    │       ├─► door_qvel   = sim.data.qvel[door_hinge_dof_adr]  ← NUOVO in v2
    │       ├─► door_speed  = |prev_angle - door_angle| × control_freq  ← NUOVO in v2
    │       └─► dist_handle = ||eef_pos − handle_pos||
    │
    ├─► [5] beta_net.predict(...)   ← SOLO se use_beta_net = True (default: False)
    │       └─► ritorna {beta_reach, beta_push, beta_hold}
    │           (tutti 1.0 se disabilitata → nessun effetto sul comportamento)
    │           [NUOVO in v2 — assente in v1]
    │
    ├─► [6] fsm.update(...)        ← OGNI STEP
    │       │
    │       │   v1: soglie costanti scritte nel codice
    │       │   v2: soglie calcolate dinamicamente a ogni step
    │       │
    │       │   Calcola soglie adattive [NUOVO in v2]:
    │       │   ├─► d_thresh = grasp_dist_thresh(handle_radius)  = 1.5r + 0.005
    │       │   ├─► g_thresh = grip_thresh(handle_friction)      = 0.75 - 0.10×norm_f
    │       │   └─► t_hold   = target_hold_steps (già calcolato al reset dalla stiffness)
    │       │
    │       │   Esegue logica di transizione:
    │       │
    │       │   PHASE_REACH:
    │       │   ├─► grasp_cond = (grip > g_thresh) AND is_phys_closed AND (dist < d_thresh)
    │       │   │   [v1: g_thresh=0.65 fisso;  v2: g_thresh adattivo alla frizione]
    │       │   │   [v1: d_thresh=0.020 fisso; v2: d_thresh adattivo al raggio]
    │       │   ├─► [se beta-net attiva] AND (beta_reach > 0.5)
    │       │   ├─► grasp_confirm_count += 1
    │       │   └─► se count >= 5: → PHASE_PUSH (just_grasped = True)
    │       │
    │       │   PHASE_PUSH:
    │       │   ├─► Aggiorna min_door_angle (traccia il minimo raggiunto)
    │       │   ├─► push_to_hold = (door_angle <= success_angle) AND (grip > 0.80)
    │       │   ├─► se push_to_hold: → PHASE_HOLD (just_succeeded = True)
    │       │   └─► se grasp perso: → PHASE_REACH (grasp_lost = True)
    │       │
    │       │   PHASE_HOLD:
    │       │   ├─► is_closed = |door_qpos| < 0.03
    │       │   ├─► hold_closed_duration += 1 (se chiusa)
    │       │   ├─► soft reset del timer se la porta rimbalza
    │       │   └─► se duration >= target_hold_steps: → PHASE_RETREAT
    │       │       [v1: target_hold_steps=60 fisso]
    │       │       [v2: target_hold_steps adattivo alla stiffness del latch]
    │       │           └─► compute_retreat_pos(eef_pos, door_quat, dist, z)
    │       │               [v1: retreat_pos = eef + [-0.13, 0, 0.04] (frame globale)]
    │       │               [v2: retreat_pos = eef + dist × door_normal (frame porta)]
    │       │
    │       └─► PHASE_RETREAT:
    │           └─► transizione gestita dall'environment (return_hold counter)
    │
    ├─► [7] grasp_strategy.compute_alignment(eef_quat, handle_pos, door_quat)
    │       [NUOVO in v2 — in v1 era un singolo dot product fisso]
    │       ├─► get_candidate_directions(K=3: top, lat-L, lat-R)
    │       │       [v2: dir_lat_L/R dipendono dall'orientazione porta via quaternione]
    │       ├─► aligns = [|dot(eef_z, d)| for d in candidates]
    │       └─► best_align = max(aligns), best_idx = argmax
    │
    ├─► [8] reward_fn.compute(...)   ← OGNI STEP
    │       │
    │       │   Jerk penalty (identico v1/v2):
    │       │   smoothness = −w_smooth × ||action[:-1] − prev_action[:-1]||
    │       │
    │       │   Potential-based shaping [NUOVO in v2]:
    │       │   [v1: reward = somma di termini ad hoc con pesi fissi]
    │       │   [v2: F = γΦ(s') − Φ(s), garantisce policy ottimale (Ng 1999)]
    │       │   ├─► REACH: phi_now = phi_reach(dist, radius, curriculum_lvl)
    │       │   │           sigma   = max(radius×3, 0.08)  [adattiva alla geometria]
    │       │   │           phi     = w × exp(-dist/sigma)
    │       │   ├─► PUSH:  phi_now = phi_push(door_angle, door_max, grip, thresh, lvl)
    │       │   │           [v1: 2000×Δangle, oscillabile]
    │       │   │           [v2: closure × sigmoid(10×(grip-thresh)), monotono]
    │       │   ├─► HOLD:  phi_now = phi_hold(duration, target, door_qpos)
    │       │   └─► RETREAT: phi_now = phi_retreat(dist_to_target)
    │       │   shaping = gamma × phi_now − phi_prev  → aggiunge a reward
    │       │
    │       └─► clip reward a [−100, +100]
    │
    ├─► [9] _flatten_obs(obs) → obs v2 (~47 dim)
    │       [v1: ~40 dim; v2: +4 grasp_feats +3 physics_feats]
    │
    ├─► [10] Aggiornamento info dict
    │        v1: {is_success, door_angle}
    │        v2: {is_success, door_angle, fsm_phase, fsm_phase_name,
    │             hold_duration, target_hold_steps, curriculum_level,
    │             best_grasp_align, best_grasp_dir_idx, door_qpos, latch_qpos}
    │
    └─► return (obs, reward, terminated, truncated, info)
```

---

### 7.4 — Statechart FSM: Confronto v1 vs v2

**Differenze strutturali v1 vs v2:**

| Aspetto FSM | v1 | v2 |
|------------|----|---------|
| Soglia distanza REACH→PUSH | `0.020 m` costante | `1.5 × radius + 0.005` adattiva |
| Soglia grip REACH→PUSH | `0.65` costante | `0.75 − 0.10 × norm_friction` adattiva |
| Conferma grasp | 5 step consecutivi | 5 step consecutivi (identico) |
| Check fisico contatto | ❌ solo geometrico | ✅ `is_phys_closed` (gripper_width check) |
| Gate beta_reach | ❌ assente | ✅ `beta_reach > 0.5` (disabilitata) |
| Soglia grip PUSH→HOLD | `0.80` costante | `0.80` costante (identico) |
| Gate beta_push | ❌ assente | ✅ `beta_push > 0.5` (disabilitata) |
| Timer HOLD | `60 step` fisso | `60 + extra(stiffness)` adattivo |
| Soft reset timer HOLD | ✅ | ✅ (identico) |
| Gate beta_hold | ❌ assente | ✅ `beta_hold > 0.5` (disabilitata) |
| Target retreat | `eef + [-0.13, 0, 0.04]` (globale) | `eef + dist × door_normal` (locale porta) |
| Archi di ritorno | Solo PUSH→REACH | Solo PUSH→REACH (identico) |

**Statechart v2 (le novità rispetto a v1 sono evidenziate):**

```
                   ┌───────────────────────────────────────────────────────┐
                   │                                                       │
                   ▼                                                       │
         ┌──────────────────┐                                             │
  start  │                  │  dist < d_thresh(r)  ← [v2] adattiva       │
 ──────► │   PHASE_REACH    │  AND grip > g_thresh(f) ← [v2] adattiva   │
         │                  │  AND is_phys_closed   ← [v2] check fisico  │
         │  reach_steps++   │  [AND beta_reach > 0.5 ← [v2] gate β]     ├──► PHASE_PUSH
         │                  │  per 5 step consecutivi                    │
         └──────────────────┘                                             │
                                                              ┌───────────┴──────────┐
                                     grasp_lost               │                      │
                        ◄────────────────────────────────     │    PHASE_PUSH        │
                                                              │                      │
                                                              │  push_steps++        │
                                                              │  min_door_angle ↓    │
                                                              └──────────────────────┘
                                                                          │
                                                            door_angle <= success_angle
                                                            AND grip > 0.80
                                                            [AND beta_push > 0.5  ← [v2] gate β]
                                                                          │
                                                                          ▼
                                                              ┌──────────────────────┐
                                                              │                      │
                                                              │    PHASE_HOLD        │
                                                              │                      │
                                                              │  hold_duration++     │
                                                              │  (se |door_qpos|<.03)│
                                                              │  soft reset se rimb. │
                                                              └──────────────────────┘
                                                                          │
                                                          hold_duration >= target_hold_steps
                                                                       ↑
                                                          [v1] 60 step fissi
                                                          [v2] 60+extra(stiffness) step
                                                          [AND beta_hold > 0.5  ← [v2] gate β]
                                                                          │
                                                                          ▼
                                                              ┌──────────────────────┐
                                                              │                      │
                                                              │   PHASE_RETREAT      │
                                                              │                      │
                                                              │  retreat_pos =       │
                                                              │  [v1] eef+[-0.13,0,0.04] │
                                                              │  [v2] eef+dist×normal+z  │
                                                              │                      │
                                                              │  return_hold++       │
                                                              │  (se vicino target)  │
                                                              └──────────────────────┘
                                                                          │
                                                          return_hold >= return_hold_steps
                                                          OR dist_to_target < tol
                                                                          │
                                                                          ▼
                                                                      SUCCESS ✓
```

---

### 7.5 — Come la Rete Neurale (SAC) Vede il Sistema (con confronto v1 vs v2)

La rete neurale SAC è un MLP con due layer da 512 neuroni.
Riceve in input l'osservazione e produce un'azione.
**Non sa niente della FSM** — non può leggerla direttamente.

Ma la FSM comunica con la rete in modo **indiretto**:

```
FSM state → FSMState.one_hot → incluso in obs → visto dalla rete
```

La one-hot encoding `[1,0,0,0]`, `[0,1,0,0]`, ecc. dice alla rete in quale fase si trova.
La rete impara che in REACH deve avvicinarsi, in PUSH deve spingere, ecc.
Questo meccanismo è **uguale in v1 e v2**.

**Confronto osservazione v1 vs v2:**

| Gruppo | Dimensioni v1 | Dimensioni v2 | Cosa cambia |
|--------|-------------|-------------|-------------|
| `base_flat` | ~32 dim | ~32 dim | Identico (joint, EEF, gripper, door raw) |
| `dist_handle` | ✅ 1 | ✅ 1 | Identico |
| `handle_radius` | ✅ 1 | ✅ 1 | Identico |
| `handle_friction` | ✅ 1 | ✅ 1 | Identico |
| `fsm_onehot` | ✅ 4 (reach/push/hold/ret) | ✅ 4 | Identico |
| `hinge_qpos` | ✅ 1 | ✅ 1 | Identico |
| `grasp_feats` | ❌ 0 | **✅ 4** | **NUOVO** — multi-approach alignment |
| `physics_feats` | ❌ 0 | **✅ 3** | **NUOVO** — stiffness, damping, massa |
| **Totale** | **~40 dim** | **~47 dim** | **+7 dim** |

**Perché i 7 nuovi valori sono necessari:**

- `grasp_feats` (+4): senza questi, la rete non sa se l'approccio top-down è
  disponibile o se deve usarne uno laterale. Vedrebbe lo stesso `best_alignment`
  indipendentemente da quale delle K direzioni sia la migliore.
- `physics_feats` (+3): senza questi, la rete non sa distinguere un episodio
  con porta pesante da uno con porta leggera. Vedrebbe le stesse forze e accelerazioni
  ma non potrebbe attributirle alla massa — non potrebbe adattare la forza di push.

**Visualizzazione dell'osservazione v2:**

```
Obs ~47 dim:
┌─────────────────────────────────────────────────────────────┐
│ base_flat (~32 dim) — IDENTICO A v1                         │
│   joint_pos (7), joint_vel (7)                              │
│   eef_pos (3), eef_quat (4)                                 │
│   gripper_qpos (2), gripper_vel (2)                         │
│   door/handle raw obs da Robosuite (~7)                     │
├─────────────────────────────────────────────────────────────┤
│ custom (8 dim) — IDENTICO A v1                              │
│   dist_handle (1), handle_radius (1), handle_friction (1)   │
│   fsm_onehot: [reach, push, hold, retreat] (4)              │
│   hinge_qpos (1)                                            │
├─────────────────────────────────────────────────────────────┤
│ grasp_feats (4 dim) — NUOVO IN v2  [§3.3]                  │
│   best_alignment (1):  max degli allineamenti sui K dir.    │
│   align_top    (1):  |dot(eef_z, dir_to_handle)|            │
│   align_latL   (1):  |dot(eef_z, -door_Y_world)|           │
│   align_latR   (1):  |dot(eef_z, +door_Y_world)|           │
│   → permette alla rete di scegliere l'approccio migliore   │
├─────────────────────────────────────────────────────────────┤
│ physics_feats (3 dim) — NUOVO IN v2  [§3.4]                │
│   norm_stiffness (1): (curr_stiff - min) / (max - min)     │
│   norm_damping   (1): (curr_damp  - min) / (max - min)     │
│   norm_mass      (1): (curr_mass  - min) / (max - min)     │
│   → permette alla rete di adattarsi alla fisica corrente   │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
   MLP 47→512→512 (Actor v2)  [v1: MLP 40→512→512]
         │
         ▼
   Action (8 dim) — IDENTICO A v1
   [Δx, Δy, Δz, Δrx, Δry, Δrz, Δwrist, gripper]
```

**Update della rete neurale** (identico in v1 e v2 — gestito da SB3):
La rete viene aggiornata dal loop SAC in `train_gen_v2.py`,
completamente esternamente all'ambiente. L'ambiente non tocca i parametri della rete.

```
env.step() → (obs, reward, done, info)
    │
    ▼
SB3 SAC buffer.add(obs, action, reward, next_obs, done)
    │
    ogni train_freq step:
    ▼
SAC.train()
    ├─► Sample batch dal replay buffer
    ├─► Calcola target Q-values (con Critic target network)
    ├─► Update Critic: minimizza MSE(Q, target)
    ├─► Update Actor: massimizza Q(s, π(s))
    └─► Update temperatura entropia α (se ent_coef = "auto")
```

---

### 7.6 — Flusso del Training Loop Completo

```
main() in train_gen_v2.py
    │
    ├─► Crea cfg = TrainConfigV2(...)
    ├─► Crea 8 envs in parallelo (DummyVecEnv)
    ├─► Wrappa con VecNormalize (normalizza obs e reward online)
    ├─► Crea SAC model (MlpPolicy, net_arch=[512, 512])
    │
    ├─► Callbacks attivi durante model.learn():
    │   ├─► SuccessRateCallback    → logga success rate ogni 10k step
    │   ├─► GraspDiagnosticV2     → logga grasp_rate, retreat_rate
    │   ├─► AdaptiveCurriculumV2  → avanza curriculum se sr>0.85 e gr>0.50
    │   └─► CustomEvalCallbackV2  → valuta su 20 episodi ogni 10k step
    │           └─► salva best_model.zip se nuovo best success rate
    │           └─► recupera best_model se degradazione > 25% per 2 eval
    │
    └─► model.learn(total_steps=1_500_000)
            │
            loop per ogni step:
            ├─► SAC.predict(obs) → action
            ├─► env.step(action) → (obs, rew, done, info)
            ├─► buffer.add(...)
            └─► ogni 1 step: SAC.train() (gradient_steps=2)
```

---

### 7.7 — Descrizione Dettagliata di ogni Oggetto

Questa sezione elenca ogni classe/oggetto del modulo v2, i suoi attributi principali,
e i metodi pubblici con firma e scopo.

---

#### `TrainConfigV2` — `config_v2.py`

Dataclass Python che contiene **tutti** i parametri di configurazione.
Non ha logica: è solo un contenitore di valori.
Viene istanziata una volta in `main()` e passata a ogni modulo come `cfg`.

| Attributo | Tipo | Default | Usato da |
|-----------|------|---------|---------|
| `seed` | int | 42 | `main()` |
| `run_dir` | str | `"runs/close_gen_v2"` | `main()`, eval callback |
| `num_envs` | int | 8 | `main()` — n° di env paralleli |
| `horizon` | int | 500 | env (max step per episodio) |
| `control_freq` | int | 30 | FSM (Hz), env |
| `gamma` | float | 0.95 | SAC, reward shaping |
| `learning_rate` | float | 3e-4 | SAC optimizer |
| `buffer_size` | int | 1_000_000 | SAC replay buffer |
| `batch_size` | int | 256 | SAC gradient step |
| `policy_net_arch` | tuple | (512, 512) | SAC MLP architecture |
| `use_beta_net` | bool | False | env → beta_net |
| `use_potential_reward` | bool | True | reward_v2 |
| `fsm_grasp_dist_k_radius` | float | 1.5 | FSM: `grasp_dist_thresh()` |
| `fsm_grasp_dist_k_offset` | float | 0.005 | FSM: `grasp_dist_thresh()` |
| `fsm_grip_thresh_base` | float | 0.75 | FSM: `grip_thresh()` |
| `fsm_grip_thresh_k_fric` | float | 0.10 | FSM: `grip_thresh()` |
| `fsm_hold_time_base` | float | 2.0 | FSM: `compute_target_hold_steps()` |
| `fsm_hold_k_stiffness` | float | 0.5 | FSM: `compute_target_hold_steps()` |
| `fsm_retreat_dist` | float | 0.13 | FSM: `compute_retreat_pos()` |
| `fsm_retreat_z_off` | float | 0.04 | FSM: `compute_retreat_pos()` |
| `phi_reach_weight` | float | 2.0 | reward: `phi_reach()` |
| `phi_push_weight` | float | 5.0 | reward: `phi_push()` |
| `phi_hold_weight` | float | 3.0 | reward: `phi_hold()` |
| `phi_retreat_weight` | float | 1.5 | reward: `phi_retreat()` |
| `curriculum_reward_k` | float | 0.5 | reward (pesi co-evoluti) |
| `curriculum_check_freq` | int | 10_000 | AdaptiveCurriculumV2 |
| `curriculum_advance_delta` | float | 0.05 | AdaptiveCurriculumV2 |
| `rand_latch_stiffness_min/max` | float | 0.5 / 2.0 | domain_rand |
| `rand_hinge_damping_min/max` | float | 0.3 / 1.5 | domain_rand |
| `rand_door_mass_min/max` | float | 0.5 / 2.0 | domain_rand |
| `grasp_n_candidates` | int | 3 | grasp_strategy |

---

#### `FSMState` — `fsm_v2.py` (dataclass)

Oggetto mutabile che rappresenta lo **stato interno corrente** della FSM.
Viene resettato a ogni inizio episodio da `FSMState.reset()`.
È esposto pubblicamente tramite `self._fsm.state` nell'env.

| Attributo | Tipo | Significato |
|-----------|------|------------|
| `phase` | int | Fase corrente: 0=REACH, 1=PUSH, 2=HOLD, 3=RETREAT |
| `grasp_confirm_count` | int | Step consecutivi in cui la condizione di grasp è vera |
| `hold_closed_duration` | int | Step in cui la porta è rimasta chiusa in HOLD |
| `return_hold` | int | Step in RETREAT in cui il robot è vicino al target |
| `min_door_angle` | float\|None | Minimo angolo porta raggiunto durante PUSH |
| `retreat_pos` | ndarray\|None | Posizione 3D target di retreat (calcolata al passaggio in RETREAT) |
| `has_grasp_bonus` | bool | Flag: il bonus REACH→PUSH è già stato dato questo episodio |
| `events` | list[str] | Log testuale degli eventi FSM (per diagnostica) |
| `reach_steps` | int | Contatore step in REACH |
| `push_steps` | int | Contatore step in PUSH |
| `hold_steps_total` | int | Contatore step in HOLD |
| `retreat_steps` | int | Contatore step in RETREAT |
| `target_hold_steps` | int\|None | Durata HOLD adattiva calcolata al reset |

**Metodi:**
- `reset() → None` — azzera tutti i campi ai valori iniziali
- `one_hot → np.ndarray[4]` — property: vettore binario `[reach, push, hold, retreat]`
- `phase_name → str` — property: nome stringa della fase corrente

---

#### `AdaptiveFSM` — `fsm_v2.py`

Gestisce la **logica delle transizioni**. Possiede un'istanza di `FSMState`.

**Costruttore:**
```python
AdaptiveFSM(cfg: TrainConfigV2)
    self.cfg   = cfg        # parametri soglie adattive
    self.state = FSMState() # stato mutabile
```

**Metodi pubblici:**

| Metodo | Input | Output | Descrizione |
|--------|-------|--------|-------------|
| `grasp_dist_thresh(handle_radius)` | `float` | `float` | `= 1.5 × radius + 0.005` |
| `grip_thresh(handle_friction)` | `float` | `float` | `= 0.75 − 0.10 × norm_friction` |
| `compute_target_hold_steps(control_freq, latch_stiffness, base_latch_stiffness)` | `int, float, float` | `int` | Timer HOLD adattivo in step |
| `compute_retreat_pos(eef_pos, door_quat_mujoco, retreat_dist, retreat_z)` | `ndarray[3], ndarray[4], float, float` | `ndarray[3]` | Posizione target perpendicolare alla porta |
| `update(**kwargs)` | *(vedi sotto)* | `dict` | Esegue un passo FSM, restituisce eventi |
| `reset(latch_stiffness, base_latch_stiffness, control_freq)` | `float, float, int` | `None` | Reset stato + pre-calcola target_hold_steps |

**Firma completa di `update()`:**
```python
fsm.update(
    door_angle          : float,        # angolo corrente porta (rad)
    success_angle       : float,        # angolo sotto cui la porta è "chiusa"
    gripper_action      : float,        # azione gripper SAC ∈ [-1, 1]
    dist_handle         : float,        # ||eef_pos - handle_pos|| (m)
    handle_radius       : float,        # raggio corrente maniglia (m)
    handle_friction     : float,        # frizione corrente maniglia
    is_physically_closed: bool,         # gripper_width ≤ handle_diam + margin
    gripper_width       : float,        # apertura fisica gripper (m)
    prev_angle          : float,        # angolo porta step precedente
    control_freq        : int,          # Hz (per calcoli temporali)
    door_qpos           : float,        # qpos grezzo giunto porta
    eef_pos             : ndarray[3],   # posizione EEF (world frame)
    door_quat_mujoco    : ndarray[4],   # quaternione porta (wxyz MuJoCo)
    latch_stiffness     : float,        # stiffness corrente latch
    base_latch_stiffness: float,        # stiffness base (non randomizzata)
    beta_probs          : dict|None,    # output BetaNetwork (opzionale)
) → dict[str, bool]
# Ritorna: {just_grasped, just_succeeded, just_hold_done, grasp_lost}
```

**Logica interna di `compute_retreat_pos()`:**
```python
# 1. Converte quaternione da formato MuJoCo (wxyz) a scipy (xyzw)
w, x, y, z = door_quat_mujoco
door_rot    = R_scipy.from_quat([x, y, z, w])

# 2. Estrae la normale alla superficie della porta
door_mat    = door_rot.as_matrix()       # matrice 3×3 di rotazione
door_normal = door_mat[:, 0]             # prima colonna = asse X locale porta
door_normal = door_normal / ||door_normal||  # normalizza

# 3. Calcola il target di retreat
retreat    = eef_pos + retreat_dist * door_normal
retreat[2] += retreat_z                  # alza leggermente il target
```

---

#### `PotentialBasedReward` — `reward_v2.py`

Calcola il reward a ogni step tramite il pattern `F = γΦ(s') − Φ(s)`.

**Costruttore:**
```python
PotentialBasedReward(cfg: TrainConfigV2, gamma: float = 0.95)
    self.cfg       = cfg
    self.gamma     = gamma        # fattore di sconto (= gamma di SAC)
    self._prev_phi = None         # Φ dello step precedente (memorizzato)
```

**Metodi:**

| Metodo | Input chiave | Output | Formula |
|--------|-------------|--------|---------|
| `reset()` | — | None | `_prev_phi = None` |
| `phi_reach(dist_handle, handle_radius, curriculum_lvl)` | distanza, raggio, livello | float | `w × exp(−dist / σ)` dove `σ = max(radius×3, 0.08)` |
| `phi_push(door_angle, door_max, gripper_action, grip_thresh, curriculum_lvl)` | angolo porta, grip | float | `w × closure × sigmoid(10×(grip − thresh))` |
| `phi_hold(hold_duration, target_steps, door_qpos)` | durata, target, qpos | float | `w × (dur/target) × (1 − \|qpos\|/tol)` |
| `phi_retreat(dist_to_target, max_dist)` | distanza target | float | `w × (1 − dist/max_dist)` |
| `compute(**kwargs)` | tutto lo stato corrente | `(float, bool, bool, dict)` | reward + terminated + truncated + rew_info |

**Come il potenziale fluisce nel reward:**

```
Step t:    phi_now  = phi_*(stato corrente)
           shaping  = gamma × phi_now  −  _prev_phi
           _prev_phi = phi_now          ← per il prossimo step

           reward  += shaping           ← aggiunto ai termini espliciti
```

**Firma completa di `compute()`:**
```python
reward_fn.compute(
    fsm_state           : FSMState,       # stato FSM corrente (fase, contatori)
    base_reward         : float,          # reward grezzo da Robosuite parent env
    door_angle          : float,          # angolo porta (rad)
    door_max            : float,          # angolo massimo porta (0.4 rad)
    door_qpos           : float,          # qpos giunto porta
    dist_handle         : float,          # distanza EEF-maniglia (m)
    dist_xy             : float,          # distanza XY EEF-maniglia (m)
    height_diff         : float,          # eef_z − handle_z (m)
    handle_radius       : float,          # raggio maniglia (m)
    handle_friction     : float,          # frizione maniglia
    grip_thresh         : float,          # soglia grip adattiva (da FSM)
    gripper_action      : float,          # azione gripper SAC ∈ [-1,1]
    gripper_width       : float,          # apertura fisica gripper (m)
    is_physically_closed: bool,           # contatto fisico verificato
    gripper_qpos        : ndarray|None,   # posizioni giunti gripper
    alignment           : float,          # max alignment su K direzioni [§3.3]
    flat_alignment      : float,          # |eef_x[2]| (piano)
    joint_vel           : ndarray|None,   # velocità giunti robot
    action              : ndarray[8],     # azione corrente SAC
    prev_eef_action     : ndarray[7],     # azione EEF step precedente
    eef_pos             : ndarray[3],     # posizione EEF (world frame)
    latch_qpos          : float,          # angolo latch (per termination check)
    door_qvel           : float,          # velocità angolare porta
    curriculum_lvl      : float,          # livello curriculum [0,1]
    just_grasped        : bool,           # evento FSM REACH→PUSH questo step
    just_succeeded      : bool,           # evento FSM PUSH→HOLD questo step
    just_hold_done      : bool,           # evento FSM HOLD→RETREAT questo step
    grasp_lost          : bool,           # evento FSM PUSH→REACH questo step
    terminated          : bool,           # env ha segnalato terminazione
    truncated           : bool,           # env ha segnalato truncation
) → (float, bool, bool, dict[str, float])
# Ritorna: (reward_totale, terminated, truncated, breakdown_per_termine)
```

---

#### `MultiApproachGrasp` — `grasp_strategy.py`

Gestisce le K=3 direzioni di approccio candidato.

**Costruttore:**
```python
MultiApproachGrasp(cfg: TrainConfigV2)
    self.cfg          = cfg
    self.n_candidates = cfg.grasp_n_candidates  # default 3
```

**Metodi:**

| Metodo | Input | Output | Descrizione |
|--------|-------|--------|-------------|
| `get_candidate_directions(handle_pos, eef_pos, door_quat_mujoco)` | pos maniglia, pos EEF, quat porta | `list[ndarray[3]]` | Restituisce K vettori unitari (top, lat-L, lat-R) |
| `compute_alignment(eef_quat, handle_pos, eef_pos, door_quat_mujoco)` | quat EEF, pos maniglia, pos EEF, quat porta | `(float, int, list[float])` | `(best_align, best_idx, all_aligns)` |
| `obs_features(eef_quat, handle_pos, eef_pos, door_quat_mujoco)` | *(stessi)* | `ndarray[K+1]` | Vettore osservazione: `[best, align_0, align_1, ..., align_{K-1}]` |

**Come viene calcolato l'alignment:**
```python
# 1. Ottieni K direzioni candidate
dirs = [dir_top, dir_lat_L, dir_lat_R]

# 2. Estrai asse z dell'EEF dalla sua quaternione
R_eef  = R_scipy.from_quat(eef_quat)   # eef_quat = xyzw (formato Robosuite)
eef_z  = R_eef.as_matrix()[:, 2]       # terza colonna = asse z locale EEF

# 3. Calcola alignment per ognuna
aligns = [abs(float(np.dot(eef_z, d))) for d in dirs]

# 4. Prendi il massimo
best_align = max(aligns)
best_idx   = int(np.argmax(aligns))
```

**Come `dir_lat_L` e `dir_lat_R` dipendono dalla porta:**
```python
w, x, y, z  = door_quat_mujoco
door_rot     = R_scipy.from_quat([x, y, z, w])

# Asse Y locale della porta → direzione laterale nel mondo
door_y_world = door_rot.apply([0.0, 1.0, 0.0])

dir_lat_L = -door_y_world   # approccio da sinistra
dir_lat_R = +door_y_world   # approccio da destra
```

---

#### `ExtendedDomainRandomizer` — `domain_rand_v2.py`

Modifica il modello MuJoCo **in-place** a ogni reset.
Dopo `randomize_episode()`, i valori correnti sono disponibili come attributi.

**Costruttore:**
```python
ExtendedDomainRandomizer(cfg: TrainConfigV2, sim_model)
    self.cfg   = cfg
    self.model = sim_model  # mujoco sim.model (diretto puntatore)
    self._init_base_values(sim_model)  # legge i valori default da MuJoCo
```

**Attributi correnti (aggiornati ogni reset):**

| Attributo | Tipo | Descrizione |
|-----------|------|-------------|
| `current_handle_radius` | float | Raggio maniglia corrente (m) |
| `current_handle_friction` | float | Frizione maniglia corrente |
| `current_latch_stiffness` | float | Stiffness molla latch corrente |
| `current_hinge_damping` | float | Damping cerniera corrente |
| `current_door_mass` | float | Massa porta corrente (kg) |
| `base_handle_radius` | float | Raggio maniglia default (non randomizzato) |
| `base_latch_stiffness` | float | Stiffness latch default (non randomizzato) |
| `base_hinge_damping` | float | Damping default (non randomizzato) |
| `base_door_mass` | float | Massa default (non randomizzata) |

**Metodi pubblici:**

| Metodo | Input | Output | Cosa fa |
|--------|-------|--------|---------|
| `randomize_episode(curriculum_level)` | float | None | Chiama i 4 `_randomize_*()` privati |
| `obs_features()` | — | `ndarray[3]` | `[norm_stiffness, norm_damping, norm_mass]` ∈ [0,1] |

**Metodi privati (chiamati da `randomize_episode()`):**

| Metodo | Attributo MuJoCo modificato | Range |
|--------|----------------------------|-------|
| `_randomize_handle()` | `model.geom_size[handle_geom_id]`, `model.geom_friction[handle_geom_id]` | raggio ×0.7–1.4, frizione ×0.3–1.2 |
| `_randomize_latch_stiffness()` | `model.jnt_stiffness[latch_joint_id]` | ×0.5–2.0 |
| `_randomize_hinge_damping()` | `model.dof_damping[hinge_joint_id]` | ×0.3–1.5 |
| `_randomize_door_mass()` | `model.body_mass[door_body_id]` | ×0.5–2.0 |

**Come `obs_features()` normalizza:**
```python
# Esempio per stiffness:
base_s   = self.base_latch_stiffness or 1.0
s_min    = base_s * 0.5   # minimo del range
s_max    = base_s * 2.0   # massimo del range
norm_s   = clip((current_latch_stiffness − s_min) / (s_max − s_min), 0, 1)
# Stesso schema per damping e massa
```

---

#### `BetaNetwork` — `beta_net.py` *(disabilitata di default)*

Tre MLP indipendenti, una per fase.

**Costruttore:**
```python
BetaNetwork(cfg: TrainConfigV2)
    self.cfg = cfg
    # Tre reti MLP indipendenti:
    self._net_reach = MLP(input_dim=5, hidden=64, output=1)  # REACH→PUSH
    self._net_push  = MLP(input_dim=5, hidden=64, output=1)  # PUSH→HOLD
    self._net_hold  = MLP(input_dim=4, hidden=64, output=1)  # HOLD→RETREAT
```

**Input per ogni rete:**

| Rete | Features di input |
|------|------------------|
| `_net_reach` | `[dist_handle, handle_radius, handle_friction, gripper_width, gripper_action]` |
| `_net_push`  | `[door_angle, door_speed, dist_handle, gripper_action, norm_door_mass]` |
| `_net_hold`  | `[hold_duration/target, door_qpos, door_qvel, norm_latch_stiffness]` |

**Metodo pubblico `predict(**kwargs) → dict`:**
```python
beta_net.predict(...) → {"beta_reach": float, "beta_push": float, "beta_hold": float}
```

Quando `use_beta_net = False` (default):
- nessuna inferenza neurale viene eseguita
- ritorna sempre `{"beta_reach": 1.0, "beta_push": 1.0, "beta_hold": 1.0}`
- effetto netto: tutti i gate FSM sono sempre `True` → nessun impatto comportamentale

---

#### `AdvancedGeneralizedDoorEnv` — `env_v2.py`

Classe principale che eredita da `RoboSuiteDoorCloseGymnasiumEnv`.

**Costruttore — oggetti creati nell'ordine:**
```python
AdvancedGeneralizedDoorEnv(cfg: TrainConfigV2, render_mode=None)

    super().__init__(cfg, render_mode)
        # Crea self._rs_env (Robosuite DoorClose)
        # Definisce action_space (Box[-1,1]^8) e observation_space (Box^~47)

    # ID del body porta in MuJoCo
    self.door_body_id = sim.model.body_name2id("Door_main")
    self.base_pos     = sim.model.body_pos[door_body_id].copy()   # posizione default
    self.base_quat    = sim.model.body_quat[door_body_id].copy()  # orientazione default

    # Moduli v2
    self._fsm            = AdaptiveFSM(cfg)
    self._reward_fn      = PotentialBasedReward(cfg, gamma=cfg.gamma)
    self._grasp_strategy = MultiApproachGrasp(cfg)
    self._domain_rand    = ExtendedDomainRandomizer(cfg, sim.model)
    self._beta_net       = BetaNetwork(cfg)

    # Variabili di diagnostica
    self._prev_action     = zeros(action_space.shape)
    self._prev_eef_action = zeros(action_space.shape[0] - 1)
    self._prev_door_angle = None
    self._diag_step       = 0
```

**Metodi pubblici:**

| Metodo | Firma | Descrizione |
|--------|-------|-------------|
| `reset(seed, options)` | `→ (ndarray[~47], dict)` | Reset fisico + randomizzazione + FSM reset |
| `step(action)` | `ndarray[8] → (ndarray[~47], float, bool, bool, dict)` | Passo simulazione + FSM + reward |
| `set_curriculum_level(level)` | `float → None` | `curriculum_level = clip(level, 0, 1)` |
| `_flatten_obs(obs)` | `dict → ndarray[~47]` | Assembla osservazione completa |

**Properties:**

| Property | Ritorna | Condizione |
|----------|---------|-----------|
| `_grasp_phase` | bool | `fsm.state.phase == PUSH` |
| `_success_latched` | bool | `fsm.state.phase in {HOLD, RETREAT}` |
| `_ready_to_retreat` | bool | `fsm.state.phase == RETREAT` |

---

### 7.8 — Dettaglio Completo del Flusso `env.step()`

```
def step(self, action: ndarray[8]):

    [A] Clip e smoothing
        action = clip(action, -1, 1)
        if alpha < 1.0:
            action = alpha * action + (1 - alpha) * self._prev_action

    [B] Freeze braccio in HOLD
        if phase == PHASE_HOLD:
            action[:-1] = 0.0          # solo gripper rimane attivo

    [C] Freeze braccio in RETREAT (se già al target)
        elif phase == PHASE_RETREAT:
            eef_pos      = sim.data.site_xpos[eef_site_id]
            dist_retreat = ||eef_pos − retreat_pos||
            if dist_retreat < cfg.return_pos_tol OR return_hold >= cfg.return_hold_steps:
                action = zeros(8)

    [D] Passo fisico MuJoCo
        obs, _, rs_done, info = self._rs_env.step(action)
        self._step_count += 1

    [E] Variabili di stato dal simulatore
        door_angle  = self._get_door_angle()
        door_qpos   = sim.data.qpos[hinge_qpos_addr]
        door_qvel   = sim.data.qvel[door_hinge_dof_adr]
        door_speed  = |prev_angle − door_angle| × control_freq
        dist_handle = ||eef_pos − handle_pos||

    [F] Beta-network (solo se use_beta_net=True, altrimenti 1.0 fissi)
        beta_probs = self._beta_net.predict(
            dist_handle, handle_radius, handle_friction,
            gripper_width, gripper_action,
            door_angle, door_speed, door_qpos, door_qvel,
            hold_duration, target_hold_steps,
            norm_latch_stiffness, norm_door_mass
        )
        → {"beta_reach": float, "beta_push": float, "beta_hold": float}

    [G] Calcolo contatto fisico
        gripper_width    = sum(|gripper_qpos|)
        handle_diam      = current_handle_radius × 2.0
        is_phys_closed   = (0.015 ≤ gripper_width ≤ handle_diam + 0.025)

    [H] FSM update (core della logica di controllo)
        fsm_events = self._fsm.update(
            door_angle, success_angle, gripper_action,
            dist_handle, handle_radius, handle_friction,
            is_phys_closed, gripper_width, prev_angle,
            control_freq, door_qpos, eef_pos,
            door_quat_mujoco, latch_stiffness,
            base_latch_stiffness, beta_probs
        )
        → {just_grasped: bool, just_succeeded: bool,
           just_hold_done: bool, grasp_lost: bool}

    [I] Aggiornamento return_hold in RETREAT
        if phase == PHASE_RETREAT AND retreat_pos is not None:
            dist_r = ||eef_pos_site − retreat_pos||
            if dist_r < cfg.return_pos_tol:
                self._fsm.state.return_hold += 1
            else:
                self._fsm.state.return_hold = 0

    [J] Grasp alignment multi-approccio (§3.3)
        best_align, best_idx, all_aligns = self._grasp_strategy.compute_alignment(
            eef_quat, handle_pos, eef_pos, door_quat_mujoco
        )
        flat_alignment = |R_eef.as_matrix()[2, 0]|

    [K] Base reward dal parent env (invariato da v1)
        base_reward, terminated, truncated = super()._calculate_reward(
            action, obs, rs_done, door_angle,
            prev_angle, fsm_events["just_succeeded"]
        )

    [L] Soglia grip adattiva (serve al reward)
        grip_thresh = self._fsm.grip_thresh(current_handle_friction)

    [M] Reward v2 (potential-based)
        reward, terminated, truncated, rew_info = self._reward_fn.compute(
            fsm_state, base_reward, door_angle, door_max=0.4,
            door_qpos, dist_handle, dist_xy, height_diff,
            handle_radius, handle_friction, grip_thresh,
            gripper_action, gripper_width, is_phys_closed,
            gripper_qpos, alignment=best_align, flat_alignment,
            joint_vel, action, prev_eef_action, eef_pos,
            latch_qpos, door_qvel, curriculum_lvl=self.curriculum_level,
            just_grasped=fsm_events["just_grasped"],
            just_succeeded=fsm_events["just_succeeded"],
            just_hold_done=fsm_events["just_hold_done"],
            grasp_lost=fsm_events["grasp_lost"],
            terminated, truncated
        )

    [N] Aggiornamento prev_action
        self._prev_action     = action.copy()
        self._prev_eef_action = action[:-1].copy()

    [O] Info dict per callback
        info["is_success"]         = phase in {HOLD, RETREAT}
        info["door_angle"]         = door_angle
        info["door_qpos"]          = door_qpos
        info["latch_qpos"]         = latch_qpos
        info["fsm_phase"]          = fsm_state.phase
        info["fsm_phase_name"]     = fsm_state.phase_name
        info["hold_duration"]      = fsm_state.hold_closed_duration
        info["target_hold_steps"]  = fsm_state.target_hold_steps
        info["curriculum_level"]   = self.curriculum_level
        info["best_grasp_align"]   = best_align
        info["best_grasp_dir_idx"] = best_idx

    [P] Osservazione v2 (~47 dim)
        obs_v2 = self._flatten_obs(obs)
            ├─► base_flat = super()._flatten_obs(obs)      (~32 dim)
            ├─► custom = [dist, radius, friction,           (8 dim)
            │             fsm_onehot×4, hinge_qpos]
            ├─► grasp_feats = grasp_strategy.obs_features(  (4 dim)
            │       eef_quat, handle_pos, eef_pos, door_quat)
            └─► physics_feats = domain_rand.obs_features()  (3 dim)
            → np.concatenate([base_flat, custom, grasp_feats, physics_feats])

    return obs_v2, reward, terminated, truncated, info
```

---

### 7.9 — Tabella Completa dei Termini di Reward per Fase

#### REACH — Avvicinamento alla maniglia

| Termine | Chiave `rew_info` | Formula | Condizione |
|---------|------------------|---------|-----------|
| Shaping potential | `phi_shape` | `γ × Φ_reach(s') − Φ_reach(s)` | Se `use_potential_reward=True` |
| Distanza 3D | `dist_3d` | `−5.0 × k × dist_handle` | Se no potential |
| Distanza XY | `dist_xy` | `−3.0 × k × dist_xy` | Se no potential |
| Distanza Z | `dist_z` | `−15.0 × k × \|height_diff\|` | Se no potential |
| Approccio troppo basso | `app_blw` | `−3.0 × \|height_diff + 0.005\|` | Se `height_diff < −0.005` |
| Approccio troppo alto | `app_top` | `−5.0 × height_diff × grip` | Se `height_diff > 0.03 AND grip > 0.2` |
| Allineamento K direzioni | `align` | `−w × (1 − best_align) × prox_factor` | Sempre |
| Allineamento piano | `flat` | `−0.5 × flat_align × prox_factor` | Sempre |
| Gripper aperto lontano | `grip` | `−1.0 × (grip − (−0.85))` | Se `dist > 0.025 AND grip > −0.85` |
| Gripper chiuso vicino | `grip` | `+2.5 × norm_g` | Se `dist ≤ 0.025` |
| Bonus REACH→PUSH | `phase_trans` | `+10.0` | Una volta su `just_grasped` |
| Penalità grasp perso | `grasp_lost_pen` | `−5.0` | Su `grasp_lost` |
| Smoothness | `smoothness` | `−w_smooth × \|\|Δaction_eef\|\|` | Tutte le fasi |
| Base reward parent | `base` | variabile | Tutte le fasi |

#### PUSH — Chiusura della porta

| Termine | Chiave `rew_info` | Formula | Condizione |
|---------|------------------|---------|-----------|
| Shaping potential | `phi_shape` | `γ × Φ_push(s') − Φ_push(s)` | Se `use_potential_reward=True` |
| Distanza 3D | `dist_3d` | `−5.0 × dist_handle` | Sempre |
| Distanza Z | `dist_z` | `−15.0 × \|height_diff\|` | Sempre |
| Progresso porta | `door_prog` | `+2000 × Δangle_min` | Se no potential, grip > thresh |
| Penalità lift | `lift_pen` | `−2.0 × action[2]` | Se `action[2] > 0.05` |
| Penalità azione | `act_pen` | `−0.005 × \|\|action_eef\|\|` | Sempre |
| Distanza grasp perso | `dist_lost` | `−6.0 × max(0, dist − 0.05)` | Su `grasp_lost` |
| Grip perso | `grip_lost` | `−5.0 × \|grip − thresh\|` | Su `grasp_lost` |
| Grip insufficiente | `grip` | `−5.0 × (1 − grip)` | Se `grip < 1.0` e non perde |

#### HOLD — Mantenimento porta chiusa

| Termine | Chiave `rew_info` | Formula | Condizione |
|---------|------------------|---------|-----------|
| Shaping potential | `phi_shape` | `γ × Φ_hold(s') − Φ_hold(s)` | Se `use_potential_reward=True` |
| Stabilità | `hold` | `+1.0 − \|door_qpos\|` | Se `\|door_qpos\| < 0.03` |
| Bounce penalty | `hold_bounce` | `−20.0 × \|door_qpos\|` | Se porta non chiusa |
| Damping velocità | `hold_veldamp` | `−25.0 × \|door_qvel\|` | Se `\|door_qvel\| > 0.01` |
| Slip fisico | `hold_slip` | `−5.0` | Se `is_phys_closed = False` |
| Grip OK | `hold_grip` | `+1.0` | Se `grip > thresh` |
| Grip insufficiente | `hold_grip` | `−2.0 × \|grip − thresh\|` | Se `grip < thresh` |
| Drop penalty | `hold_drop_pen` | `−10.0 × \|grip\|` | Se `grip < 0.0` |
| Joint freeze | `hold_jnt_freeze` | `−1.0 × \|\|joint_vel\|\|` | Se joint_vel disponibile |
| Arm frozen | `hold_act` | `+1.0` | Se `\|\|action_eef\|\| < 0.05` |
| Arm attivo | `hold_act` | `−2.0 × action_norm` | Se arm si muove |
| Wrist torsion | `hold_flat` | `−2.0 × flat_alignment` | Sempre |
| Distanza EEF | `hold_dist` | `−3.0 × (dist − 0.06)` | Se `dist > 0.06` |

#### RETREAT — Sfilamento dalla maniglia

| Termine | Chiave `rew_info` | Formula | Condizione |
|---------|------------------|---------|-----------|
| Shaping potential | `phi_shape` | `γ × Φ_retreat(s') − Φ_retreat(s)` | Se `use_potential_reward=True` |
| Gripper aperto | `ret_grip` | `+2.0` | Se `grip < −0.85` |
| Gripper chiuso | `ret_grip` | `−1.0 × \|grip + 1\|` | Se `grip ≥ −0.85` |
| Rotazione polso | `ret_rot` | `−3.0 × \|\|action[3:6]\|\|` | Sempre |
| Penalità laterale | `ret_lat` | `−5.0 × \|action[1]\|` | Se `dist_handle < 0.12` |
| Penalità discesa | `ret_down` | `−5.0 × \|action[2]\|` | Se `dist < 0.12 AND action[2] < 0` |
| Verso target | `ret_dir` | `+3.0 × dot(action[:3], dir_to_target)` | Se `dist_to_target > 0.02` |
| Componente perpendicolare | `ret_perp` | `−2.0 × \|\|perp\|\|` | Se `dist_to_target > 0.02` |
| Freeze al target | `ret_freeze` | `−20.0 × \|\|action_eef\|\|` | Se `dist_to_target ≤ 0.02` |
| Joint freeze progressivo | `ret_jnt_prog` | `−5.0 × fw × \|\|joint_vel\|\|` | Se joint_vel disponibile |
| Latch monitor | `latch_ret` | `−1.0 × \|latch_qpos\|` | Sempre |
| Stabilità porta | `hold` | `+1.0 − \|door_qpos\|` | Se `\|door_qpos\| < 0.03` |

---

### 7.10 — Dettaglio del Loop di Training

#### Oggetti e variabili principali in `main()`

```python
# ── Configurazione ──────────────────────────────────────────────────────
cfg = TrainConfigV2(run_dir="runs/close_gen_v2", num_envs=8, horizon=500)

# ── 8 environment paralleli (training) ─────────────────────────────────
env = DummyVecEnv([lambda: AdvancedGeneralizedDoorEnv(cfg)
                   for _ in range(cfg.num_envs)])
# DummyVecEnv: esegue gli env sequenzialmente nello stesso thread
# (no multiprocessing → no overhead IPC, più semplice da debuggare)

env = VecMonitor(env)
# Aggiunge tracciamento di: reward episodico, lunghezza, info["is_success"]

env = VecNormalize(env, norm_obs=True, norm_reward=True)
# Normalizza obs online con (obs - mean) / std  usando RunningMeanStd
# IMPORTANTE: va salvato/caricato insieme al modello (vecnormalize.pkl)

# ── 1 environment separato per valutazione ─────────────────────────────
eval_env = DummyVecEnv([lambda: AdvancedGeneralizedDoorEnv(cfg)])
eval_env = VecMonitor(eval_env)
eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False)
# norm_reward=False: non normalizza il reward durante valutazione

# ── Callback ────────────────────────────────────────────────────────────
scb = SuccessRateCallback(log_every=10_000)
# Legge info["is_success"] da ogni env ogni step
# Accumula successes / episodes → logga su TensorBoard

gcb = GraspDiagnosticCallbackV2(log_every=10_000)
# Legge env._grasp_phase e env._ready_to_retreat
# Accumula grasps / episodes e retreats / episodes → logga

ccb = AdaptiveCurriculumV2(scb, gcb, cfg)
# Ogni cfg.curriculum_check_freq step:
#   sr = scb.successes / scb.episodes
#   gr = gcb.grasps / gcb.episodes
#   se sr > 0.85 AND gr > 0.50:
#       env.env_method("set_curriculum_level", current_level + 0.05)
#       → chiama set_curriculum_level() su ognuno degli 8 env

eval_cb = CustomEvalCallbackV2(eval_env, ...)
# Ogni cfg.eval_freq step:
#   1. Sincronizza obs_rms da env training → eval_env
#   2. Valuta su n_eval_episodes=20 episodi (deterministic=True)
#   3. Se mean_success > best → salva best_model.zip + vecnormalize.pkl
#   4. Se degradazione > 25% per 2 eval consecutivi → ricarica best_model
#      (carica parametri rete con model.set_parameters(best_model.zip))
#      (carica obs_rms con pickle.load(vecnormalize.pkl))

# ── Modello SAC (Stable-Baselines3) ────────────────────────────────────
model = SAC(
    "MlpPolicy", env,
    policy_kwargs = dict(net_arch=[512, 512]),
    # Actor e Critic sono MLP con 2 hidden layer da 512 neuroni
    learning_rate = 3e-4,       # Adam optimizer per tutti i parametri
    buffer_size   = 1_000_000,  # capacità replay buffer
    batch_size    = 256,        # transizioni per gradient step
    gamma         = 0.95,       # fattore di sconto
    tau           = 0.005,      # soft update delle target networks
    train_freq    = 1,          # aggiorna ogni step
    gradient_steps= 2,          # 2 gradient steps per env step
    learning_starts=10_000,     # riempi buffer prima di aggiornare
    ent_coef      = "auto",     # temperatura entropia auto-tuned
)
```

#### Cosa succede ogni step nel training loop SB3

```
Per ogni step t = 1, 2, ..., total_steps:

1. action_t, _ = actor.predict(obs_t)   ← forward pass Actor
   action_t ~ Normal(μ(obs_t), σ(obs_t))  con reparametrization trick

2. obs_{t+1}, rew_t, done_t, info_t = env.step(action_t)
   ← 8 env in sequenza (DummyVecEnv)

3. replay_buffer.add(obs_t, action_t, rew_t, obs_{t+1}, done_t)

4. Se t >= learning_starts:
   Per i in range(gradient_steps=2):

     a. batch = replay_buffer.sample(256)
        → (s, a, r, s', d) uniformemente casuali

     b. Con torch.no_grad():
        a'_new, log_prob_new = actor.forward(s')
        q1_target = critic_target_1(s', a'_new)
        q2_target = critic_target_2(s', a'_new)
        q_target  = r + gamma × (1 − d) × (
                        min(q1_target, q2_target)
                        − alpha × log_prob_new
                    )

     c. loss_critic = MSE(critic_1(s,a), q_target)
                    + MSE(critic_2(s,a), q_target)
        optimizer_critic.zero_grad()
        loss_critic.backward()
        optimizer_critic.step()

     d. a_new, log_prob = actor.forward(s)
        loss_actor = mean(alpha × log_prob
                          − min(critic_1(s,a_new), critic_2(s,a_new)))
        optimizer_actor.zero_grad()
        loss_actor.backward()
        optimizer_actor.step()

     e. loss_alpha = mean(−alpha × (log_prob + target_entropy))
        optimizer_alpha.step()   # target_entropy = −dim(action) = −8

     f. # Soft update target networks
        critic_target_1 = tau × critic_1 + (1−tau) × critic_target_1
        critic_target_2 = tau × critic_2 + (1−tau) × critic_target_2
```

---

### 7.11 — Flusso Dati tra Moduli (Diagramma)

```
                    ┌─────────────────────────────────────────────────────┐
                    │                  train_gen_v2.py                    │
                    │                                                     │
                    │  cfg = TrainConfigV2(...)                           │
                    │  env = DummyVecEnv([AdvancedGeneralizedDoorEnv(cfg)]│
                    │  model = SAC("MlpPolicy", env, ...)                 │
                    │  model.learn(...)          ←── callbacks            │
                    └───────────────────┬─────────────────────────────────┘
                           action (8)  │  (obs, rew, done, info)
                                       ▼
                    ┌─────────────────────────────────────────────────────┐
                    │          AdvancedGeneralizedDoorEnv  (env_v2.py)    │
                    │                                                     │
                    │  ┌─────────────────┐  stiffness   ┌─────────────┐  │
                    │  │ExtendedDomain   │  ──────────► │AdaptiveFSM  │  │
                    │  │Randomizer       │              │(fsm_v2.py)  │  │
                    │  │(domain_rand_    │  obs_feat[3] │             │  │
                    │  │ v2.py)          │  ──────────► │FSMState     │  │
                    │  │                 │              │ .phase      │  │
                    │  │ modifica MuJoCo │              │ .one_hot    │  │
                    │  │ in-place        │              │ .retreat_pos│  │
                    │  └─────────────────┘              └──────┬──────┘  │
                    │         ▲ reset ogni ep.                 │ events  │
                    │                              grip_thresh │         │
                    │  ┌─────────────────┐                    ▼         │
                    │  │MultiApproachGrasp│  alignment  ┌─────────────┐  │
                    │  │(grasp_strategy  │  ──────────► │Potential    │  │
                    │  │ .py)            │              │BasedReward  │  │
                    │  │ K=3 candidati   │              │(reward_v2.  │  │
                    │  └─────────────────┘              │ py)         │  │
                    │                                    │             │  │
                    │  ┌─────────────────┐  beta_probs  │ (reward,    │  │
                    │  │BetaNetwork      │  ──────────► │  rew_info)  │  │
                    │  │(beta_net.py)    │  (opzionale) └─────────────┘  │
                    │  │ disabilitata    │                               │
                    │  └─────────────────┘                               │
                    │                                                     │
                    │  obs (~47 dim) =                                    │
                    │    base_flat (32) + custom (8) +                   │
                    │    grasp_feats (4) + physics_feats (3)             │
                    └───────────────────┬─────────────────────────────────┘
                                        │ (obs, reward, done, info)
                                        ▼
                    ┌─────────────────────────────────────────────────────┐
                    │                 VecNormalize                        │
                    │   obs_norm = clip((obs - obs_rms.mean)             │
                    │              / sqrt(obs_rms.var + 1e-8), -10, 10) │
                    └───────────────────┬─────────────────────────────────┘
                                        │ obs_norm (~47 dim)
                                        ▼
                    ┌─────────────────────────────────────────────────────┐
                    │                  SAC (SB3)                          │
                    │                                                     │
                    │  Actor MLP:  47 → 512 → 512 → 8   → action        │
                    │  Critic MLP: 55 → 512 → 512 → 1   → Q-value       │
                    │             (47 obs + 8 action)                    │
                    │                                                     │
                    │  Replay Buffer: 1M transizioni                      │
                    │    (obs, action, reward_norm, obs_next, done)      │
                    └─────────────────────────────────────────────────────┘
```


---

## 8. Riferimenti Bibliografici

| # | Autori | Anno | Titolo | Usato in |
|---|--------|------|--------|---------|
| [1] | Sutton, Precup, Singh | 1999 | Between MDPs and Semi-MDPs | `beta_net.py`, `fsm_v2.py` |
| [2] | Konidaris & Barto | 2009 | Skill Chaining | `fsm_v2.py` |
| [3] | Ng, Russell, Harada | 1999 | Policy Invariance Under Reward Transformations | `reward_v2.py` |
| [4] | Devlin & Kudenko | 2012 | Dynamic Potential-Based Reward Shaping | `reward_v2.py`, `config_v2.py` |
| [5] | Nachum et al. | 2018 | HIRO | — (futuro) |
| [6] | Schaul et al. | 2015 | Universal Value Function Approximators | `train_gen_v2.py` |
| [7] | Andrychowicz et al. | 2017 | Hindsight Experience Replay | `train_gen_v2.py` |
| [8] | Tobin et al. | 2017 | Domain Randomization for Transfer | `domain_rand_v2.py` |
| [9] | Mehta et al. | 2020 | Active Domain Randomization | `domain_rand_v2.py` |
| [10] | Portelas et al. | 2020 | Automatic Curriculum Learning Survey | `train_gen_v2.py` |
| [11] | Florensa et al. | 2017 | Reverse Curriculum Generation | `train_gen_v2.py` |
| [12] | Rajeswaran et al. | 2017 | Learning Complex Dexterous Manipulation | — |
| [13] | ManipForce | 2015 | Force-Based Manipulation Primitives | `grasp_strategy.py`, `fsm_v2.py` |
| [14] | Handa et al. | 2020 | DexPilot | `grasp_strategy.py` |
| [15] | ten Pas et al. | 2017 | Grasp Pose Detection in Point Clouds | `grasp_strategy.py` |
| [16] | Krakovna et al. | 2020 | Avoiding Side Effects in Complex Environments | `reward_v2.py` |
| [17] | Zhao et al. | 2020 | Sim-to-Real Transfer in Deep RL | `domain_rand_v2.py` |


---

### 8.1 — Architettura e struttura del codice

| Aspetto | v1 | v2 |
|---------|----|----|
| Organizzazione | **Monolitica**: un env che eredita da `RoboSuiteDoorCloseGymnasiumEnv`, logica di fase e reward intrecciate | **Modulare**: `env_v2` (orchestratore) + `fsm_v2` + `reward_v2` + `config_v2` + `domain_rand_v2` + `grasp_strategy` + `beta_net` |
| Fonte dei parametri | Costanti sparse nel codice | `TrainConfigV2`: dataclass unica, senza logica, passata a ogni modulo |
| Testabilità | Difficile isolare un modulo | Ogni modulo testabile in isolamento (es. `reward_v2` con un `FSMState` fittizio, senza MuJoCo) |
| Scrittura su MuJoCo | Diffusa | Concentrata in `domain_rand_v2` (unico modulo che scrive su `sim.model`) |

### 8.2 — FSM e transizioni

| Aspetto | v1 | v2 |
|---------|----|----|
| Struttura del task | Logica di fase **implicita** nella reward | **FSM esplicita a 4 fasi** REACH→PUSH→HOLD→RETREAT (`fsm_v2.py`) |
| Conferma grasp | N step consecutivi | N step consecutivi **+ Schmitt trigger + isteresi temporale** sul rilascio (§1.11, anti-chatter) |
| Check di contatto | Solo geometrico | `is_physically_closed` (controllo sull'apertura del gripper) |
| Terminazione fase (β) | Deterministica | Deterministica + gate `β` *opzionale* (`beta_net`, disattivato di default) |
| Archi di ritorno | Solo PUSH→REACH | Solo PUSH→REACH (identico), ma robusto al rumore via isteresi |

### 8.3 — Soglie (costanti → adattive alla fisica)

| Soglia | v1 | v2 |
|--------|----|----|
| Distanza grasp REACH→PUSH | `0.020 m` fissa | `1.5 × handle_radius + 0.005` (§1.1) |
| Soglia grip | `0.65` fissa | `0.75 − 0.10 × norm_friction` (§1.2) |
| Timer HOLD | `60` step fissi | `60 + extra(latch_stiffness)` (§1.3) |
| Target di retreat | `eef + [−0.13, 0, 0.04]` (frame **globale**) | `eef + dist × door_normal` (perpendicolare alla **porta**, §1.4) |

### 8.4 — Generalizzazione

| Dimensione | v1 | v2 |
|------------|----|----|
| Geometria/frizione maniglia | ✅ randomizzata | ✅ (identica) |
| Posizione/yaw porta | ✅ ma **non** guidata da curriculum | ±15 cm / ±17°, **guidata dal curriculum** e ora **selezionabile** (posa fissa `0` / variabile `1`, §1.15) |
| Rigidità latch, damping cerniera (per-DOF), massa porta | ❌ fisse | ✅ randomizzate (§1.7 / §3.4); fix di indicizzazione per-DOF (§1.10.D) |
| Curriculum | ❌ assente | ✅ livello 0→1; gate su `success_rate` *windowed* (§1.12); pinning a livello fisso (§1.15) |
| Approccio di grasp | 1 fisso (top-down) | **K=3** candidati, alignment = max (§1.8, ten Pas 2017) |

### 8.5 — Reward e penalità (struttura del codice)

| Aspetto | v1 | v2 |
|---------|----|----|
| Impianto | Somma di ~15 termini *ad hoc* a pesi fissi | Denso (REACH) + **potential shaping** (Ng 1999) in PUSH/HOLD/RETREAT + `door_prog` ratchet |
| Progresso porta | `2000 × Δangle` (oscillabile) | `door_prog` ratchet **di proprietà del reward** (`_min_door_angle` solo decrescente, §1.10.C) |
| Bonus grasp | `+50` istantaneo (**cliff**) | Bonus di transizione ~`+10` dal salto di Φ cumulativo continuo (§1.9.D), niente cliff |
| Pesi del potenziale | — | Piccoli `O(1–5)` per evitare il **drift di sconto** `(γ−1)Φ` (§1.10.A); `γ=0.95` → invarianza esatta |
| Tassa grip in PUSH | Pretendeva gripper a `+1.0` esatto | Penalizza solo prese *sotto soglia*, in modo dolce (§1.13) |
| Co-evoluzione col curriculum | Pesi identici a ogni livello | `w_eff = w·(1 + k·level)` (Devlin & Kudenko 2012, §3.1) |

### 8.6 — Terminazione ed episodio

| Aspetto | v1 | v2 |
|---------|----|----|
| Fine episodio | (specifico v1) | **Stato terminale** a retreat-completo: RETREAT sostenuto + porta chiusa + latch neutro → `terminated` + `success_bonus` una tantum (§1.14) |
| Lunghezza episodio | — | `ep_len ≈ 125` (dopo §1.14; prima girava all'orizzonte 500) |
| Comportamento in RETREAT | — | Braccio **immobilizzato** + rilascio maniglia nella zona di settle allargata (§1.15) |
| Criterio di successo | (specifico v1) | `phase ∈ {HOLD, RETREAT}` — *permissivo* (vedi residui nel doc 08) |

### 8.7 — Osservazione ed esplorazione

| Aspetto | v1 | v2 |
|---------|----|----|
| Osservazione | Stato base (~40 dim) | + **feature fisiche** normalizzate (3) + **feature multi-approach** del grasp (4) → ~47 dim |
| Esplorazione SAC | `ent_coef` auto (`target_entropy = −dim_azione`) | + **pavimento di entropia** `target_entropy = −3` per evitare il collasso (§1.9.C / §1.13) |
| Architettura rete | MLP (512, 512) | MLP (512, 512) (identica) |

---

## 9. Spunti di Teoria e Inquadramento Bibliografico (per la stesura della tesi)

> Sezione aggiunta come supporto alla stesura: ogni pilastro del progetto è qui
> ricondotto alla letteratura di riferimento con citazioni inline `[n]` (numerazione della
> tabella §8; le voci nuove `[18]+` sono definite nella **Bibliografia estesa**, §11).
> L'obiettivo è fornire i *spunti teorici* da espandere nei capitoli di background e
> metodo della tesi, collegandoli alle scelte implementative e alle correzioni §1.9–§1.15.

### 9.1 — RL a massima entropia: perché SAC (e perché il «pavimento» di entropia)

Il problema di chiusura della porta è un controllo continuo ad alta dimensionalità
(azione 8-D, osservazione ~47-D) con dinamiche di contatto: un attore stocastico
off-policy con riuso del replay buffer è la scelta naturale. **Soft Actor-Critic** `[18]`
ottimizza l'obiettivo a *massima entropia* `J(π) = Σ_t E[r_t + α·H(π(·|s_t))]`, dove il
termine di entropia `H` incentiva l'esplorazione e la robustezza, e la temperatura `α`
viene auto-tarata per mantenere un'entropia-bersaglio `[18]`. Questo inquadra
teoricamente la patologia §1.9.C/§1.13 (collasso di `ent_coef`): con il bersaglio di
default `−dim(azione)` la temperatura collassava *prima* che la policy scoprisse la
sequenza di chiusura, congelandola in un ottimo locale di «accampamento». Il
**pavimento di entropia** (`target_entropy = −3`) è esattamente l'intervento previsto
dal meccanismo di auto-tuning di `[18]`: alza il bersaglio per tenere viva
l'esplorazione attraverso la finestra di scoperta, senza impedire la convergenza finale.
*Spunto per la tesi:* discutere il trade-off esplorazione/sfruttamento come scelta di
`target_entropy`, citando l'auto-tuning della temperatura di `[18]`.

### 9.2 — Astrazione temporale: la FSM come *options*

La FSM a 4 fasi (REACH→PUSH→HOLD→RETREAT) è formalizzabile nel framework delle
**options** `[1]`: ogni fase è un'opzione `⟨I, π, β⟩` con *initiation set* `I`, policy
interna `π` e funzione di terminazione `β(s)`. Nella nostra FSM `β` è **deterministica**
(0/1, le soglie di transizione); il modulo `beta_net.py` (disattivato di default) la
renderebbe *appresa e probabilistica*, in linea con l'idea di apprendere le
**precondizioni** delle opzioni dello *skill chaining* `[2]`. Le soglie *adattive*
(§1.1–§1.4) sono una forma leggera della stessa idea: invece di apprendere la
precondizione da zero `[2]`, si parametrizza la soglia esistente in funzione del contesto
fisico (raggio, frizione, rigidità, normale alla porta). L'estensione gerarchica a due
livelli — un controller alto che propone sotto-goal a uno basso — è l'orizzonte di
**HIRO** `[5]`. *Spunto per la tesi:* presentare la FSM esplicita come *prior strutturale*
che riduce l'orizzonte effettivo di credito, citando `[1]` per il formalismo e `[2]` per
la generalizzazione delle precondizioni.

### 9.3 — Reward shaping potenziale: garanzia di invarianza e il «drift di sconto»

Il cuore teorico del reward v2 è il **teorema di invarianza della policy** `[3]`:
aggiungendo al reward il termine `F(s,a,s') = γ·Φ(s') − Φ(s)` con `Φ` funzione di
potenziale, l'insieme delle policy ottimali **non cambia**. Questo legittima l'uso di `Φ`
come *guida* senza distorcere l'obiettivo. La trappola scoperta nei run (§1.10.A) è
direttamente leggibile dalla formula: sostare in una regione a potenziale `Φ` costa
`(γ−1)·Φ` per step; con pesi grandi (`Φ ~ 75–100`) e `γ = 0.95` la penalità implicita
diventa `≈ −4/−5` per step, sufficiente a rendere negativo il valore di HOLD/RETREAT.
La correzione (pesi piccoli `O(1–5)`, `γ` mantenuto identico a quello di SAC) conserva
l'invarianza **esatta** di `[3]`. Il fatto che i pesi possano *scalare con il curriculum*
(`w_eff = w·(1+k·level)`) senza rompere la garanzia è il risultato di **Dynamic PBRS**
`[4]`: pesi dinamici preservano l'ottimo purché convergano (qui convergono perché
`level → 1.0`). *Spunto per la tesi:* derivare esplicitamente `(γ−1)Φ` come «tassa di
sosta» e usarla per giustificare quantitativamente la scelta dei pesi `phi_*`.

### 9.4 — RL goal-conditioned (contesto per le estensioni)

Sebbene v2 non sia goal-conditioned, due risultati inquadrano possibili estensioni: gli
**UVFA** `[6]` condizionano la funzione di valore su un goal `V(s,g)`, permettendo a una
sola rete di risolvere molti goal (es. livelli di chiusura parziali); l'**Hindsight
Experience Replay** `[7]` rietichetta le traiettorie fallite con il goal effettivamente
raggiunto, mitigando la sparsità del reward — tecnica compatibile con il nostro replay
buffer. *Spunto per la tesi:* citarli nel capitolo «lavori futuri» come via per rendere la
policy riusabile su gradi di chiusura variabili.

### 9.5 — Domain randomization e sim-to-real

La randomizzazione fisica estesa (rigidità latch, damping cerniera, massa porta, oltre a
geometria/frizione maniglia) è il contributo di generalizzazione *meccanica* del progetto,
fondato su **Tobin et al.** `[8]`: randomizzare ampiamente i parametri del simulatore rende
la policy robusta alle variazioni che esistono nel mondo reale. La survey di **Zhao et
al.** `[17]` indica la randomizzazione di *stiffness/damping/massa* come il fattore più
impattante per il transfer nei task di manipolazione con contatto — la motivazione diretta
di §3.4/§1.7. L'evoluzione naturale è la **Active Domain Randomization** `[9]`, che adatta
la distribuzione di randomizzazione verso le configurazioni in cui la policy fallisce di
più (non implementata, ma coerente con l'osservazione delle feature fisiche). *Spunto per
la tesi:* legare l'inclusione delle feature fisiche normalizzate in osservazione alla
nozione di policy *condizionata al contesto* di `[8]`.

### 9.6 — Curriculum learning: criterio di avanzamento corretto

Il curriculum sulla posa (posizione/yaw guidati da `curriculum_level`) si fonda
sull'idea di **Bengio et al.** `[19]`: presentare esempi in ordine di difficoltà crescente
accelera e stabilizza l'apprendimento. La survey di **Narvekar et al.** `[20]` formalizza il
curriculum per l'RL (generazione, sequenziamento, transfer) e motiva la scelta §1.12: il
cancello di avanzamento corretto è la *competenza sul livello corrente* — cioè il
`success_rate` — non un sottoprodotto di conteggio come `grasp_rate` (che dopo l'anti-chatter
§1.11 vale ~1 ed era una metrica fuorviante). La **Automatic Curriculum Learning** `[10]`
classifica le strategie (goal-/task-/reward-based: il nostro è task-based sulla posa, con
una componente reward-based §3.1), mentre la **Reverse Curriculum Generation** `[11]` (partire
da stati vicini al successo) è una valida alternativa di scheduling per le fasi post-grasp.
*Spunto per la tesi:* presentare lo step `Δlevel = 0.10` e il gate *windowed* come
istanziazione concreta dei principi di `[19, 20]`.

### 9.7 — Grasping e manipolazione contact-rich

La strategia multi-approccio (K=3 direzioni candidate, reward = max allineamento) si basa
sull'intuizione di **ten Pas et al.** `[15]`: il grasp è una posa 6-D e per ogni oggetto
esistono *molteplici* afferraggi validi; premiare il migliore tra K candidati evita di
penalizzare il robot quando un approccio (es. top-down) è geometricamente ostruito.
**DexPilot** `[14]` sottolinea l'importanza della rappresentazione del contatto, mentre i
*force-based manipulation primitives* `[13]` propongono di usare segnali di forza come
trigger di transizione — il flag `is_physically_closed` è un'approssimazione di questo
concetto basata sull'apertura del gripper. Per il contesto più ampio di manipolazione
dexterous appresa con RL profondo si veda **Rajeswaran et al.** `[12]`. *Spunto per la
tesi:* motivare le K direzioni come *robustezza all'errore di localizzazione* (un
approccio laterale può essere più tollerante di quello top-down), citando `[15]`.

### 9.8 — Specifica del reward, reward hacking e «side effects»

Le patologie §1.6 (oscillazione su `2000·Δangle`), §1.13 (accampamento in REACH) e §1.14
(mungitura della reward di RETREAT a episodio infinito) sono casi da manuale di **reward
hacking / misspecification**: la policy massimizza il proxy invece dell'obiettivo. È lo
stesso fenomeno discusso da **Turner et al.** `[16]` (l'esempio canonico dell'agente che
raccoglie i checkpoint invece di finire la gara). Le mitigazioni adottate — ratchet
monotono su `door_prog` (non sfruttabile), stato terminale a task completo con bonus
*sconto-aware* (§1.14), e la verifica del *valore atteso di fase* prima di irrigidire una
penalità (§1.13) — sono contromisure pratiche alla misspecification. *Spunto per la tesi:*
inquadrare la sezione «pathologie e correzioni» come studio di caso di reward design
robusto, citando `[16]` per il framing e `[3]` per la garanzia che lo shaping non
introduce nuovi ottimi.

---

## 10. Mappa Teoria → Implementazione (con citazioni)

| Scelta di progetto | Modulo / file | Fondamento teorico |
|--------------------|---------------|--------------------|
| Attore stocastico off-policy + temperatura auto-tarata | SAC in `train_gen_v2.py` | Haarnoja et al. `[18]` |
| Pavimento di entropia `target_entropy=−3` (anti-collasso) | `config_v2.py`, SAC | auto-tuning entropia `[18]` (§1.9.C/§1.13) |
| FSM a 4 fasi come *options* a terminazione deterministica | `fsm_v2.py` | Sutton, Precup, Singh `[1]` |
| Soglie adattive (precondizioni parametriche) | `fsm_v2.py` | Konidaris & Barto `[2]` |
| β-network (terminazione appresa, opzionale) | `beta_net.py` | `[1]`, `[2]` |
| Estensione gerarchica a due livelli (futuro) | — | Nachum et al. `[5]` |
| Reward shaping `F=γΦ(s′)−Φ(s)`, invarianza | `reward_v2.py` | Ng, Harada, Russell `[3]` |
| Pesi `Φ` scalati col curriculum | `reward_v2.py`, `config_v2.py` | Devlin & Kudenko `[4]` |
| Ratchet monotono `door_prog` (anti-oscillazione) | `reward_v2.py` | `[3]` (§1.6/§1.10.C) |
| Stato terminale + bonus sconto-aware (anti-mungitura) | `reward_v2.py`, `config_v2.py` | reward design `[16]`, `[3]` (§1.14) |
| Randomizzazione fisica estesa | `domain_rand_v2.py` | Tobin `[8]`, Zhao `[17]`, Mehta `[9]` |
| Feature fisiche in osservazione (policy context-conditioned) | `env_v2.py`, `domain_rand_v2.py` | `[8]` |
| Curriculum sulla posa, gate su `success_rate` | `train_gen_v2.py`, `config_v2.py` | Bengio `[19]`, Narvekar `[20]`, Portelas `[10]` |
| Pinning del livello (posa fissa/variabile, §1.15) | `config_v2.py`, `env_v2.py`, `train_gen_v2.py` | scheduling del curriculum `[20]` |
| Grasp multi-approccio (K candidati, max-align) | `grasp_strategy.py` | ten Pas `[15]`, Handa `[14]` |
| `is_physically_closed` come trigger di contatto | `env_v2.py`, `fsm_v2.py` | force-based primitives `[13]` |
| Reward di mantenimento contatto in PUSH (`grip_contact`, §1.16) | `reward_v2.py`, `config_v2.py` | reward genuino `R` `[3]`; qualità/contatto della presa `[15]`, `[13]` |
| Goal-conditioning / HER (estensioni future) | — | Schaul `[6]`, Andrychowicz `[7]` |
| Simulatore, ambiente, libreria RL | tutto il pacchetto | MuJoCo `[21]`, robosuite `[22]`, Stable-Baselines3 `[23]` |

---

## 11. Bibliografia Estesa (riferimenti completi)

> I riferimenti `[1]`–`[17]` mantengono la numerazione della tabella §8 e ne sono la
> versione con dettagli completi (autori, anno, sede). Le voci `[18]`–`[24]` sono fonti
> centrali del progetto non ancora elencate (algoritmo SAC, curriculum learning,
> strumenti software): è consigliabile citarle nella tesi. **Nessuna voce esistente è
> stata modificata**; due note di accuratezza sono segnalate per `[13]` e `[16]`.

1. R. S. Sutton, D. Precup, S. Singh (1999). *Between MDPs and Semi-MDPs: A Framework for Temporal Abstraction in Reinforcement Learning.* Artificial Intelligence, 112(1–2): 181–211.
2. G. Konidaris, A. Barto (2009). *Skill Discovery in Continuous Reinforcement Learning Domains using Skill Chaining.* Advances in Neural Information Processing Systems (NeurIPS) 22.
3. A. Y. Ng, D. Harada, S. Russell (1999). *Policy Invariance Under Reward Transformations: Theory and Application to Reward Shaping.* Int. Conf. on Machine Learning (ICML).
4. S. Devlin, D. Kudenko (2012). *Dynamic Potential-Based Reward Shaping.* Int. Conf. on Autonomous Agents and Multiagent Systems (AAMAS).
5. O. Nachum, S. Gu, H. Lee, S. Levine (2018). *Data-Efficient Hierarchical Reinforcement Learning (HIRO).* NeurIPS 31.
6. T. Schaul, D. Horgan, K. Gregor, D. Silver (2015). *Universal Value Function Approximators.* ICML.
7. M. Andrychowicz, F. Wolski, A. Ray, J. Schneider, R. Fong, P. Welinder, B. McGrew, J. Tobin, P. Abbeel, W. Zaremba (2017). *Hindsight Experience Replay.* NeurIPS 30.
8. J. Tobin, R. Fong, A. Ray, J. Schneider, W. Zaremba, P. Abbeel (2017). *Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World.* IEEE/RSJ IROS.
9. B. Mehta, M. Diaz, F. Golemo, C. J. Pal, L. Paull (2020). *Active Domain Randomization.* Conf. on Robot Learning (CoRL), PMLR 100.
10. R. Portelas, C. Colas, L. Weng, K. Hofmann, P.-Y. Oudeyer (2020). *Automatic Curriculum Learning For Deep RL: A Short Survey.* Int. Joint Conf. on Artificial Intelligence (IJCAI).
11. C. Florensa, D. Held, M. Wulfmeier, M. Zhang, P. Abbeel (2017). *Reverse Curriculum Generation for Reinforcement Learning.* CoRL, PMLR 78.
12. A. Rajeswaran, V. Kumar, A. Gupta, G. Vezzani, J. Schulman, E. Todorov, S. Levine (2018). *Learning Complex Dexterous Manipulation with Deep Reinforcement Learning and Demonstrations.* Robotics: Science and Systems (RSS). (arXiv:1709.10087, 2017.)
13. *Force-Based Manipulation Primitives* (etichettata «ManipForce, 2015» nella §8). **Nota di accuratezza:** non risulta un paper canonico con questo esatto titolo/anno; un lavoro «ManipForce» esiste ma è del 2025 e ha tutt'altro scopo (force-guided policy learning multimodale). Per l'idea citata — *primitive di manipolazione basate sulla forza / contact-rich* nel 2015 — un riferimento verificabile e pertinente è: S. Levine, N. Wagener, P. Abbeel (2015). *Learning Contact-Rich Manipulation Skills with Guided Policy Search.* IEEE ICRA. *(Da scegliere/verificare in base alla fonte che avevi in mente.)*
14. A. Handa, K. Van Wyk, W. Yang, J. Liang, Y.-W. Chao, Q. Wan, S. Birchfield, N. Ratliff, D. Fox (2020). *DexPilot: Vision-Based Teleoperation of Dexterous Robotic Hand-Arm System.* IEEE ICRA.
15. A. ten Pas, M. Gualtieri, K. Saenko, R. Platt (2017). *Grasp Pose Detection in Point Clouds.* The International Journal of Robotics Research (IJRR), 36(13–14): 1455–1473.
16. **Avoiding Side Effects in Complex Environments.** **Nota di accuratezza:** nella §8 è attribuito a «Krakovna et al.», ma con questo titolo esatto è di **A. M. Turner, N. Ratzlaff, P. Tadepalli (2020), NeurIPS 33** (metodo AUP). V. Krakovna et al. hanno lavori *correlati* sui side effects ma con titoli diversi (es. *Penalizing Side Effects using Stepwise Relative Reachability*, 2018; *Avoiding Side Effects by Considering Future Tasks*, 2020). Verificare quale dei due si intende citare.
17. W. Zhao, J. P. Queralta, T. Westerlund (2020). *Sim-to-Real Transfer in Deep Reinforcement Learning for Robotics: a Survey.* IEEE Symposium Series on Computational Intelligence (SSCI).
18. T. Haarnoja, A. Zhou, P. Abbeel, S. Levine (2018). *Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL with a Stochastic Actor.* ICML. — E: T. Haarnoja et al. (2018). *Soft Actor-Critic Algorithms and Applications.* arXiv:1812.05905 (auto-tuning della temperatura / entropia-bersaglio).
19. Y. Bengio, J. Louradour, R. Collobert, J. Weston (2009). *Curriculum Learning.* ICML.
20. S. Narvekar, B. Peng, M. Leonetti, J. Sinapov, M. E. Taylor, P. Stone (2020). *Curriculum Learning for Reinforcement Learning Domains: A Framework and Survey.* Journal of Machine Learning Research (JMLR), 21(181): 1–50.
21. E. Todorov, T. Erez, Y. Tassa (2012). *MuJoCo: A Physics Engine for Model-Based Control.* IEEE/RSJ IROS.
22. Y. Zhu, J. Wong, A. Mandlekar, R. Martín-Martín, A. Joshi, S. Nasiriany, Y. Zhu (2020). *robosuite: A Modular Simulation Framework and Benchmark for Robot Learning.* arXiv:2009.12293.
23. A. Raffin, A. Hill, A. Gleave, A. Kanervisto, M. Ernestus, N. Dormann (2021). *Stable-Baselines3: Reliable Reinforcement Learning Implementations.* JMLR, 22(268): 1–8.
24. G. Brockman, V. Cheung, L. Pettersson, J. Schneider, J. Schulman, J. Tang, W. Zaremba (2016). *OpenAI Gym.* arXiv:1606.01540. — Successore mantenuto: M. Towers et al. (2023), *Gymnasium*, software framework.


---

## 12. §1.16 — Grip in Chiusura: Reward di Mantenimento del Contatto

> Rifinitura comportamentale (qualità del moto, **non** del successo) introdotta dopo il
> raggiungimento del 100% in entrambe le modalità di posa (§1.15). Obiettivo: una presa
> più salda **durante la chiusura**, riducendo lo slittamento osservato nei play.

**Motivazione (osservazione).** A successo invariato, durante PUSH/HOLD la presa può
allentarsi sulla maniglia mentre la porta ruota lungo l'arco (residuo geometrico §3.1):
si manifesta come eventi `hold_slip` e transizioni `PUSH→REACH (grip=…)`.

**Sottigliezza fisica — perché NON «stringere di più».** Il criterio `is_physically_closed`
è vero quando `gripper_width` è nella *banda* di buona presa `[0.015, handle_diam + 0.025]`.
Su maniglie sottili, comandare più presa (`gripper_action → +1`) fa scendere `gripper_width`
**sotto** `0.015` → contatto dichiarato perso. Un reward che spingesse il gripper al massimo
sarebbe quindi controproducente: l'obiettivo corretto è **restare nella banda di contatto**,
non chiudere di più.

**Intervento (`reward_v2.py`, blocco PUSH).** Un unico termine positivo e limitato che
premia il *mantenimento del contatto* scalato sul progresso di chiusura:

```python
if is_physically_closed and door_max > 1e-6:
    closing_progress = float(np.clip(1.0 - door_angle / door_max, 0.0, 1.0))
    rew_info["grip_contact"] = self.cfg.w_grip_contact * closing_progress   # w = 0.5
```

`config_v2.py`: nuovo peso `w_grip_contact = 0.5`.

**Perché è §1.13-safe (verifica del valore atteso di fase).**
- **Positivo e limitato** (≤ `w_grip_contact`): non crea una «valle» negativa attorno a
  PUSH, quindi non genera l'attrattore di accampamento del §1.13.
- **Scalato sul progresso** (`closing_progress ≈ 0` a porta aperta): tenere ferma una porta
  *aperta* paga ~0 → nessun nuovo incentivo ad accamparsi tenendo la maniglia senza chiudere.
  Il segnale cresce solo mentre/dopo la chiusura, dove PUSH→HOLD scatta subito
  (`door_angle ≤ success_angle`), quindi non è «mungibile».
- **Vive in `R`, non nello shaping `Φ`**: l'invarianza di policy di Ng et al. (1999) `[3]`
  resta intatta; `grip_contact` è parte del reward genuino del task (premia lo stato di
  contatto desiderato), non un potenziale.
- **Indipendente dal `curriculum_level`** (nessun fattore `1 + k·level`): effetto **identico**
  a posa fissa (`--curriculum 0`) e variabile (`--curriculum 1`).

**Comando di validazione (un run per modalità).**

```bash
python -m close_generalized_v2.train_gen_v2 --curriculum 0 --total-steps 800000
python -m close_generalized_v2.train_gen_v2 --curriculum 1 --total-steps 1500000
```

**Segnali attesi nei log (conferma positiva).** Compare `grip_contact: +…` nelle frame di
PUSH; `success_rate` resta 100% in entrambe le modalità; `ep_len` ed `ent_coef` stabili;
**meno** eventi `hold_slip` e `PUSH→REACH (grip=…)` (la presa resta agganciata durante la
chiusura). Se lo slittamento persistesse, la leva successiva è rinforzare il contatto anche
in HOLD oppure stringere la banda geometrica di `is_physically_closed`.

**Mappatura teorica.** Premia lo *stato* di contatto della presa: collegato a
`is_physically_closed` come trigger di contatto (force-based primitives `[13]`) e alla nozione
di qualità della presa (ten Pas `[15]`); resta nel reward genuino `R`, quindi compatibile con
l'invarianza di Ng `[3]`.
