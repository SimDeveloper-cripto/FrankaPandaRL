# 🗺️ Mappa Completa del Codice — FrankaPandaRL / close_generalized

## Panoramica della Repository

```
FrankaPandaRL/
├── close_generalized/          ← Modulo principale (task generalizzato)
│   ├── env_gen.py              ← Ambiente RL generalizzato (FSM + Reward + Randomizzazione)
│   ├── train_gen.py            ← Entry point training + callbacks + play mode
│   └── diag_phase34.py         ← Diagnostica avanzata delle fasi 3 e 4
├── config/
│   └── train_close_config.py   ← Dataclass di configurazione (iperparametri + env params)
├── train_close.py              ← Classe base dell'ambiente Gymnasium + training base
├── eval_stats.py               ← Script di valutazione e generazione plots
├── scratch/                    ← Script di test ablazione e analisi
│   ├── test_wait_logic.py      ← Test: blocco transizione HOLD→RETREAT finché latch non neutro
│   ├── test_freeze_logic.py    ← Test: congelamento braccio in RETREAT finché latch non neutro
│   ├── test_override_grip.py   ← Test: forzatura apertura gripper negli ultimi 15 step di HOLD
│   ├── test_hold_freeze.py     ← Test: congelamento braccio durante HOLD (senza grip override)
│   ├── test_hold_freeze_grip.py← Test: congelamento braccio + grip check a transizione
│   ├── test_original_wait_logic.py ← Reimplementazione della wait logic storica (commit 517e021c)
│   ├── inspect_eval_stats.py   ← Utility: ispezione file .pkl di statistiche
│   ├── inspect_test_model.py   ← Utility: ispezione modello SAC e VecNormalize
│   ├── model_almost_almost_complete_21_05/ ← Checkpoint modello 21/05
│   │   ├── test_model.zip
│   │   ├── test_vn.pkl
│   │   ├── eval_stats.pkl
│   │   ├── avg_phase_time.png
│   │   ├── max_phase_dist.png
│   │   └── success_rate.png
│   ├── model_almost_complete_14_05/ ← Checkpoint modello 14/05
│   ├── plots/                  ← Grafici generati da eval_stats.py
│   └── eval_stats.pkl          ← Statistiche di valutazione serializzate
├── open_generalized/           ← Modulo task apertura porta (architettura analoga)
│   ├── env_goal_door.py
│   ├── play.py
│   ├── teacher.py
│   └── train_curriculum.py
├── runs/                       ← Output training (modelli, log TensorBoard)
├── docs_fin.md                 ← Documentazione finale fix e risultati
├── eval_stats.py               ← Script valutazione globale
├── inspect_env.py              ← Utility ispezione env
├── inspect_env_details.py      ← Utility ispezione env dettagliata
├── train_close.py              ← Classe base ambiente + training deterministico
├── train_open.py               ← Training task apertura (non usato nel close_generalized)
├── scratch.py                  ← Script scratch rapido
├── requirements.txt            ← Dipendenze Python (Windows/Linux)
└── requirements_mac.txt        ← Dipendenze Python (macOS)
```

---

## Gerarchia delle Classi

```
gymnasium.Env
└── RoboSuiteDoorCloseGymnasiumEnv   [train_close.py]
    └── GeneralizedDoorEnv           [close_generalized/env_gen.py]
        │
        ├── Eredita tutta la logica base di wrapping Robosuite
        ├── Aggiunge FSM a 4 stati
        ├── Aggiunge reward shaping denso
        ├── Aggiunge randomizzazione handle + posizione porta
        └── Override di: __init__, _flatten_obs, step, _calculate_reward, reset
```

---

## Dettaglio File per File

### 1. `config/train_close_config.py`

**Tipo:** Dataclass di configurazione pura  
**Dipendenze:** nessuna

```python
@dataclass
class TrainConfig:
    # Identità sessione
    seed     : int   = 42
    run_dir  : str   = "runs/close_det"
    tb_dir   : str   = "runs/tb"

    # Ambiente Robosuite
    env_name     : str   = "Door"           # Task Robosuite
    robot        : str   = "Panda"          # Robot Franka Panda
    horizon      : int   = 600              # Step max per episodio
    control_freq : int   = 30              # Hz del controller (30 step/s)

    # Flags reward e osservazioni
    reward_shaping       : bool  = True
    reward_scale         : float = 1.0
    use_object_obs       : bool  = True
    use_camera_obs       : bool  = False
    terminate_on_success : bool  = False

    # Vettorizzazione
    num_envs    : int  = 8
    vecnormalize: bool = True

    # Iperparametri SAC
    total_steps    : int   = 800_000
    learning_rate  : float = 3e-4
    buffer_size    : int   = 1_000_000
    batch_size     : int   = 256
    gamma          : float = 0.95            # Discount factor
    tau            : float = 0.005           # Soft update coefficient
    train_freq     : int   = 1
    gradient_steps : int   = 2
    learning_starts: int   = 10_000         # Step prima del primo aggiornamento
    ent_coef       : str   = "auto"         # Entropia automatica

    # Architettura rete
    policy_net_arch: Tuple[int, int] = (512, 512)  # 2 layer nascosti

    # Valutazione
    eval_freq       : int = 50_000
    n_eval_episodes : int = 10
    checkpoint_freq : int = 200_000

    # Condizioni di chiusura
    close_fraction          : float = 0.015  # 1.5% del range → porta "chiusa"
    init_open_min_fraction  : float = 0.70   # Apertura iniziale min (70%)
    init_open_max_fraction  : float = 1.00   # Apertura iniziale max (100%)

    # Reward base (usati da RoboSuiteDoorCloseGymnasiumEnv)
    w_progress   : float = 0.0
    w_delta      : float = 2.0
    w_action     : float = 0.0
    time_penalty : float = 0.5
    success_bonus: float = 5.0

    # Fase di ritorno
    enable_return_stage : bool  = True
    w_return_pos        : float = 2.0
    w_door_regress      : float = 4.0
    return_hold_steps   : int   = 10
    return_pos_tol      : float = 0.05      # 5cm = tolleranza per considerarsi "arrivati"
    action_smooth_alpha : float = 0.8       # Alpha EMA per smoothing azioni

    # Friction handle
    limit_handle_friction : bool  = True
    handle_friction_max   : float = 0.8

    # Distanza umano-porta (per randomizzazione)
    human_dist_min : float = 0.50
    human_dist_max : float = 0.60
```

---

### 2. `train_close.py` — Classe Base

**Classe principale:** `RoboSuiteDoorCloseGymnasiumEnv(gym.Env)`

#### `__init__(self, cfg, render_mode=None)`
- Crea l'env Robosuite (`suite.make`) con controller BASIC
- Individua il DOF del giunto porta (`Door_hinge`) e calcola:
  - `_door_hinge_qpos_adr`: indirizzo qpos del giunto cerniera
  - `_door_hinge_dof_adr`: indirizzo DOF (per velocità)
  - `_door_min`, `_door_max`: range angolare del giunto
  - `_success_angle`: soglia di chiusura = `_door_min + close_fraction * range`
- Costruisce `action_space` e `observation_space` (Box continui)
- Inizializza `_prev_action`, `_retreat_pos`

#### `_flatten_obs(self, obs) → np.ndarray`
- Concatena tutte le osservazioni scalari/array 1D in un vettore piatto
- Ordine deterministico tramite `sorted(obs_keys)`

#### `_get_door_angle(self) → float`
- Legge `qpos[hinge_qpos_addr]` e lo clippa a `[door_min, door_max]`

#### `reset(self, seed, options) → (obs, info)`
- Reset Robosuite
- Salva posizione iniziale EEF (`_start_eef_pos`) per il punto di retreat
- Imposta angolo iniziale porta: uniforme in `[init_open_min, init_open_max] × range`
- Reset `_success_latched`, contatori step

#### `step(self, action) → (obs, reward, terminated, truncated, info)`
- EMA smoothing: `action = alpha * action_raw + (1-alpha) * prev_action`
- Se `_success_latched`: scala azioni braccio × 0.2 (damping in HOLD base)
- Esegue `_rs_env.step(action)`
- Calcola `door_angle`, controlla transizione a `_success_latched`
- Chiama `_calculate_reward`
- Costruisce `info`: `is_success`, `door_angle`, `custom_ready_to_retreat`

#### `_calculate_reward(self, action, obs, rs_done, door_angle, prev_angle, just_succeeded) → (reward, terminated, truncated)`
- Calcola `delta_close` (progresso chiusura)
- `reward += w_progress * progress + w_delta * delta_close - time_penalty`
- `if just_succeeded: reward += success_bonus`
- Se `enable_return_stage` e `_success_latched`:
  - Penalizza regressione porta
  - Se `_ready_to_retreat`: reward `tanh`-shaped verso `_retreat_pos`
  - Se `_return_hold >= return_hold_steps`: `terminated = True`, `reward += 500`
- Clipping: `reward = clip(reward, -100, 100)`
- `truncated = step_count >= horizon`

#### Classi Callback in `train_close.py`

| Classe | Scopo |
|--------|-------|
| `SuccessRateCallback` | Logga `rollout/success_rate` ogni `log_every` step |
| `SaveVecNormalizeCallback` | Salva `vecnormalize.pkl` a ogni step |

#### Funzioni stand-alone in `train_close.py`

| Funzione | Scopo |
|----------|-------|
| `make_env_fn(cfg)` | Factory closure che crea `RoboSuiteDoorCloseGymnasiumEnv` wrappato in `Monitor` |
| `train(cfg)` | Training deterministico base: crea env, modello SAC, callbacks, lancia `learn` |
| `play(model_path, cfg)` | Modalità play: carica modello e esegue episodi con rendering visivo |
| `parse_args()` | Parser argomenti CLI |
| `main()` | Entry point CLI |

---

### 3. `close_generalized/env_gen.py` — Ambiente Generalizzato

**Classe principale:** `GeneralizedDoorEnv(RoboSuiteDoorCloseGymnasiumEnv)`

#### Costanti globali (reward weights & FSM thresholds)

```python
# FSM
_GRASP_CONFIRM_STEPS  = 5       # Step consecutivi per confermare grasp
_GRASP_LOSE_STEPS     = 4       # Step consecutivi per perdere grasp
_GRIPPER_CLOSE_THRESH = 0.65    # Azione gripper = "chiuso"
_GRIPPER_OPEN_THRESH  = -0.85   # Azione gripper = "aperto"
_APPROACH_HEIGHT_TOL  = 0.005   # Tolleranza altezza approccio (5mm)

# Reward weights
_W_REACH_3D       = 5.0         # Distanza EEF-handle 3D
_W_REACH_XY       = 3.0         # Distanza EEF-handle nel piano XY
_W_REACH_Z        = 15.0        # Differenza di altezza (penalità forte)
_W_LATERAL_ORI    = 1.5         # Allineamento laterale
_W_GRIPPER_OPEN   = 1.5         # Reward gripper aperto (Fase 1)
_W_GRIPPER_CLOSE  = 2.5         # Reward gripper chiuso (Fase 1, prossimità)
_W_PUSH_PENALTY   = 5.0         # Penalità push prematuro dall'alto
_W_APPROACH_BELOW = 3.0         # Penalità approccio da sotto
_W_GRASP_BONUS    = 50.0        # Bonus una-tantum per transizione REACH→PUSH
_W_GRASP_LOST     = 6.0         # Penalità perdita grasp in PUSH
_W_PROGRESS_GRASP = 2000.0      # Reward progresso chiusura porta in PUSH
_W_ACTION_PHASE2  = 0.005       # Penalità azione braccio in PUSH (piccola)
_W_ACTION_PHASE1  = 0.0         # Penalità azione braccio in REACH (nulla)
```

#### `__init__(self, cfg, render_mode=None)`
- Chiama `super().__init__`
- Recupera `door_body_id` (per randomizzazione posizione)
- Salva posizione e quaternione base della porta (`base_pos`, `base_quat`)
- Trova `handle_geom_id` (per randomizzazione geometria)
- Salva `base_friction` dell'handle
- Inizializza variabili di stato FSM:
  - `_grasp_phase`: bool
  - `_grasp_confirm_count`: int
  - `_grasp_lose_count`: int
  - `_return_hold`: int
  - `_diag_step`: int (per logging periodico)
  - `_prev_door_angle`: float|None

#### `set_curriculum_level(self, level: float)`
- Imposta `curriculum_level` in `[0.0, 1.0]`
- Controlla l'ampiezza della randomizzazione in `reset`

#### `_flatten_obs(self, obs) → np.ndarray`
- Estende l'osservazione base con 8 features aggiuntive:
  1. `dist` — distanza EEF-handle (norma euclidea 3D)
  2. `_current_handle_radius` — raggio attuale dell'handle (da randomizzazione)
  3. `_current_handle_friction` — frizione attuale (da randomizzazione)
  4. `fsm_reach` — one-hot: 1.0 se in REACH
  5. `fsm_push` — one-hot: 1.0 se in PUSH
  6. `fsm_hold` — one-hot: 1.0 se in HOLD
  7. `fsm_retreat` — one-hot: 1.0 se in RETREAT
  8. `hinge_qpos` — angolo corrente porta (letto live da `sim.data.qpos`)

> **Nota critica:** `obs['hinge_qpos']` da Robosuite è sempre 0.0 (cached); deve essere letto da `sim.data.qpos[hinge_qpos_addr]`.

#### `step(self, action) → (obs, reward, terminated, truncated, info)`

Flusso completo:
1. Clip action a `[-1, 1]`
2. EMA smoothing
3. **Branch HOLD/RETREAT** (se `_success_latched`):
   - HOLD: `action[:-1] = 0.0` (braccio completamente fermo)
   - RETREAT: nessuna modifica (scala 1.0); se arrivato → `action = zeros`
4. `_rs_env.step(action)` → `obs, _, rs_done, info`
5. Lettura `door_angle`, `prev_angle`
6. Check transizione a `_success_latched` (richiede `action[-1] > 0.80`)
7. `_calculate_reward(...)` → `reward, terminated, truncated`
8. Popola `info` con: `is_success`, `door_angle`, `door_qpos`, `latch_qpos`, `ready_retreat`
9. Ritorna `_flatten_obs(obs), reward, terminated, truncated, info`

#### `_calculate_reward(self, action, obs, rs_done, door_angle, prev_angle, just_succeeded)`

Vedi documento dedicato al reward (doc 04).

#### `reset(self, seed, options) → (obs, info)`

1. Reset di tutti i contatori FSM
2. **Randomizzazione handle** (sempre):
   - `r_scale ~ U(0.7, 1.4)` → `_current_handle_radius = 0.02 * r_scale`
   - `l_scale ~ U(0.8, 1.2)` → lunghezza handle scalata
   - `f_scale ~ U(0.3, 1.2)` → `_current_handle_friction = base_f * f_scale ∈ [0.05, 2.0]`
3. **Randomizzazione porta** (se `curriculum_level > 0`):
   - `p_var = 0.15 * curriculum_level` → offset posizione XY in `[-p_var, p_var]`
   - `r_var = 0.30 * curriculum_level` → yaw rotazione in `[-r_var, r_var]` rad
   - Se `human_dist_min/max` definiti: override della posizione X con distanza campionata
   - Calcola quaternione composto: `q_new = R(yaw) * q_base`
4. `super().reset(seed, options)` → esegue reset Robosuite + imposta angolo porta

---

### 4. `close_generalized/train_gen.py` — Training Generalizzato

#### Classi Callback

| Classe | Frequenza | Cosa logga/fa |
|--------|-----------|---------------|
| `GraspDiagnosticCallback` | ogni `log_every=10k` step | `custom/grasp_rate`, `custom/retreat_rate`, `custom/episodes` |
| `AdaptiveCurriculumCallback` | ogni `check_freq=25k` step | Incrementa `curriculum_level` di 0.05 se `sr > 0.85` e `grasp_rate > 0.50` |
| `CustomEvalCallback` | ogni `eval_freq=10k` step | Valuta su 20 episodi, salva best model, recovery da degradazione |
| `SuccessRateCallback` | ogni `10k` step | Da `train_close.py`, logga `rollout/success_rate` |

#### `GraspDiagnosticCallback._on_step()`
- Monitora le transizioni `_grasp_phase` (`False → True` = nuovo grasp)
- Monitora le transizioni `_ready_to_retreat` (`False → True` = nuovo retreat)
- Conta episodi completati (da `dones`)
- Logga `grasp_rate = grasps / episodes`, `retreat_rate = retreats / episodes`

#### `AdaptiveCurriculumCallback._on_step()`
- Ogni `check_freq` step:
  - Calcola `sr` (success rate recente) e `gr` (grasp rate recente)
  - Se `sr > 0.85` e `gr > 0.50`: `curriculum_level += 0.05` (max 1.0)
  - Se `sr > 0.85` ma `gr <= 0.50`: stampa warning (robot usa pushing senza grasp)
- Resetta contatori dopo ogni aggiornamento

#### `CustomEvalCallback._on_step()`
- Ogni `eval_freq` step:
  - Sincronizza `obs_rms` e `ret_rms` tra env training e eval
  - Lancia `evaluate()` su `n_eval_episodes=20` episodi deterministici
  - Se nuovo best: salva `best_model.zip` e `vecnormalize.pkl`
  - Se degradazione > 25% per 2 check consecutivi: carica best model salvato (recovery)

#### `CustomEvalCallback.evaluate() → (mean_reward, mean_length, mean_success)`
- Loop su `n_eval_episodes` episodi con `deterministic=True`
- Raccoglie reward, lunghezze, successi
- Ritorna medie

#### `main()` — Entry point

**Modalità PLAY** (`--play`):
1. Crea `GeneralizedDoorEnv` con `render_mode="human"`
2. Carica VecNormalize (se disponibile) per normalizzare manualmente obs
3. `curriculum_level = 1.0` (massima difficoltà)
4. Loop infinito: predice azione, esegue step, aggiunge marker visivi al viewer:
   - Sfera colorata: **rosso** (REACH), **giallo** (PUSH), **verde** (HOLD), **blu** (RETREAT)
   - Sfera ciano/grigia: stato gripper
   - Barra verticale verde/rosso: reward corrente

**Modalità TRAIN**:
1. Crea `DummyVecEnv` con `num_envs=8` istanze di `GeneralizedDoorEnv`
2. Avvolge in `VecMonitor`
3. Carica o crea `VecNormalize` (normalizza obs e reward)
4. Istanzia callbacks: `[scb, gcb, ccb, eval_cb]`
5. Crea eval env separato (normalizzazione obs senza reward)
6. Se `--resume`: carica pesi SAC e VecNormalize statistiche
7. Crea o carica modello `SAC("MlpPolicy", ...)`
8. `model.learn(total_timesteps, callback=[...], reset_num_timesteps=...)`
9. Salva `best_model` e `vecnormalize.pkl`

---

### 5. `close_generalized/diag_phase34.py` — Diagnostica

**Funzioni principali:**

| Funzione | Input | Output | Scopo |
|----------|-------|--------|-------|
| `test_latch_spring()` | - | `(return_steps, trajectory)` | Misura in quanti step il latch_joint torna a 0 con molla (senza gripper) |
| `test_hinge_damping()` | - | `velocities: list` | Misura velocità di bounce della cerniera dopo chiusura |
| `test_with_model()` | - | `dict risultati` | Esegue N_EPISODES con il modello e raccoglie dati T3-T6 |
| `find_latest_model(runs_dir)` | `str` | `str path` | Trova il file .zip più recente nella cartella |
| `make_env()` | - | `GeneralizedDoorEnv` | Factory env con `curriculum_level=1` |

**Test diagnostici (T1-T6):**

| ID | Nome | Cosa misura |
|----|------|-------------|
| T1 | Latch spring | Step per latch_qpos < 0.1 rad partendo da 1.2 rad |
| T2 | Hinge damping | Velocità di bounce della porta dopo impatto con frame |
| T3 | hold_act conflict | `action_norm` durante HOLD: il robot è davvero fermo? |
| T4 | ret_rot | `‖action[3:6]‖` durante RETREAT (penalità rotazione polso) |
| T5 | latch at transition | `latch_qpos` al momento esatto della transizione HOLD→RETREAT |
| T6 | Bounce check | Quante volte `door_qvel > 0.1` rad/s durante Phase 3 |

---

### 6. `eval_stats.py` — Valutazione e Plots

**Funzioni:**

| Funzione | Scopo |
|----------|-------|
| `make_env(cfg)` | Crea `GeneralizedDoorEnv` con `curriculum_level=1.0` |
| `classify_failure(...)` | Classifica il tipo di fallimento in base alla fase massima raggiunta |
| `run_eval(n_episodes, deterministic)` | Esegue N episodi, raccoglie statistiche complete |
| `print_summary(stats, mode_name)` | Stampa tabella riassuntiva |
| `create_plots(train_stats, eval_stats)` | Genera 3 grafici: success rate, phase distribution, avg time per phase |

**Tipi di fallimento classificati:**

```
SUCCESS              → Episodio completato con successo
REACH timeout        → Non ha raggiunto la maniglia entro horizon
GRASP lost           → In Phase 2, dist_handle > 0.08 m
PUSH timeout         → In Phase 2, non ha chiuso la porta
HOLD bounce/timeout  → In Phase 3, rimbalzo o timeout
RETREAT door bounce  → In Phase 4, porta si è riaperta (door_angle >= 0.03)
RETREAT latch not neutral → In Phase 4, latch non neutro (|latch_qpos| >= 0.08)
RETREAT timeout      → In Phase 4, non ha raggiunto retreat_pos
```

**Plots generati (`scratch/plots/`):**
1. `success_rate.png` — Barchart successo stocastic vs deterministico
2. `max_phase_dist.png` — Distribuzione fase massima raggiunta (stacked bar)
3. `avg_phase_time.png` — Tempo medio in ogni fase (grouped bar)

---

### 7. Script di Ablazione in `scratch/`

| File | Classe Env | Logica testata | Risultato |
|------|-----------|----------------|-----------|
| `test_wait_logic.py` | `WaitDoorEnv` | Blocca `_ready_to_retreat=True` finché `latch_qpos >= 0.15` | **0% successo** (deadlock) |
| `test_freeze_logic.py` | `FreezeDoorEnv` | In RETREAT: congela braccio finché `latch_qpos >= 0.15` | **0% latch neutro**, 439s medio |
| `test_override_grip.py` | `LogOverrideGripDoorEnv` | Forza `action[-1]=-1.0` negli ultimi 15 step di HOLD | Latch neutro solo a step 110 (durante RETREAT) |
| `test_hold_freeze.py` | `HoldFreezeDoorEnv` | `action[:-1]=0.0` durante HOLD (senza grip check) | Base per il fix finale |
| `test_hold_freeze_grip.py` | `HoldFreezeGripDoorEnv` | `action[:-1]=0.0` durante HOLD + `action[-1]>0.80` per transizione | Fix finale implementato |
| `test_original_wait_logic.py` | `ExactWaitDoorEnv` | Reimplementa wait logic storica (commit 517e021c) con `latch_is_neutral` check | 0% successo |

---

## Flusso di Esecuzione End-to-End

```
train_gen.py main()
│
├── TrainConfig(run_dir="runs/close_gen", num_envs=8, horizon=500)
│
├── DummyVecEnv([GeneralizedDoorEnv(cfg)] × 8)
│   └── VecMonitor → VecNormalize (norm_obs=True, norm_reward=True)
│
├── Callbacks:
│   ├── SuccessRateCallback     → logga success rate ogni 10k step
│   ├── GraspDiagnosticCallback → logga grasp/retreat rate ogni 10k step
│   ├── AdaptiveCurriculumCallback → incrementa curriculum ogni 25k step
│   └── CustomEvalCallback     → valuta e salva best model ogni 10k step
│
├── SAC("MlpPolicy", env, net_arch=[512,512], gamma=0.95, ...)
│
└── model.learn(total_steps, callback=[...])
    │
    └── Per ogni step:
        ├── model.predict(obs) → action
        ├── env.step(action)
        │   └── GeneralizedDoorEnv.step()
        │       ├── EMA smoothing
        │       ├── HOLD/RETREAT action override
        │       ├── _rs_env.step() → Robosuite / MuJoCo
        │       ├── FSM update (_grasp_phase, _success_latched, _ready_to_retreat)
        │       └── _calculate_reward() → reward, terminated, truncated
        ├── replay_buffer.add(obs, action, reward, next_obs, done)
        └── model.train() (ogni gradient_steps steps)
```

---

## Dipendenze Python Chiave

| Package | Versione | Ruolo |
|---------|---------|-------|
| `robosuite` | latest | Simulatore robot + env `Door` |
| `mujoco` | ≥ 2.3 | Motore fisico sottostante |
| `stable-baselines3` | latest | Implementazione SAC |
| `gymnasium` | latest | Interfaccia standard RL |
| `numpy` | latest | Calcoli numerici |
| `scipy` | latest | Trasformazioni rotazione (`R_scipy.from_euler`) |
| `torch` | latest | Backend rete neurale |
| `tensorboard` | optional | Logging training |
| `matplotlib` | latest | Generazione plots |
| `python-dotenv` | latest | Caricamento variabili ambiente |
