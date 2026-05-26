# 🧠 Rete Neurale — Soft Actor-Critic (SAC)

## Algoritmo: SAC (Soft Actor-Critic)

Il progetto usa **SAC** (Haarnoja et al., 2018) implementato da Stable-Baselines3.  
SAC è un algoritmo **off-policy**, **model-free**, **actor-critic** ottimizzato per spazi d'azione continui.

### Caratteristica chiave di SAC
SAC massimizza un obiettivo con **entropia massima**:

```
π* = argmax_π Σ_t E[r(s_t, a_t) + α · H(π(·|s_t))]
```

Dove `α` è il coefficiente di temperatura (entropia), appreso automaticamente (`ent_coef="auto"`).  
Questo garantisce che la policy sia sia performante che **esplori attivamente** ambienti complessi.

---

## Spazio delle Osservazioni (Input della Rete)

### Osservazioni base da Robosuite (in `train_close.py::_flatten_obs`)
Tutte le osservazioni scalari/1D fornite da Robosuite vengono concatenate in ordine lessicografico:

| Chiave | Dim | Descrizione |
|--------|-----|-------------|
| `robot0_eef_pos` | 3 | Posizione end-effector (x, y, z) |
| `robot0_eef_quat` | 4 | Orientazione end-effector (quaternione wxyz) |
| `robot0_gripper_qpos` | 2 | Posizione giunti gripper (sinistra, destra) |
| `robot0_gripper_qvel` | 2 | Velocità giunti gripper |
| `robot0_joint_pos` | 7 | Posizione giunti braccio Panda (7 DOF) |
| `robot0_joint_vel` | 7 | Velocità giunti braccio Panda |
| `handle_pos` | 3 | Posizione 3D della maniglia porta |
| `hinge_qpos` | 1 | Angolo giunto cerniera (Nota: valore cached=0, letto live in env_gen) |
| `door_to_robot0_eef_pos` | 3 | Vettore relativo porta→EEF |

> Totale base: ~32 dimensioni (esatta dipende da config Robosuite)

### Features aggiuntive di `GeneralizedDoorEnv` (in `env_gen.py::_flatten_obs`)
Appese in coda al vettore base:

| Feature | Dim | Valori | Descrizione |
|---------|-----|--------|-------------|
| `dist` | 1 | [0, ∞) | Distanza euclidea 3D EEF→handle |
| `_current_handle_radius` | 1 | [0.014, 0.028] | Raggio attuale handle (randomizzato) |
| `_current_handle_friction` | 1 | [0.024, 2.0] | Frizione attuale handle (randomizzata) |
| `fsm_reach` | 1 | {0, 1} | One-hot: fase REACH attiva |
| `fsm_push` | 1 | {0, 1} | One-hot: fase PUSH attiva |
| `fsm_hold` | 1 | {0, 1} | One-hot: fase HOLD attiva |
| `fsm_retreat` | 1 | {0, 1} | One-hot: fase RETREAT attiva |
| `hinge_qpos_live` | 1 | [−π, π] | Angolo porta letto live da `sim.data.qpos` |

**Totale features aggiuntive: 8**

### Dimensione totale input: ~40 dimensioni

Dopo `VecNormalize`, ogni feature è normalizzata con running mean/variance:
```
obs_norm = clip((obs - μ) / √(σ² + ε), -10, 10)
```
con `ε = 1e-8`, aggiornate in streaming durante il training.

---

## Spazio delle Azioni (Output della Rete)

Il controller Robosuite usato è **BASIC** (operational space control).

| Dimensione | Tipo | Range | Descrizione |
|------------|------|-------|-------------|
| `action[0]` | Continuo | [-1, 1] | Traslazione EEF asse X |
| `action[1]` | Continuo | [-1, 1] | Traslazione EEF asse Y |
| `action[2]` | Continuo | [-1, 1] | Traslazione EEF asse Z |
| `action[3]` | Continuo | [-1, 1] | Rotazione EEF asse X (roll) |
| `action[4]` | Continuo | [-1, 1] | Rotazione EEF asse Y (pitch) |
| `action[5]` | Continuo | [-1, 1] | Rotazione EEF asse Z (yaw) |
| `action[6]` | Continuo | [-1, 1] | Apertura/chiusura gripper (-1=aperto, +1=chiuso) |

**Dimensione totale output: 7**

---

## Architettura della Rete (MlpPolicy)

### Configurazione
```python
policy_kwargs = dict(net_arch=[512, 512])
```

SAC con `MlpPolicy` usa **tre reti distinte** ma con struttura condivisa:

### 1. Actor Network (π — Policy)
```
Input Layer:  ~40 nodi  (obs normalizzata)
              ↓
Hidden Layer 1: 512 nodi  (ReLU)
              ↓
Hidden Layer 2: 512 nodi  (ReLU)
              ↓
Output: μ(s)  7 nodi  (media azioni, lineare)
Output: log_σ(s)  7 nodi  (log-varianza, clippata in [-20, 2])
              ↓
Sampling: a = μ + σ · ε,  ε ~ N(0, I)
              ↓
Squashing: ã = tanh(a)   → output in [-1, 1]^7
```

**Parametri totali actor (stima):**
- Layer 1: 40 × 512 + 512 = ~21k parametri
- Layer 2: 512 × 512 + 512 = ~263k parametri
- Output μ: 512 × 7 + 7 = ~3.5k parametri
- Output log_σ: 512 × 7 + 7 = ~3.5k parametri
- **Totale actor: ~291k parametri**

### 2. Critic Network 1 (Q₁)
```
Input: concat(obs, action) → (~40 + 7) = ~47 nodi
              ↓
Hidden Layer 1: 512 nodi  (ReLU)
              ↓
Hidden Layer 2: 512 nodi  (ReLU)
              ↓
Output: Q₁(s, a)  1 nodo  (valore Q scalare)
```

### 3. Critic Network 2 (Q₂)
Identica a Q₁, con pesi separati. SAC usa il minimo `min(Q₁, Q₂)` per stabilità.

**Parametri totali per critic (stima):**
- Layer 1: 47 × 512 + 512 = ~24.6k parametri
- Layer 2: 512 × 512 + 512 = ~263k parametri
- Output: 512 × 1 + 1 = ~513 parametri
- **Totale critic: ~288k parametri**

**Totale parametri rete: ~867k parametri** (actor + Q1 + Q2)

> SAC mantiene anche copie "target" congelate di Q₁ e Q₂ (soft-updated con τ=0.005).

---

## Algoritmo di Aggiornamento SAC

### Raccolta Esperienza
- **8 ambienti paralleli** (`DummyVecEnv`)
- Ogni step raccoglie una transizione `(s, a, r, s', done)` per env
- Tutte le transizioni finiscono nel **Replay Buffer** (size: 1,000,000)
- `learning_starts = 10,000` step prima del primo aggiornamento

### Aggiornamento Rete (ogni step, `gradient_steps=2`)

#### Step 1 — Aggiornamento Critic (minimizzazione Bellman error)
```
y = r + γ · (1 - done) · [min(Q₁_target(s', ã'), Q₂_target(s', ã')) - α · log π(ã'|s')]
dove ã' ~ π(·|s')

Loss_critic = MSE(Q₁(s, a) - y) + MSE(Q₂(s, a) - y)
```

#### Step 2 — Aggiornamento Actor (massimizzazione reward + entropia)
```
Loss_actor = -E[min(Q₁(s, ã), Q₂(s, ã)) - α · log π(ã|s)]
dove ã ~ π(·|s)
```

#### Step 3 — Aggiornamento Temperatura (automatico)
```
Loss_α = -α · (log π(ã|s) + H_target)
dove H_target = -dim(A) = -7  (entropia target = negativo dimensione azioni)
```

#### Step 4 — Soft Update Reti Target
```
Q₁_target ← τ · Q₁ + (1 - τ) · Q₁_target
Q₂_target ← τ · Q₂ + (1 - τ) · Q₂_target
con τ = 0.005
```

---

## Iperparametri SAC

| Parametro | Valore | Motivazione |
|-----------|--------|-------------|
| `learning_rate` | 3e-4 | Standard per SAC (Adam optimizer) |
| `buffer_size` | 1,000,000 | Buffer grande → maggiore diversità esperienze |
| `batch_size` | 256 | Trade-off efficienza/varianza gradient |
| `gamma` | 0.95 | Discount factor: bilancia reward immediato vs futuro |
| `tau` | 0.005 | Soft update lento → stabilità Q-function |
| `train_freq` | 1 | Aggiornamento ad ogni step environment |
| `gradient_steps` | 2 | 2 aggiornamenti rete per step env (data efficiency) |
| `learning_starts` | 10,000 | Raccolta dati casuali iniziale per riempire buffer |
| `ent_coef` | "auto" | Temperatura entropia appresa automaticamente |
| `policy_net_arch` | (512, 512) | Rete ampia per task complesso a 4 fasi |

---

## VecNormalize — Normalizzazione Osservazioni

```python
VecNormalize(env, norm_obs=True, norm_reward=True)
```

### Training
- `norm_obs=True`: normalizza obs con running mean/var (aggiornata ad ogni step)
- `norm_reward=True`: normalizza reward con running var (senza sottrarre media)
- Parametri salvati in `vecnormalize.pkl`

### Evaluation
```python
eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False)
eval_env.obs_rms = training_env.obs_rms  # sync statistiche
```
- Non normalizza reward per non distorcere la valutazione
- Usa le stesse statistiche obs dell'env di training

### Play Mode
```python
obs_norm = (obs - obs_rms.mean) / sqrt(obs_rms.var + 1e-8)
obs_norm = clip(obs_norm, -10, 10)
```
Normalizzazione manuale usando statistiche salvate in pkl.

---

## Flusso di Inferenza (Play Mode)

```
obs_raw (env) → VecNormalize.normalize(obs) → obs_norm
                                                    ↓
                                         Actor Network Forward Pass
                                         [obs_norm] → [512 ReLU] → [512 ReLU]
                                         → [μ(s), log_σ(s)]
                                                    ↓
                                    deterministic=True → a = tanh(μ(s))
                                    deterministic=False → a = tanh(μ + σ·ε)
                                                    ↓
                                         EMA Smoothing (α=0.8):
                                         a_final = 0.8·a + 0.2·a_prev
                                                    ↓
                                         FSM Override in env.step():
                                         - HOLD: a[:-1] = 0.0
                                         - RETREAT (arrivato): a = zeros
                                                    ↓
                                         Robosuite → MuJoCo step
```

---

## Convergenza Osservata

| Metrica | Valore al convergenza |
|---------|----------------------|
| `ep_rew_mean` | 1050–1140 |
| `rollout/success_rate` | 100.0% |
| `ent_coef` | ~0.00034 |
| `ep_len_mean` | ~122 step (training) / ~110 step (eval) |
| Steps totali training | ~1.82M |

### Interpretazione `ent_coef = 0.00034`
Il coefficiente di entropia è sceso a un valore estremamente basso.  
Questo indica:
1. **Policy deterministica**: la distribuzione π(·|s) è molto peakatta
2. **Convergenza Q-function**: le stime Q sono stabili e consistenti
3. **Fine esplorazione**: l'agente è uscito dalla fase di ricerca, ha una strategia consolidata
4. **Rischio catastrophic forgetting**: modificare il reward ora causerebbe disallineamento nei gradienti

### Bilancio Reward Appreso (log diagnostico Phase 3)
```
base: -0.50 | hold: +1.00 | hold_grip: +1.00 | hold_jnt_freeze: -0.03 
hold_act: +1.00 | hold_flat: -1.18 → TOT: +1.29
```
L'agente massimizza `hold_act` (braccio fermo) e `hold_grip` (gripper chiuso),  
pagando la penalità minima su `hold_flat` (allineamento piano del polso).
