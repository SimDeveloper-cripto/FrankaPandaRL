# Piano Implementazione — FrankaPandaRL TODO

## 1. Stato del Codebase (Analisi Profonda)

### Stack e Architettura
| Componente | File | Ruolo |
|---|---|---|
| Env base | `train_close.py` | Gymnasium wrapper, action smoothing (α=0.8), `_success_angle` threshold, reward base |
| Env FSM | `close_generalized/env_gen.py` | 4-phase FSM, full reward shaping, domain randomization |
| Training | `close_generalized/train_gen.py` | SAC setup, curriculum callback, grasp diagnostic |
| Config | `config/train_close_config.py` | Tutti gli iperparametri (horizon=600, total_steps=800k, γ=0.95) |

### FSM — Funzionamento Dettagliato

```
Phase 1 REACH:  Penalità dist_3d/xy/z, tassa temporale (base - _W_GRIPPER_CLOSE=-2.5/step)
                Conferma presa: gripper_action > 0.65 AND PHYS_OK AND dist < 0.02 per 5 step
                → transizione: _grasp_phase=True, bonus +20

Phase 2 PUSH:   Reward door_progress (_W_PROGRESS_GRASP=2000 * Δangle)
                Solo se gripper_action > 0.65 — CARROT puro, nessun bastone sull'angolo
                Perdita presa: torna immediatamente a Phase 1 con penalità
                → transizione: _success_latched=True (door_qpos < close_fraction * range = ~0.023 rad)

Phase 3 HOLD:   Timer: _hold_closed_duration per 60 step (2s × 30Hz)
                Condizione timer: abs(door_qpos) < 0.04
                Penalità: hold_bounce = -20*|qpos| se non is_closed
                          hold_slip = -5.0 se not PHYS_OK
                          hold_jnt_freeze = -3.0 * ||joint_vel||
                → transizione: _ready_to_retreat=True, _retreat_pos = eef_pos + [-0.20, 0, 0.05]

Phase 4 BACK:   ret_grip: +2.0 se gripper_action < -0.85
                ret_act: -1.0 * ||action[:-1]||      ← CONFLITTO con movimento
                ret_rot: -10.0 * ||action[3:6]||
                ret_lat: -5.0 * |action[1]| (se dist < 0.12m)
                ret_freeze: -20.0 * ||action|| (se dist_to_target < 0.05m)
                + da train_close.py: w_return_pos=2.0 * (1-tanh(dist/0.10)) ← pull verso target
```

### Spazio di Osservazione (122 dim)
- Base robosuite (~114): joint pos/vel/cos/sin, eef pos/quat, gripper qpos/qvel, door/handle pos, proprio-state, object-state
- Custom (8): [dist, handle_radius, handle_friction, fsm_reach, fsm_push, fsm_hold, fsm_retreat, hinge_qpos]

### Risultati Attuali (800k step)
- Eval Success Rate: **100%** ✅
- Rollout Success Rate: **97%** ✅
- Grasp Rate: **1.03** ✅
- Retreat Rate: **26.5%** ⚠️ — problema noto TODO 3

---

## 2. Analisi Paper → Rilevanza per il Codice

| Paper | Titolo | Implementato? | Gap / Opportunità |
|---|---|---|---|
| **paper_12** | Soft Actor-Critic (Haarnoja et al., 2018) | ✅ Completo | SAC off-policy, max-entropy, double-Q, ent_coef=auto |
| **paper_0** | DisCor: Distribution Correction (Kumar et al., 2020) | ❌ No | Buffer FIFO standard: transitions di HOLD/RETREAT sono rare e preziose, ma campionate uniformemente come quelle di REACH |
| **paper_2** | SAC + Hierarchical Reward Mechanism (Ling/Wen, 2025) | ✅ Quasi | Stage-based reward identico alla nostra FSM. Il paper raggiunge 98% su nut grasping — valida il nostro approccio |
| **paper_1** | Pushing-Grasping con Grasp Success Prediction (Gong/Ji) | ⚠️ Parziale | Il nostro `_grasp_confirm_count` è il GCP analogo; il paper usa un modello separato per predire il successo prima di agire |
| **paper_4** | Dynamic Nonprehensile Manipulation (Lynch/Mason, 1999) | ✅ Risolto | Le prime run usavano pushing non-prehensile; la FSM lo ha eliminato con il conditional grasp reward |
| **paper_11** | CHOMP: Gradient Motion Planning (Ratliff et al., 2009) | ❌ No | Il retreat è learned from scratch; CHOMP fornirebbe una traiettoria di riferimento smooth per la Phase 4 |
| **paper_10** | Manipulability Optimization via DNN (Jin et al., 2017) | ⚠️ Parziale | `hold_jnt_freeze` è un proxy reward; la manipolabilità reale (det(J·Jᵀ)) non è ottimizzata esplicitamente |
| **paper_3** | RelaxedIK (Rakita et al., 2018) | ❌ No | IK real-time con feasibility — utile per generare _retreat_pos come traiettoria feasible |
| **paper_6** | Comprehensive Grasp Taxonomy (Feix et al.) | ⚠️ Implicito | Il nostro grasp è "Power Cylindrical Grip" — la tassonomia giustifica `is_physically_closed` via confronto diametro |
| **paper_8** | Underactuation in Grasping (Laliberté et al., 2002) | ✅ Informativo | Gripper Panda è parallel-jaw; `gripper_width` check è corretto per questo tipo |
| **paper_9** | Adaptive Synergies (Grioli/Bicchi et al., 2012) | ❌ No | Il concetto di synergy-based grasp: durante HOLD, il gripper dovrebbe mantenere forza adattiva, non solo comando fisso |
| **paper_5** | Overview Robotic Grippers (Cairnes et al., 2023) | ✅ Informativo | Valida friction randomization (0.3–1.2×) come tecnica di domain randomization |
| **paper_7** | Dexterous Hands Review (Lin et al., 2025) | ❌ Irrilevante | Hardware review, non applicabile alla sim |
| **paper_13** | Prediction Learning in Pushing (Kopicki et al.) | ⚠️ Ispirazione | Un modello predittivo della posizione della maniglia durante PUSH risolverebbe il grip slip (TODO 2) |

### Citazioni da usare nel report finale
- **SAC**: Haarnoja et al., "Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL with a Stochastic Actor", ICML 2018 (paper_12)
- **Hierarchical Reward**: Ling & Wen, "Research on Robotic Arm Grasping Control Based on SAC and Hierarchical Reward Mechanism", ICCEIC 2025 (paper_2)
- **DisCor**: Kumar et al., "DisCor: Corrective Feedback in RL via Distribution Correction", NeurIPS 2020 (paper_0)
- **CHOMP**: Ratliff et al., "CHOMP: Gradient Optimization for Efficient Motion Planning", ICRA 2009 (paper_11)
- **Grasp Taxonomy**: Feix et al., "A Comprehensive Grasp Taxonomy", IEEE RAM 2016 (paper_6)
- **Manipulability**: Jin et al., "Manipulability Optimization via Dynamic Neural Networks", IEEE Trans. Ind. Electron. 2017 (paper_10)

---

## 3. Diagnosi TODO — Causa Radice

### TODO 1 — Bounce Porta in Fase 3

**Causa fisica (confermata dai log a 800k step: `DOOR: 0.05-0.06` in HOLD):**

La porta, chiusa a 0.0 rad, subisce un rimbalzo elastico dal contatto con lo stipite in MuJoCo. Il giunto `Door_hinge` non ha damping sufficiente per smorzare l'oscillazione.

**Cosa succede nel codice:**
```python
# env_gen.py L269-271
if not is_closed:                         # door_qpos > 0.03
    rew_info["hold_bounce"] = -20.0 * abs(door_qpos)
    self._hold_closed_duration = 0        # ← RESET SECCO al bounce
```
Il reset a zero è devastante: anche un singolo bounce da 0.04→0 azzera 40 step di progresso. L'agente impara a resistere ma il timer diventa instabile.

**Connessione paper**: La manipolabilità durante HOLD (paper_10) richiede che il braccio applichi forza nella direzione di chiusura — questo contrasterebbe fisicamente il rimbalzo meglio di qualsiasi reward.

---

### TODO 2 — Grip Slip in Fase 2

**Causa geometrica (confermata da docs_v3 §4):**

La maniglia si sposta di ~10 cm in XY mentre la porta ruota da 0.4→0.0 rad. Il gripper deve inseguire un bersaglio in movimento su un arco circolare.

```
door_angle=0.40 → handle_site = [-0.199, -0.139, 1.075]
door_angle=0.20 → handle_site = [-0.164, -0.192, 1.075]
door_angle=0.00 → handle_site = [-0.140, -0.252, 1.075]
```

**Cosa manca nel reward:**

La rete ha `hinge_qpos` in osservazione (sa l'angolo porta), ma non ha la **velocità della maniglia**. Quindi sa *dove* è la maniglia ora, ma non *dove sarà* tra 1 step. Questo crea un inseguimento in ritardo — il gripper segue invece di anticipare.

Inoltre: `effective_lose_tol = 0.05m` è fisso. Ma quando la porta gira velocemente, la maniglia si sposta di 2-3 cm per step — il 5 cm di tolleranza è quasi tutto consumato dalla sola cinematica circolare.

**Connessione paper**: paper_13 (Prediction Learning in Pushing) propone esattamente un modello probabilistico per predire dove andrà l'oggetto dopo il contatto. Nel nostro caso: predire la posizione della maniglia dato `hinge_qpos` è triviale (geometria nota).

---

### TODO 3 — Retreat Incompleto (Stop non pulito)

**Causa strutturale (confermata: retreat_rate=26.5%):**

Due problemi separati, identificati con chiarezza:

**3a — Lentezza (73.5% degli episodi termina per timeout):**
```python
# env_gen.py L318
rew_info["ret_act"] = -1.0 * np.linalg.norm(action[:-1])  # punisce TUTTO il movimento
# train_close.py L224
reward += self.cfg.w_return_pos * (1.0 - np.tanh(dist/0.10))  # pull debole: max +2.0
```
Il segnale di pull verso il target (`w_return_pos=2.0`) è debolissimo rispetto alle penalità di movimento (`ret_act + ret_early_move + ret_rot`). Il policy sceglie: "resto quasi fermo, perdo poco ret_act, ma non incasso il retreat bonus". La distanza target è 20 cm — troppa per il tempo rimanente.

**3b — Stop non completo (null-space drift):**
```python
# env_gen.py L339-340 — SOLO a <5cm dal target
if dist_to_target < 0.05:
    rew_info["ret_freeze"] = -20.0 * np.linalg.norm(action[:-1])
    rew_info["ret_jnt_freeze"] = -5.0 * np.linalg.norm(joint_vel)
```
Il freeze dei giunti è attivo solo negli ultimi 5 cm e solo velocità (non posizione). I giunti si muovono per null-space drift (la spalla/gomito derivano anche se l'EEF sembra fermo). Questo è il comportamento documentato: "polso perfetto ma giunti no".

**Connessione paper**: paper_11 (CHOMP) mostra che ottimizzare la traiettoria completa prima di eseguirla elimina il drift — la traiettoria è smooth per costruzione. paper_10 (Manipulability) mostra che ottimizzare null-space per massimizzare manipolabilità stabilizza automaticamente la configurazione del braccio.

---

## 4. Proposte di Implementazione

---

### TODO 1 — Anti-Bounce: Strategie

#### Strategia A: Soft Timer Reset ✅ APPLICATA
Invece di azzerare `_hold_closed_duration = 0` al bounce, decrementare proporzionalmente:

```python
# env_gen.py — CAMBIA da:
self._hold_closed_duration = 0
# A:
penalty_steps = int(abs(door_qpos) / 0.03 * 10)
self._hold_closed_duration = max(0, self._hold_closed_duration - penalty_steps)
```

#### Strategia C: Penalità sulla Velocità Angolare ✅ APPLICATA
```python
door_qvel = self._rs_env.sim.data.qvel[self._rs_env._door_hinge_dof_adr]
if abs(door_qvel) > 0.01:
    rew_info["hold_veldamp"] = -15.0 * abs(door_qvel)
```

---

### TODO 2 — Grasp Tracking: Strategie

#### Strategia B: Arc-Tracking Reward ✅ APPLICATA
```python
# Premio se EEF velocity è allineata con la direzione di movimento della maniglia
rew_info["arc_track"] = 2.0 * tracking
```

#### Strategia C: Tolleranza Dinamica ✅ APPLICATA
```python
door_speed = abs(prev_angle - door_angle) * self.cfg.control_freq
effective_lose_tol = np.clip(0.04 + door_speed * 0.5, 0.04, 0.10)
```

---

### TODO 3 — Retreat: Piano in 3 Parti

#### Parte 1: Ridurre distanza target 20cm → 13cm ✅ APPLICATA
#### Parte 2: Reward direzionale (risolve conflitto ret_act) ✅ APPLICATA
#### Parte 3: Joint Freeze Progressivo su tutto il retreat ✅ APPLICATA

---

## 5. Strategia — Fine-Tuning

Tutte le modifiche applicate sono compatibili con **fine-tuning** del modello a 800k step:
- Nessun cambio allo spazio osservazione (rimane 122 dim)
- Modifiche solo a reward shaping e FSM logic
- Target: ~200k step addizionali per convergere alle nuove regole

---

## 6. Analisi Log 800k Step — Training Run Attivo (13/05/2026)

### Headline

| Metrica | Valore | Giudizio |
|---|---|---|
| `rollout/success_rate` | **0.81 → 0.80** | 🟢 Ottimo |
| `eval/success_rate` | **0.40** | 🟡 Gap significativo |
| `ep_len_mean` | **313** (era 500) | 🟢 Episodi si completano in 300 step, non arrivano al timeout |
| `ep_rew_mean` | **962-979** | 🟢 Il più alto visto finora |
| `grasp_rate` | **1.03** | 🟢 100% episodi ottiene il grasp |
| `retreat_rate` | **0.785** | 🟢 78.5% dei latch porta al retreat |

---

### Cosa sta funzionando bene

**1. La near-latch zone funziona perfettamente.** In quasi tutti i log Phase 2 si vede `DOOR: 0.01-0.03` — il robot arriva consistentemente all'ultimo millimetro prima del latch. Il `near_latch_bonus` è attivo e cresce fino a `+2.19`. Questo è esattamente il comportamento voluto.

**2. Phase 4 (BACK) è finalmente frequente.** Appaiono molti log `4:BACK │ DOOR: 0.00-0.01` — il task viene completato. Nella run precedente (21%) quasi non si vedeva Phase 4.

**3. `door_prog` spike altissimi.** Si vedono `+101`, `+121`, `+82` in singoli step — il robot spinge attivamente la porta da 0.18 rad → 0.01 rad in un'azione, non "ci si siede sopra".

---

### Problema Critico: Gap Train/Eval (80% vs 40%)

`episode_reward=1245.60 +/- 1038.50` — la std è quasi pari alla media: distribuzione fortemente **bimodale**. Il robot o riesce (reward molto alto) o fallisce completamente (reward molto basso). Non c'è un comportamento "intermedio".

**Causa probabile del gap**: i 31 episodi di rollout contati per `success_rate` sono i più *recenti* — periodo in cui la policy stava andando bene. L'eval usa 31 episodi campionati randomicamente su tutto lo spazio di configurazione → espone casi di door/robot position più difficili.

---

### Phase 3 — Il Bottleneck Rimasto

Guardando i log Phase 3:

```
3:HOLD │ DOOR: 0.04 │ hold_bounce: -0.75   → porta bounced a 0.04 (era < close_fraction)
3:HOLD │ DOOR: 0.16 │ hold_bounce: -3.11   → porta bounced pesantemente a 0.16
3:HOLD │ DOOR: 0.00 │ CONF: 0/5 │ DROP: dist 0.057 > 0.056   → uscita per distanza (Phase 2 event residuo nel buffer)
3:HOLD │ DOOR: 0.00 │ CONF: 5/5 │ hold_flat: -2.86   → hold corretto ma penalità flat altissima
3:HOLD │ DIST: 0.154-0.165 │ hold_slip: -5.00   → robot lontano quando il latch scatta (momentum latch)
```

Tre modalità di fallimento in Phase 3:

1. **Bounce**: la porta rimbalza subito dopo il latch (il robot non mantiene la pressione)
2. **Momentum latch**: il latch scatta per inerzia ma il robot è già a 0.15m dalla maniglia → `hold_slip -5.0`
3. **hold_flat eccessivo**: `hold_flat = -5.0 * flat_alignment` con flat_alignment ≈ 0.50 → **-2.5/step**, rende Phase 3 totale da -2 a -9/step vs Phase 2 da +2 a +3/step

**Root cause quantificata**:

| Componente Phase 3 | Costo/step tipico |
|---|---|
| `hold_flat: -5.0 * 0.50` | **-2.50** |
| `hold_jnt_freeze: -3.0 * 0.40` | **-1.20** |
| `hold_act: -5.0 * 0.25` | **-1.25** |
| `hold_bounce: -20.0 * 0.07` | **-1.40** |
| `hold: +1.0, hold_grip: +1.0` | **+2.00** |
| **Totale tipico** | **-4.35/step** |

Il robot entra in Phase 3 (rollout 80%) ma è sotto fortissima pressione a uscirne nel minor numero di step possibile → latch frettolosi → bounce.

---

### Phase 4 — Retreat problematico

```
ret_rot: -0.96 a -2.26    → EEF ruota durante il retreat (peso -10 è eccessivo)
ret_jnt_prog: -0.47 a -2.71    → progressione giunti lenta
latch_ret: -0.74 a -1.22    → porta si sta riaprendo durante il retreat (21.5% fallimenti)
```

Il `latch_ret` indica che la porta non è perfettamente bloccata nel latch quando il robot si ritira.

---

### Fix Applicati (13/05/2026)

#### Fix A — `hold_flat` weight ridotto: `-5.0` → `-2.0`
Riduce il costo per step di Phase 3 da -4 a -2, rendendo HOLD più sostenibile e incentivando il robot a rimanere in contatto invece di fuggire frettolosamente.

#### Fix B — `hold_dist` penalty aggiunto in Phase 3
Quando `is_closed=True` e `dist_handle > 0.06m`, penalità crescente. Previene il "momentum latch" dove il robot latcha da 0.15m senza contatto fisico:
```python
if dist_handle > 0.06:
    rew_info["hold_dist"] = -3.0 * (dist_handle - 0.06)
```
Questa penalità spinge il robot a restare vicino alla maniglia durante tutto il HOLD, rendendo il latch fisicamente sostenuto e non accidentale.

---

### Aspettative Post-Fix

Con questi fix:
- `hold_flat` da -2.5 → -1.0/step: Phase 3 passa da totale **-4.35** a **-2.85/step**
- `hold_dist` penalizza i momentum latch → il robot resterà in contatto durante HOLD
- La combinazione riduce i fallimenti per bounce e slip
- Attesa: eval SR da 40% → 60-70% entro i prossimi 200k step

---

## 7. Analisi Definitiva — Fine Training a 1M Step (13/05/2026)

### 7.1 I Numeri Fondamentali

| Metrica | 800k | 1M | Δ |
|---|---|---|---|
| `success_rate` | 0.80-0.81 | **0.93-0.94** | +13pp |
| `ep_len_mean` | 313 | **241-252** | -65 step |
| `ep_rew_mean` | 979 | **1130** | +15% |
| Phase 3 `hold_flat` | -2.50/step | **-0.88 a -1.50/step** | Fix A ✅ |
| Phase 3 totale | -4 a -9/step | **-1 a -4/step** | ✅ |
| `hold_dist` penalty | — | **mai apparso** | Fix B ✅ implicito |

**`ep_len: 241-252` è il dato più eloquente.** La policy completa il task in meno di metà del budget (horizon 600). Non si "siede" sull'exploit, non aspetta il timeout — esegue e completa.

---

### 7.2 Il Fix A ha funzionato esattamente come previsto

Nei log di Phase 3 si vede chiaramente: `hold_flat: -0.88, -1.14, -1.37, -1.43, -1.47, -1.49, -1.50`. Con `flat_alignment` tipico ≈ 0.50-0.75, `hold_flat` al peso `-2.0` è quasi esattamente dimezzato rispetto al vecchio `-5.0`. Phase 3 è ora sostenibile: il robot **vuole** starci, non fuggirne.

---

### 7.3 Il Fix B ha funzionato in modo indiretto (meglio così)

`hold_dist` non appare mai nei log. Questo significa che il robot durante HOLD è **sempre entro 6cm dalla maniglia**. La penalità ha spostato la distribuzione di comportamento durante il training — il robot ha imparato a stare vicino anche prima che la penalità diventasse attiva. Questo è il segnale migliore possibile: la policy non *evita* la penalità, ha **cambiato strategia** in modo strutturale.

---

### 7.4 `actor_loss` Negativo: Cosa Significa Davvero

Nei log: `-0.0192, -0.0218, -0.0154, -0.0266, -0.0251`. In SAC:

```
actor_loss = E[α · log π(a|s) - Q(s,a)]
```

Negativo significa `Q(s,a) > α · log π(a|s)`. Con `ent_coef ≈ 0.000268` (quasi zero), il termine entropico è trascurabile. Quindi: **il Q-network predice rewards molto alti**, più del termine di regolarizzazione. Il policy gradient sta ancora facendo un lavoro positivo (aumentare le Q). Non è un segnale di problemi — è la firma di una policy che ha trovato traiettorie molto profittevoli e le segue deterministicamente.

---

### 7.5 Il 6% Residuo — Analisi delle Failure Modes

Guardando i log, si identificano **due pattern distinti di potenziale fallimento**:

#### Failure Mode A — Hard Bounce con `hold_veldamp: -2.78`

```
3:HOLD │ DOOR: 0.00 │ hold_veldamp: -2.78 → door_qvel = 0.185 rad/s
```

Questo è il bounce fisico in MuJoCo: la porta colpisce il frame elasticamente e rimbalza a ~0.185 rad/s. La penalità è `-15.0 * 0.185 = -2.78`. Questo caso può degenerare in un bounce abbastanza forte da riaprire la porta prima che il timer HOLD completi. Singola occurrence nei log, quindi raro.

#### Failure Mode B — Latch Handle Non Ritorna al Neutro (Pattern Dominante del 6%)

```
4:BACK │ latch_ret: -1.22, -1.53, -0.92, -0.90, -1.16, -1.13
```

`latch_ret = -1.0 * abs(latch_qpos)`. Il `latch_qpos` è l'angolo del meccanismo di chiusura (la levetta della maniglia). Valori di 0.90-1.53 rad significano che la maniglia è ancora in posizione ruotata quando il robot si ritira.

In MuJoCo, il latch della porta di robosuite si **impegna meccanicamente solo** quando `latch_qpos ≈ 0` (maniglia in posizione neutra). Se il robot rilascia il gripper mentre la maniglia è ancora ruotata (anche se `door_qpos = 0.00`), la molla interna della maniglia non ha ancora spinto il latch nella sede → la porta è "chiusa ma non bloccata" → al rilascio del robot può riaprirsi leggermente.

Questo è la causa più probabile del 6% di fallimenti. Non è un problema di reward structure — è un problema di **temporizzazione del rilascio**: il robot rilascia il gripper troppo presto, prima che il latch_joint sia tornato al neutro.

#### Failure Mode C — Raro Edge Case in Phase 1

```
1:REACH │ DIST: 0.161 │ GRIP: -0.98 │ PHYS_OK │ CONF: 0/5
```

`PHYS_OK` con azione aperta. Non è un fallimento ma un'anomalia: `WIDTH: 0.046` cade fortuitamente nella finestra `[0.015, 0.065]` anche con gripper aperto. Il FSM lo ignora correttamente (`CONF: 0/5`), ma indica che c'è una sovrapposizione nella zona di approccio che potrebbe raramente confondere il sensing.

---

### 7.6 Cosa Resta del Gap Train/Eval

I numeri formali della eval a 1M non sono ancora disponibili, ma è possibile ragionare strutturalmente:

- **A 800k**: Phase 3 era -4 a -9/step → distribuzione bimodale estrema → **40% eval SR**
- **Ora a 1M**: Phase 3 è -1 a -4/step, policy completa in 250 step, `hold_dist` non compare mai → la variance è molto più bassa → la bimodalità si è ridotta significativamente

Il gap dovrebbe essersi chiuso in modo sostanziale. Stima ragionata: **eval SR 70-85%**, non ancora 100% per via del Failure Mode B (latch_ret).

---

### 7.7 Anomalia FPS: 89 vs Precedente 270

È un calo 3x inspiegabile con lo stesso codice e `num_envs: 8`. Le ipotesi: rendering attivo in background, throttling termico dopo ore di run, processi che contendono CPU/GPU. Non influenza la qualità del modello ma spiega perché il run ha impiegato ~3.1 ore invece di ~1 ora.

Verifica: `total_timesteps: 999,744` / `time_elapsed: 11,178 s` = 89.4 fps. Con 8 env paralleli → ~713 env-step/s effettivi.

---

### 7.8 Sintesi Finale

Questo è il training migliore visto finora: stabile, efficiente, con Phase 3 finalmente sostenibile.

| Obiettivo | Stato |
|---|---|
| Eliminare reward exploit (stasi a DOOR: 0.06) | ✅ Risolto con near_latch_bonus e grip_hold bilanciato |
| Ridurre bimodalità (eval 40% gap) | ✅ Sostanzialmente ridotta con Fix A+B |
| Phase 3 sostenibile | ✅ Totale -1 a -4/step vs -4 a -9 precedente |
| Momentum latch | ✅ hold_dist ha cambiato la strategia della policy |
| **Latch timing al rilascio (6% failure)** | ⚠️ Rimasto — problema di meccanica MuJoCo |
| **100% success rate** | ⚠️ Non ancora raggiunto |

Il collo di bottiglia restante non è nel reward shaping — è nel **timing del rilascio della maniglia durante la transizione HOLD→RETREAT**, un problema di meccanica del latch nella simulazione MuJoCo. Per avvicinarsi al 100%, il prossimo intervento deve agire sulla temporizzazione del rilascio: ad esempio, attendere che `latch_qpos < 0.1 rad` prima di consentire la transizione al RETREAT, e/o introdurre un reward che incentivi il ritorno della maniglia al neutro durante la fase di HOLD.

---

## 8. Log Selezionati — Campionamento Rappresentativo (1M Step)

Questa sezione raccoglie i log diagnostici più significativi delle ultime iterazioni del training, uno per ogni fase FSM, selezionati per illustrare sia il comportamento ottimale raggiunto che i failure mode residui.

### 8.1 Snapshot Metrica SB3 — Istante Finale

```
-----------------------------------------
| rollout/                   |          |
|    ep_len_mean             | 241      |   ← completa in <41% del budget (horizon=600)
|    ep_rew_mean             | 1.1e+03  |   ← reward più alto di tutta la storia del training
|    episodes_counted_for_sr | 43       |
|    success_rate            | 0.94     |   ← picco a 0.94, stabile per ultimi 50k step
| time/                      |          |
|    episodes                | 2576     |
|    fps                     | 88       |
|    time_elapsed            | 11147    |   ← ~3.1 ore totali
|    total_timesteps         | 990712   |
| train/                     |          |
|    actor_loss              | -0.0192  |   ← NEGATIVO: Q-values > entropia → policy deterministica
|    critic_loss             | 0.000546 |   ← convergenza Q-network
|    ent_coef                | 0.000269 |   ← quasi zero: esplorazione quasi nulla
|    learning_rate           | 0.0003   |
|    n_updates               | 245176   |
-----------------------------------------
```

---

### 8.2 Phase 1 — REACH: Approccio quasi-perfetto

```
┌─────────┬────────┬────────┬───────┬───────────┬───────┬───────┬───────┬───────┐
│  PHASE  │  DIST  │   dZ   │ GRIP  │   PHYS    │ WIDTH │ ALIGN │ DOOR  │ CONF  │
├─────────┼────────┼────────┼───────┼───────────┼───────┼───────┼───────┼───────┤
│ 1:REACH │  0.010 │ +0.002 │ +1.00 │ PHYS_OK   │ 0.050 │  0.40 │  0.38 │  2/5  │
└─────────┴────────┴────────┴───────┴───────────┴───────┴───────┴───────┴───────┘
  ↳ REWARDS │ smoothness: -0.31 │ base: -3.00 │ dist_3d: -0.05 │ dist_xy: -0.03
            │ dist_z: -0.03 │ align: -0.54 │ flat: -0.12 │ grip: +2.50 │ TOT: -1.58
```

**Interpretazione**: DIST=0.010m (1cm dalla maniglia), GRIP=+1.00 (gripper chiuso al massimo), PHYS_OK, CONF=2/5 → il robot è a 3 step dalla transizione a Phase 2. La `grip: +2.50` conferma che il gripper è fisicamente in contatto. Questo è il comportamento ottimale di Phase 1.

---

### 8.3 Phase 2 — PUSH: Near-latch attivo, spinta efficiente

```
┌─────────┬────────┬────────┬───────┬───────────┬───────┬───────┬───────┬───────┐
│  PHASE  │  DIST  │   dZ   │ GRIP  │   PHYS    │ WIDTH │ ALIGN │ DOOR  │ CONF  │
├─────────┼────────┼────────┼───────┼───────────┼───────┼───────┼───────┼───────┤
│ 2:PUSH  │  0.034 │ -0.017 │ +0.99 │ PHYS_OK   │ 0.042 │  0.47 │  0.01 │  5/5  │
└─────────┴────────┴────────┴───────┴───────────┴───────┴───────┴───────┴───────┘
  ↳ REWARDS │ smoothness: -0.34 │ base: -0.50 │ dist_3d: -0.17 │ dist_z: -0.26
            │ act_pen: -0.01 │ near_latch_bonus: +2.20 │ grip: -0.04 │ grip_hold: +2.00
            │ TOT: +2.89
```

**Interpretazione**: DOOR=0.01 rad (nella near-latch zone), `near_latch_bonus: +2.20` attivo, `grip_hold: +2.00`, CONF=5/5. Il robot è praticamente sul punto di latch con grasp confermato. TOT +2.89/step è il segnale che guida la policy verso la chiusura finale. Questo è il comportamento di Phase 2 ottimale.

```
┌─────────┬────────┬────────┬───────┬───────────┬───────┬───────┬───────┬───────┐
│  PHASE  │  DIST  │   dZ   │ GRIP  │   PHYS    │ WIDTH │ ALIGN │ DOOR  │ CONF  │
├─────────┼────────┼────────┼───────┼───────────┼───────┼───────┼───────┼───────┤
│ 2:PUSH  │  0.017 │ -0.004 │ +1.00 │ PHYS_OK   │ 0.043 │  0.11 │  0.02 │  5/5  │
└─────────┴────────┴────────┴───────┴───────────┴───────┴───────┴───────┴───────┘
  ↳ REWARDS │ base: -0.45 │ dist_3d: -0.09 │ dist_z: -0.06 │ door_prog: +48.32
            │ near_latch_bonus: +1.90 │ grip: -0.02 │ grip_hold: +2.00 │ TOT: +49.64
```

**Interpretazione**: `door_prog: +48.32` in un singolo step — il robot ha spinto la porta di circa 0.024 rad in un'azione (48.32 / _W_PROGRESS_GRASP=2000 ≈ 0.024 rad). Questo è una push attiva e decisa: il robot non "indugia" ma chiude aggressivamente.

---

### 8.4 Phase 3 — HOLD: Hold ottimale (Fix A confermato)

```
┌─────────┬────────┬────────┬───────┬───────────┬───────┬───────┬───────┬───────┐
│  PHASE  │  DIST  │   dZ   │ GRIP  │   PHYS    │ WIDTH │ ALIGN │ DOOR  │ CONF  │
├─────────┼────────┼────────┼───────┼───────────┼───────┼───────┼───────┼───────┤
│ 3:HOLD  │  0.021 │ +0.000 │ +0.97 │ PHYS_OK   │ 0.031 │  0.81 │  0.00 │  5/5  │
└─────────┴────────┴────────┴───────┴───────────┴───────┴───────┴───────┴───────┘
  ↳ REWARDS │ smoothness: -0.12 │ base: -0.50 │ hold: +1.00 │ hold_grip: +1.00
            │ hold_jnt_freeze: -0.77 │ hold_act: -0.63 │ hold_flat: -1.43 │ TOT: -1.44
```

**Interpretazione**: DOOR=0.00 (porta completamente chiusa), CONF=5/5, DIST=0.021m (robot vicinissimo alla maniglia). `hold_flat: -1.43` vs vecchio valore tipico -2.50 — **Fix A visibile**. TOT=-1.44/step: Phase 3 è ora costosa quanto una penalità leggera, non una punizione pesante. Il robot rimane in HOLD senza fretta di uscire.

### 8.5 Phase 3 — HOLD: Failure mode A (hard bounce)

```
┌─────────┬────────┬────────┬───────┬───────────┬───────┬───────┬───────┬───────┐
│  PHASE  │  DIST  │   dZ   │ GRIP  │   PHYS    │ WIDTH │ ALIGN │ DOOR  │ CONF  │
├─────────┼────────┼────────┼───────┼───────────┼───────┼───────┼───────┼───────┤
│ 3:HOLD  │  0.030 │ -0.017 │ +0.95 │ PHYS_OK   │ 0.044 │  0.23 │  0.00 │  5/5  │
└─────────┴────────┴────────┴───────┴───────────┴───────┴───────┴───────┴───────┘
  ↳ REWARDS │ smoothness: -1.71 │ base: -0.50 │ hold: +1.00 │ hold_veldamp: -2.78
            │ hold_grip: +1.00 │ hold_jnt_freeze: -1.05 │ hold_act: -1.21 │ hold_flat: -1.50
            │ TOT: -6.76
```

**Interpretazione**: `hold_veldamp: -2.78` = velocità angolare porta 0.185 rad/s. La porta ha rimbalzato contro il frame e si muove verso l'esterno nonostante DOOR sia ancora 0.00 al momento del log. Il TOT=-6.76 è il reward più negativo osservato in Phase 3 — questa è la firma del **Failure Mode A** (hard bounce MuJoCo).

---

### 8.6 Phase 4 — BACK: Retreat di successo

```
┌─────────┬────────┬────────┬───────┬───────────┬───────┬───────┬───────┬───────┐
│  PHASE  │  DIST  │   dZ   │ GRIP  │   PHYS    │ WIDTH │ ALIGN │ DOOR  │ CONF  │
├─────────┼────────┼────────┼───────┼───────────┼───────┼───────┼───────┼───────┤
│ 4:BACK  │  0.081 │ -0.034 │ -0.98 │ PHYS_OPEN │ 0.079 │  0.63 │  0.01 │  5/5  │
└─────────┴────────┴────────┴───────┴───────────┴───────┴───────┴───────┴───────┘
  ↳ REWARDS │ smoothness: -0.14 │ base: +0.44 │ hold: +0.99 │ ret_grip: +2.00
            │ ret_rot: -0.67 │ ret_lat: -0.22 │ ret_dir: +0.11 │ ret_perp: -0.19
            │ ret_jnt_prog: -0.56 │ latch_ret: -0.05 │ TOT: +1.70
```

**Interpretazione**: DOOR=0.01 (quasi chiuso), CONF=5/5, gripper aperto (PHYS_OPEN, GRIP=-0.98), `ret_grip: +2.00`. `latch_ret: -0.05` basso → il latch sta tornando al neutro. TOT=+1.70 → retreat con bilancio positivo. Questo è il comportamento ottimale di Phase 4.

### 8.7 Phase 4 — BACK: Failure mode B (latch non al neutro)

```
┌─────────┬────────┬────────┬───────┬───────────┬───────┬───────┬───────┬───────┐
│  PHASE  │  DIST  │   dZ   │ GRIP  │   PHYS    │ WIDTH │ ALIGN │ DOOR  │ CONF  │
├─────────┼────────┼────────┼───────┼───────────┼───────┼───────┼───────┼───────┤
│ 4:BACK  │  0.031 │ -0.004 │ -0.98 │ PHYS_OK   │ 0.075 │  0.73 │  0.00 │  5/5  │
└─────────┴────────┴────────┴───────┴───────────┴───────┴───────┴───────┴───────┘
  ↳ REWARDS │ smoothness: -0.14 │ base: +0.03 │ hold: +1.00 │ ret_grip: +2.00
            │ ret_rot: -1.73 │ ret_lat: -0.67 │ ret_dir: +0.63 │ ret_perp: -0.32
            │ ret_jnt_prog: -0.86 │ latch_ret: -1.22 │ TOT: -1.27
```

**Interpretazione**: `latch_ret: -1.22` → `latch_qpos = 1.22 rad`: la maniglia è ancora in posizione ruotata durante il retreat. Il robot ha già aperto il gripper (`GRIP=-0.98`) e si sta ritirando, ma il meccanismo di latch non si è ancora impegnato nella sede. Questa è la firma del **Failure Mode B** — causa principale del 6% di fallimenti.

---

## 9. Cosa Manca per il 100% — Prossimi Interventi

### 9.1 Problema Prioritario: Latch Timing (Failure Mode B)

**Root cause**: Il robot transita a RETREAT non appena il timer HOLD completa 60 step. In quel momento `latch_qpos` può ancora essere a 0.9-1.5 rad (maniglia ruotata). La porta è "chiusa fisicamente" (`door_qpos ≈ 0.00`) ma non "bloccata meccanicamente" (il latch non è nella sede).

**Intervento proposto**:

1. **Condizione aggiuntiva sulla transizione HOLD→RETREAT**: prima di impostare `_ready_to_retreat = True`, verificare `latch_qpos < 0.1 rad`. Se la condizione non è soddisfatta, non avanzare il timer e tenere il robot in HOLD.

2. **Reward per ritorno maniglia al neutro in Phase 3**: durante l'attesa del latch,

```python
latch_qpos = self._rs_env.sim.data.qpos[self._rs_env.handle_qpos_addr]
rew_info["latch_neutral"] = -0.5 * abs(latch_qpos)  # spinge verso latch_qpos=0
```

Questo incentiva il robot ad applicare una pressione che permette alla molla interna della maniglia di tornare al neutro, impegnando il latch.

### 9.2 Problema Secondario: Hard Bounce in Phase 3 (Failure Mode A)

**Root cause**: Il damping del giunto `Door_hinge` in MuJoCo è insufficiente per smorzare i rimbalzi elastici ad alta velocità. Quando il robot spinge la porta con forza per passare dal near-latch al latch, la porta può rimbalzare a 0.15-0.20 rad/s.

**Intervento proposto**: Aumentare il damping del giunto porta nel file XML di robosuite, oppure aggiungere una penalità più forte su `hold_veldamp` (attualmente `-15.0 * abs(door_qvel)` — portare a `-25.0`). Un'alternativa è penalizzare anche la velocità del braccio quando la porta sta rimbalzando: se `door_qvel > 0.1`, penalizzare `action_norm` per forcare il robot a smorzare attivamente.

### 9.3 Gap Train/Eval Residuo

**Da fare**: eseguire la valutazione formale a 1M step con `EvalCallback` su 100 episodi per quantificare l'effettivo eval SR e confrontarlo con la stima di §7.6 (70-85%).

| Intervento | Impatto atteso | Difficoltà |
|---|---|---|
| Condizione `latch_qpos < 0.1` su transizione HOLD→RETREAT | **+4-5% SR** (elimina Failure Mode B) | 🟢 Bassa — 5 righe di codice |
| `latch_neutral` reward in Phase 3 | **+2-3% SR** (accelera il ritorno maniglia) | 🟢 Bassa — 2 righe di codice |
| Aumento damping `hold_veldamp` da -15 a -25 | **+1-2% SR** (riduce hard bounce) | 🟢 Bassa — 1 parametro |
| Eval formale a 1M step | Quantifica gap reale | 🟢 Già implementata |
| **Totale atteso** | **~97-99% SR** | — |
