# 📚 Storico Completo — FrankaPandaRL / close_generalized

> Documento integrale che unisce tutte le documentazioni, test, risultati e analisi  
> prodotte durante lo sviluppo del task di chiusura porta generalizzato.

---

## 1. Contesto e Obiettivo del Progetto

### Task
Addestrare un robot **Franka Panda** (simulato in MuJoCo tramite Robosuite) a:
1. **Raggiungere** la maniglia di una porta aperta
2. **Afferrare** saldamente la maniglia
3. **Spingere** la porta fino alla posizione chiusa
4. **Mantenere** la porta chiusa per 2 secondi (permettendo al latch di scattare)
5. **Ritirarsi** dalla maniglia verso una posizione sicura

Il tutto in modo **generalizzato**: il robot deve funzionare con maniglie di dimensioni e frizione diverse,  
e con la porta in posizioni e orientazioni variabili.

### Stack Tecnologico
- **Simulatore:** MuJoCo 2.3+ via Robosuite
- **Robot:** Franka Panda (7 DOF + gripper)
- **Controller:** BASIC (operational space control)
- **Algoritmo RL:** SAC (Soft Actor-Critic) via Stable-Baselines3
- **Linguaggio:** Python 3.10

---

## 2. Architettura del Sistema (Sintesi)

### Gerarchia Classi
```
gymnasium.Env
└── RoboSuiteDoorCloseGymnasiumEnv  [train_close.py]
    └── GeneralizedDoorEnv          [close_generalized/env_gen.py]
```

### Files Principali
| File | Ruolo |
|------|-------|
| `config/train_close_config.py` | Configurazione (iperparametri, env params) |
| `train_close.py` | Classe base env + training deterministico |
| `close_generalized/env_gen.py` | Env generalizzato (FSM + reward + randomizzazione) |
| `close_generalized/train_gen.py` | Training generalizzato + curriculum + callbacks |
| `close_generalized/diag_phase34.py` | Diagnostica fasi 3-4 |
| `eval_stats.py` | Valutazione finale + plots |

---

## 3. FSM — Macchina a Stati Finiti (Sintesi)

### 4 Fasi del Task

| Fase | Nome | Condizione Ingresso | Condizione Uscita |
|------|------|--------------------|--------------------|
| 1 | REACH | Reset | `grasp_confirm_count >= 5` |
| 2 | PUSH | Grasp confermato | `door_angle <= 1.5% range AND grip > 0.80` |
| 3 | HOLD | Door chiusa con grip saldo | `hold_closed_duration >= 60 step (2.0s)` |
| 4 | RETREAT | Hold timer scaduto | `dist_to_retreat < 5cm` → TERMINATED |

### Variabili Chiave FSM
- `_grasp_phase` (bool): PUSH attiva
- `_success_latched` (bool): HOLD/RETREAT attiva
- `_ready_to_retreat` (bool): RETREAT attiva
- `_grasp_confirm_count` (int): contatore step grasp consecutivi
- `_hold_closed_duration` (int): step con porta chiusa in HOLD
- `_retreat_pos` (np.array): target = `eef_pos_at_hold + [-0.13, 0, 0.04]`

### One-Hot nell'Osservazione
```
[fsm_reach, fsm_push, fsm_hold, fsm_retreat] ∈ {0,1}^4
```

---

## 4. Rete Neurale — SAC MLP

| Componente | Dimensione |
|------------|-----------|
| Input (obs) | ~40 dim (base Robosuite + 8 custom features) |
| Actor (2 hidden layer) | 512 → 512 → [μ(7), log_σ(7)] |
| Critic Q1/Q2 (2 hidden layer) | 512 → 512 → 1 |
| Action output | 7 dim (6 DOF + 1 gripper) |
| Totale parametri | ~867k |

### Features Custom nell'Osservazione (8 dim)
1. `dist` — distanza EEF-handle
2. `_current_handle_radius` — raggio handle (randomizzato)
3. `_current_handle_friction` — frizione handle (randomizzata)
4. `fsm_reach`, `fsm_push`, `fsm_hold`, `fsm_retreat` — stato FSM one-hot
5. `hinge_qpos` (live) — angolo porta

### Iperparametri SAC Chiave
```
learning_rate  = 3e-4
buffer_size    = 1,000,000
batch_size     = 256
gamma          = 0.95
tau            = 0.005
gradient_steps = 2
ent_coef       = "auto"
net_arch       = [512, 512]
```

---

## 5. Generalizzazione — Randomizzazioni

### Per Episodio (ogni reset):
| Variabile | Range | Scopo |
|-----------|-------|-------|
| Handle radius | ×0.7 – ×1.4 (base=2cm) | Grip su maniglie diverse |
| Handle length | ×0.8 – ×1.2 (base=8cm) | Geometria contatto |
| Handle friction | ×0.3 – ×1.2, clip[0.05,2.0] | Forza presa adattiva |
| Door angle iniziale | 70%–100% del range | Partenza da angoli diversi |

### Con Curriculum (se `curriculum_level > 0`):
| Variabile | Range max (level=1.0) | Scopo |
|-----------|----------------------|-------|
| Offset posizione XY | ±15 cm | Invarianza traslazione |
| Yaw rotazione porta | ±17.2° | Invarianza orientazione |
| Distanza robot-porta | 50–60 cm | Invarianza distanza |

### Curriculum Adattivo
- Avanza di +0.05 ogni 25k step se `success_rate > 0.85` AND `grasp_rate > 0.50`
- Bloccato se il robot usa solo pushing (grasp_rate basso)

---

## 6. Sistema di Reward (Sintesi)

### FASE 1 — REACH
| Termine | Valore |
|---------|--------|
| `dist_3d` | -5.0 × dist |
| `dist_z` | -15.0 × |dZ| (segnale dominante) |
| `dist_xy` | -3.0 × dist_xy |
| `phase_trans` | **+50.0** (una-tantum, grasp confermato) |

### FASE 2 — PUSH
| Termine | Valore |
|---------|--------|
| `door_prog` | **+2000.0 × Δangle** (segnale dominante) |
| `dist_3d` | -5.0 × dist (mantieni contatto) |
| Tolleranza perd. grasp | dinamica: 5–12 cm base su velocità porta |

### FASE 3 — HOLD
| Termine | Valore |
|---------|--------|
| `hold_veldamp` | -25.0 × |door_qvel| (segnale dominante anti-bounce) |
| `hold_bounce` | -20.0 × |door_qpos| |
| `hold_grip` | +1.0 (gripper chiuso) |
| `hold_act` | +1.0 (braccio fermo, grazie all'override) |

### FASE 4 — RETREAT
| Termine | Valore |
|---------|--------|
| `ret_grip` | +2.0 (gripper aperto) |
| `ret_dir` | +3.0 × alignment (verso target) |
| `ret_jnt_prog` | -5.0 × w × ‖joint_vel‖ (freeze progressivo) |

### Terminazione
| Evento | Reward |
|--------|--------|
| Success completo | +500.0 |
| Latch non neutro O porta non chiusa | -500.0, terminated=False |

---

## 7. Test di Ablazione — Cartella `scratch/`

### Cronologia dei Test

#### TEST A — `test_wait_logic.py`: WaitDoorEnv
**Data:** Durante sviluppo (pre-fix)  
**Meccanismo:** Blocca la transizione `_ready_to_retreat=True` finché `|latch_qpos| >= 0.15 rad`  
**Risultato:** **0% successo** (10/10 episodi in timeout a 500 step)

**Analisi:**  
Si crea un **deadlock meccanico**:
- L'agente è in HOLD → gripper chiuso (grip ≈ +0.83)
- Il gripper comprime la maniglia curva → le dita bloccano fisicamente la molla del latch
- Il latch non può tornare a neutro finché le dita sono chiuse sulla maniglia
- L'agente aspetta che il latch torni neutro → deadlock

**Conclusione:** Impossibile aspettare il latch in HOLD prima di transitare a RETREAT.

---

#### TEST B — `test_freeze_logic.py`: FreezeDoorEnv
**Data:** Durante sviluppo (pre-fix)  
**Meccanismo:** Permette transizione a RETREAT, ma congela `action[:-1]=0.0` finché `|latch_qpos| >= 0.15`  
**Risultato:** **0% latch neutro** alla transizione, media **439.3 step/episodio** (max 500)

```
Ep 1  - Steps: 500 - Success: False - Latch qpos: N/A (timeout)
Ep 2  - Steps: 500 - Success: False ...
[Tutti gli episodi in timeout]
Tasso Latch Neutro alla transizione = 0.0%
```

**Analisi:**  
Congelando il braccio in RETREAT finché il latch non è neutro, il braccio non si muove.  
Se il braccio non si muove, il gripper non si sfila dalla maniglia.  
Se il gripper non si sfila, il latch non può tornare neutro.  
→ Secondo deadlock, diverso ma ugualmente catastrofico.

**Conclusione:** Impossibile bloccare il braccio in RETREAT aspettando il latch.

---

#### TEST C — `test_override_grip.py`: LogOverrideGripDoorEnv
**Data:** Durante sviluppo (pre-fix)  
**Meccanismo:** Forza `action[-1]=-1.0` (gripper aperto) negli ultimi 15 step di HOLD (`_hold_closed_duration >= 45`)  
**Risultato:** Latch non torna neutro durante HOLD; torna solo durante RETREAT

**Log step-per-step:**
```
Step 62 (duration=0):  Gripper action=+0.9985. Latch before step: 1.4720
Step 63 (duration=1):  Gripper action=+0.9985. Latch before step: 1.4664
...
Step 76 (duration=14): Enforcing gripper open. Latch before step: 1.4588
Step 77 (duration=15): Enforcing gripper open. Latch before step: 1.4532
Step 78 (duration=16): Enforcing gripper open. Latch before step: 1.4476
...
Step 87 (duration=25): Enforcing gripper open. Latch before step: 1.2796
Step 88:               RETREAT phase started. Latch before step: 1.2682
Step 110:              Latch returned to neutral. Latch before step: 0.0984
```

**Analisi:**  
Anche forzando il gripper ad aprirsi per 15 step durante HOLD (braccio fermo):
- `latch_qpos` scende lentamente: 1.4588 → 1.2682 in 15 step (solo 0.19 rad in 0.5s!)
- Al momento della transizione RETREAT, il latch è ancora a 1.27 rad (ben oltre 0.15 rad)
- Il latch torna neutro (0.098 rad) al **step 110**, ovvero 22 step DOPO l'inizio del RETREAT

**Conclusione critica:** Lo sblocco fisico del latch avviene **esclusivamente grazie al movimento di ritirata** del braccio (sfilamento dinamico). Aspettarlo in HOLD è impossibile.

---

#### TEST D — `test_hold_freeze.py`: HoldFreezeDoorEnv
**Data:** Durante sviluppo  
**Meccanismo:** `action[:-1]=0.0` in HOLD (braccio completamente fermo, gripper libero)  
**Risultato:** Verifica base della fattibilità del freeze — successo se il gripper rimane chiuso

**Significato:** Questo test ha dimostrato che il freeze del braccio non impedisce la chiusura porta  
(il gripper mantiene indipendentemente la presa), ma manca il check grip alla transizione.

---

#### TEST E — `test_hold_freeze_grip.py`: HoldFreezeGripDoorEnv
**Data:** Pre-fix finale  
**Meccanismo:** `action[:-1]=0.0` in HOLD + richiede `action[-1] > 0.80` per transizione PUSH→HOLD  
**Risultato:** Questo è il **fix finale implementato** in `env_gen.py`

**Innovazione rispetto a TEST D:**  
La condizione `action[-1] > 0.80` alla transizione garantisce un grip solido iniziale,  
eliminando i micro-slip che causavano bounce immediato all'entrata in HOLD.

---

#### TEST F — `test_original_wait_logic.py`: ExactWaitDoorEnv  
**Data:** Post-training (validazione retroattiva)  
**Meccanismo:** Reimplementa la wait logic storica (commit 517e021c): blocca RETREAT finché `|latch_qpos| < 0.15`  
**Risultato:** **0% successo** su 10 episodi

**Note:** Questo test usa il modello già addestrato con i fix applicati.  
Anche con una policy convergente a 100% success, re-introdurre la wait logic causa 0% successo.  
Conferma retroattivamente che la wait logic non è mai stata una soluzione valida.

---

### Tabella Riassuntiva Test Ablazione

| Test | File | Meccanismo | Risultato |
|------|------|-----------|-----------|
| A | `test_wait_logic.py` | Blocca RETREAT se latch>0.15 | 0% successo, deadlock |
| B | `test_freeze_logic.py` | Congela braccio RETREAT se latch>0.15 | 0% latch neutro, 439s medio |
| C | `test_override_grip.py` | Forza gripper aperto ultimi 15 step HOLD | Latch neutro solo 22 step dopo RETREAT |
| D | `test_hold_freeze.py` | action[:-1]=0.0 in HOLD | Funzionale, manca grip check |
| E | `test_hold_freeze_grip.py` | D + grip>0.80 alla transizione | **Fix finale implementato** |
| F | `test_original_wait_logic.py` | Wait logic storica (commit 517e021c) | 0% successo |

---

## 8. Fix Finali Implementati

### Fix 1 — Grip Solido alla Transizione PUSH → HOLD

**Problema:**  
La transizione avveniva non appena `door_angle <= success_angle`, indipendentemente  
dallo stato del gripper. Con grip incerto, la porta rimbalzava subito all'entrata in HOLD.

**Soluzione:** (implementata in `env_gen.py::step`)
```python
if door_angle <= self._success_angle and not self._success_latched:
    if action[-1] > 0.80:   # ← AGGIUNTA questa condizione
        self._success_latched = True
        just_succeeded = True
```

**Effetto:** La transizione richiede che il gripper stia stringendo con almeno l'80% dell'intensità massima.  
Elimina i falsi positivi alla transizione che causavano rimbalzi immediati.

### Fix 2 — Congelamento Totale del Braccio in HOLD

**Problema:**  
Precedentemente, le azioni del braccio venivano scalate × 0.1 durante HOLD (classe base).  
Nonostante lo scaling, rimanevano micro-oscillazioni che accumulavano penalità.

**Soluzione:** (implementata in `env_gen.py::step`)
```python
if self._success_latched:
    is_ready_retreat = getattr(self, "_ready_to_retreat", False)
    if not is_ready_retreat:
        action[:-1] = 0.0   # ← HARD FREEZE (0.0, non 0.1×)
        # action[-1] invariato: gripper mantiene presa
```

**Effetto:**
- Braccio completamente fermo → eliminazione totale delle oscillazioni
- `hold_act = +1.0` ad ogni step (azione_norm = 0 garantito)
- `hold_jnt_freeze ≈ -0.03` (quasi zero, residuo fisico compensazione gravità)

---

## 9. Risultati Verificazione Pre-Resume (Post Fix)

**Dataset:** 50 episodi deterministici + 50 episodi stocastici  
**Modello:** Checkpoint più recente con fix applicati

| Metrica | Deterministico | Stocastico |
|---------|---------------|-----------|
| Success Rate | **100.0%** (50/50) | **100.0%** (50/50) |
| Lunghezza media episodio | **110.7 step** | **112.3 step** |
| Latch Neutro al termine | **100.0%** | **100.0%** |
| Porta Chiusa al termine | **100.0%** | **100.0%** |

**Miglioramento rispetto ai checkpoint precedenti:**
- Step medi deterministici: 111.4 → **110.7** (−0.7 step)
- Step medi stocastici: 120.5 → **112.3** (−8.2 step, −6.8%)

---

## 10. Analisi dei Log di Training (1.82M Step)

### Metriche di Rollout

| Metrica | Valore | Interpretazione |
|---------|--------|----------------|
| `success_rate` | **100.0%** | Policy convergente, nessuna regressione |
| `ep_rew_mean` | 1050–1140 | Agente massimizza quasi tutti i bonus |
| `ep_len_mean` | ~122 step | Ottimizzato: ~20 REACH + ~20 PUSH + 60 HOLD + 22 RETREAT |
| `ent_coef` | **0.00034** | Policy deterministica e convergente |

### Scomposizione Temporale Media (Training)
```
~20 step  → Phase 1: REACH
~20 step  → Phase 2: PUSH
 60 step  → Phase 3: HOLD (obbligatori, 2.0s @ 30Hz)
~22 step  → Phase 4: RETREAT
= 122 step totali
```

### Log Diagnostico Phase 3 (HOLD)
```
│ 3:HOLD  │  0.020 │ -0.013 │ +0.99 │ PHYS_OK   │ 0.051 │  0.14 │  0.00 │ +1.30 │
  ↳ REWARDS │ base: -0.50 │ hold: +1.00 │ hold_grip: +1.00 │ 
             hold_jnt_freeze: -0.03 │ hold_act: +1.00 │ hold_flat: -1.18 │ TOT: +1.29
```

**Analisi:**
- `hold_act: +1.00`: conferma il freeze del braccio — azione_norm = 0 garantito
- `hold_jnt_freeze: -0.03`: velocità giunti quasi a zero (-0.03 = residuo gravità)
- `hold_grip: +1.00` + `GRIP +0.99`: gripper completamente serrato
- `TOT: +1.29`: reward positivo in HOLD (sistema stabile)

### Log Diagnostico Phase 4 (BACK - Terminazione)
```
│ 4:BACK  │  0.103 │ -0.100 │ +0.00 │ PHYS_OK   │ 0.061 │  0.56 │  0.00 │ +0.11 │
  ↳ REWARDS │ base: +500.93 │ hold: +1.00 │ ret_grip: -1.00 │ 
             ret_jnt_prog: -0.92 │ latch_ret: -0.11 │ TOT: -0.25
```

**Analisi:**
- `base: +500.93`: SUCCESS BONUS — distanza retreat < 5cm → terminazione riuscita
- `LATCH: +0.11 rad`: latch quasi neutro (< soglia 0.15), confermato dal movimento di ritirata
- `GRIP: +0.00`: gripper completamente aperto in RETREAT
- `TOT: -0.25`: il reward netto è lievemente negativo ma il bonus terminazione domina

---

## 11. Diagnostica Avanzata — `diag_phase34.py`

### T1 — Latch Spring Test
**Risultato tipico:** Il latch torna a < 0.1 rad in **22–90 step** (dipende da stiffness/damping MuJoCo)  
**Implicazione:** La molla è sufficiente a riportare il latch al neutro, MA solo se il gripper non blocca.

### T2 — Hinge Damping
**Risultato:** Velocità bounce max: dipende da `damping` del giunto hinge  
**Penalità veldamp a peso -25:** efficace nel prevenire rimbalzi durante HOLD

### T3 — Action Norm durante HOLD (post-fix)
```
Frac action_norm < 0.05 (reward +1): ~100%  ← effetto del hard freeze
Media action_norm: ~0.000
Penalità media hold_act: ~0.000
```

### T4 — Wrist Rotation durante RETREAT
```
Peso penalità ret_rot: -3.0 (ridotto da -10.0 nella logica storica)
Media wrist_rot: ~0.05–0.15 rad/step
```

### T5 — Latch qpos alla Transizione HOLD→RETREAT
```
Media latch_qpos: 1.2–1.4 rad (ben sopra 0.15)
% sopra 0.15 rad: ~100% (ovvio: il latch non è mai neutro in HOLD)
```
Questo conferma che la wait logic (aspettare latch < 0.15 in HOLD) sarebbe sempre in deadlock.

### T6 — Bounce Events in Phase 3 (post-fix)
```
Bounce events (|door_qvel| > 0.05): 0 (post-fix)
Porta stabile in Phase 3.  ← effetto del hard freeze + grip saldo
```

---

## 12. Diagnostica Valutazione — `eval_stats.py`

### Classificazione Fallimenti
```python
classify_failure(max_phase_idx, dist_handle, door_angle, latch_qpos, step_count, is_success)
```

| Tipo Fallimento | Condizione | Causa Tipica |
|-----------------|-----------|--------------|
| SUCCESS | is_success = True | — |
| REACH timeout | max_phase = 1 | Non riesce a raggiungere la maniglia |
| GRASP lost | max_phase = 2, dist > 8cm | Perde il grip durante PUSH |
| PUSH timeout | max_phase = 2, dist <= 8cm | Raggiunge la maniglia ma non chiude la porta |
| HOLD bounce/timeout | max_phase = 3 | Bounce porta in HOLD |
| RETREAT door bounce | max_phase = 4, door_angle >= 0.03 | Porta si riapre in RETREAT |
| RETREAT latch not neutral | max_phase = 4, |latch| >= 0.08 | Latch non neutro alla fine |
| RETREAT timeout | max_phase = 4, altrimenti | Non raggiunge retreat_pos |

### Plots Generati
1. **`success_rate.png`**: barchart successo deterministico vs stocastico
2. **`max_phase_dist.png`**: distribuzione fase massima raggiunta
3. **`avg_phase_time.png`**: tempo medio in ogni fase

---

## 13. Checkpoint e Modelli Storicizzati

### `scratch/model_almost_complete_14_05/`
Checkpoint del **14 maggio**. Versione quasi completa ma con problemi residui:
- Oscillazioni in HOLD (freeze non ancora hard)
- Latch non sempre neutro alla fine

### `scratch/model_almost_almost_complete_21_05/`
Checkpoint del **21 maggio**. Versione con fix quasi completi:
- `test_model.zip`: pesi modello SAC
- `test_vn.pkl`: statistiche VecNormalize  
- `eval_stats.pkl`: statistiche valutazione
- Plots: `avg_phase_time.png`, `max_phase_dist.png`, `success_rate.png`

### `runs/close_gen/` (versione finale)
Modello convergente al **100% success rate** con tutti i fix applicati.

---

## 14. Analisi della Fisica MuJoCo

### Problema del Latch (Dettaglio)

Il meccanismo del latch è il **nodo critico** del task:
```
Durante PUSH:
  - Il gripper ruota fisicamente la maniglia curva
  - latch_qpos aumenta da 0 a ~1.2–1.5 rad
  - Questo è il "comprimere la maniglia verso il basso" che permette lo scatto

Durante HOLD:
  - Il gripper tiene compressa la maniglia
  - Le dita del robot bloccano fisicamente la molla del latch
  - latch_qpos rimane alto (~1.2–1.4 rad)
  - NON può tornare neutro

Durante RETREAT:
  - Il braccio si muove indietro (-13 cm in X)
  - Il gripper apre (action[-1] → -1.0)
  - Il gripper si SFILA dalla maniglia (cinematica di sfilamento)
  - La molla del latch ora è LIBERA di spingere → torna a 0 in ~22 step
```

### Vincolo Temporale Hard: 60 Step di HOLD

La durata di HOLD (2.0s @ 30Hz) non è arbitraria:
- Meno di 60 step → porta non stabilizzata → bounce al RETREAT
- Tentare di accorciare provoca wedging delle dita nella maniglia curva
- Forzare la ritirata anticipata → fallimento di ritirata o rimbalzo elastico della porta

---

## 15. Conclusioni Tecniche

### A. Saturazione Matematica delle Metriche
Con `success_rate = 100%` e `latch_neutral = 100%`, qualsiasi modifica rischia:
- Instabilità in una policy già convergente
- Exploitation secondario di nuovi reward shaping
- Catastrophic forgetting a causa di disallineamento nei gradienti Q-function

### B. Vincoli Fisici (MuJoCo)
Il minimo teorico dell'episodio è:
```
~20 step REACH + ~20 step PUSH + 60 step HOLD (tassativo) + ~22 step RETREAT = ~122 step
```
Non è possibile scendere sotto questo limite senza violare i vincoli fisici del latch.

### C. Convergenza SAC (`ent_coef = 0.00034`)
A questo livello di convergenza:
- La policy è praticamente deterministica
- La Q-function è stabilizzata
- Modificare i reward causerebbe "disapprendimento" (catastrophic forgetting)
- Il sistema ha raggiunto il **massimo compromesso** tra efficienza temporale, fluidità e stabilità

### D. Sequenza Cinematicamente Corretta (Unica)
I test di ablazione dimostrano che l'**unica sequenza fisicamente corretta** è:
```
HOLD (braccio fermo, gripper chiuso, 60 step)
  → RETREAT (braccio si muove -13cm, gripper si apre)
     → latch ritorna neutro (22 step dopo l'inizio del RETREAT)
        → TERMINATED (success)
```
Qualsiasi variazione (wait logic, freeze RETREAT, override grip in HOLD)  
porta a deadlock meccanico o fallimento temporale.

---

## 16. Comandi Principali

```bash
# Training da zero
python close_generalized/train_gen.py

# Resume training
python close_generalized/train_gen.py --resume --total-steps <steps_aggiuntivi>

# Play (visualizzazione)
mjpython close_generalized/train_gen.py --play

# Valutazione completa (50 episodi det + stocastici)
python eval_stats.py

# Diagnostica fasi 3-4
python close_generalized/diag_phase34.py

# Test ablazione
python scratch/test_wait_logic.py
python scratch/test_freeze_logic.py
python scratch/test_override_grip.py
python scratch/test_hold_freeze.py
python scratch/test_hold_freeze_grip.py
python scratch/test_original_wait_logic.py
```

---
