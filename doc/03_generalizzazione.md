# 🔀 Generalizzazione — Tutto ciò che viene randomizzato durante il Training

## Obiettivo della Generalizzazione

Il task di chiusura porta generalizzato (`GeneralizedDoorEnv`) mira a produrre una policy **robusta**  
che funzioni correttamente indipendentemente da:
1. La **dimensione** e la **frizione** della maniglia
2. La **posizione** e l'**orientazione** della porta nell'ambiente
3. La **distanza** del robot dalla porta
4. La **fase FSM corrente** (comunicata esplicitamente come one-hot)

---

## 1. Randomizzazione della Geometria dell'Handle

**Dove:** `GeneralizedDoorEnv.reset()` — eseguita ad **ogni reset episodio**

### 1a. Raggio dell'Handle

```python
base_radius = 0.02   # 2 cm (raggio base MuJoCo)
r_scale = np.random.uniform(0.7, 1.4)
self._current_handle_radius = base_radius * r_scale
# → range: [0.014 m, 0.028 m]  (1.4 cm – 2.8 cm)

self._rs_env.sim.model.geom_size[self.handle_geom_id][0] = self._current_handle_radius
```

**Effetto fisico:**
- Handle più piccola → grip più difficile (tolleranza dist ridotta)
- Handle più grande → grip più facile ma richiede gripper più aperto
- La variazione ±40% simula diversi tipi di maniglie reali

### 1b. Lunghezza dell'Handle

```python
base_length = 0.08   # 8 cm
l_scale = np.random.uniform(0.8, 1.2)
self._rs_env.sim.model.geom_size[self.handle_geom_id][1] = base_length * l_scale
# → range: [0.064 m, 0.096 m]  (6.4 cm – 9.6 cm)
```

**Effetto fisico:**
- Variazione ±20% sulla lunghezza
- Influenza la geometria di contatto gripper-handle

### 1c. Frizione dell'Handle (Bidirezionale)

```python
f_scale = np.random.uniform(0.3, 1.2)   # range ampio: 0.3× a 1.2× la base
base_f  = self.base_friction[0]          # frizione base del modello MuJoCo
new_f   = float(np.clip(base_f * f_scale, 0.05, 2.0))
# → range assoluto: [0.05, 2.0]

self._rs_env.sim.model.geom_friction[self.handle_geom_id][0] = new_f
self._current_handle_friction = new_f
```

**Effetto fisico:**
- `f_scale < 1`: handle scivolosa (difficile tenere grip durante PUSH)
- `f_scale > 1`: handle aderente (più facile mantenere il contatto)
- Range 0.3×–1.2× = dal 30% al 120% della frizione base
- Insegna alla policy ad **adattare la forza di presa** in base alla frizione

---

## 2. Randomizzazione Posizione e Orientazione della Porta

**Dove:** `GeneralizedDoorEnv.reset()` — solo se `curriculum_level > 0`

### Parametri di Scala

```python
p_var = 0.15 * curriculum_level   # varianza posizione: 0→15 cm al massimo
r_var = 0.30 * curriculum_level   # varianza rotazione: 0→0.30 rad (~17°) al massimo
```

### 2a. Offset Posizione XY

```python
pos_offset = np.random.uniform(-p_var, p_var, size=3)
pos_offset[2] = 0   # solo piano orizzontale (no offset verticale)
# → range max (level=1.0): ±15 cm in X, ±15 cm in Y
```

### 2b. Rotazione Yaw (attorno a Z)

```python
yaw = np.random.uniform(-r_var, r_var)
# → range max (level=1.0): ±0.30 rad = ±17.2°

q_scipy = R_scipy.from_euler('z', yaw).as_quat()
```

Questo varia l'**angolo di apertura** della porta rispetto al robot. La policy deve imparare a raggiungere la maniglia indipendentemente dall'orientazione della porta.

### 2c. Composizione del Quaternione Finale

```python
# Conversione quaternione base (MuJoCo usa wxyz, scipy usa xyzw)
q_base = R_scipy.from_quat([
    base_quat[1], base_quat[2], base_quat[3], base_quat[0]
])
q_new = R_scipy.from_quat(q_scipy) * q_base   # moltiplicazione rotazione
res_q = q_new.as_quat()  # xyzw

# Ri-conversione a MuJoCo format (wxyz)
sim.model.body_quat[door_body_id] = np.array([res_q[3], res_q[0], res_q[1], res_q[2]])
```

### 2d. Randomizzazione Distanza Robot-Porta (Opzionale)

```python
if hasattr(cfg, "human_dist_min") and hasattr(cfg, "human_dist_max"):
    dist_sample = np.random.uniform(cfg.human_dist_min, cfg.human_dist_max)
    # cfg.human_dist_min = 0.50, cfg.human_dist_max = 0.60
    sim.model.body_pos[door_body_id][0] = dist_sample + pos_offset[0]
    # → distanza porta in [0.50, 0.60] m + offset
```

Questo simula diverse distanze tra il robot e la porta (50–60 cm), rendendo la policy invariante alla distanza iniziale.

---

## 3. Curriculum Learning Adattivo

### Logica di Progressione

Il curriculum level parte da **0.0** (nessuna randomizzazione) e cresce gradualmente fino a **1.0** (massima randomizzazione).

```python
class AdaptiveCurriculumCallback:
    check_freq = 25_000  # step

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            sr = success_rate (recente)
            gr = grasp_rate (recente)
            current_level = env.curriculum_level

            if sr > 0.85 and gr > 0.50 and current_level < 1.0:
                new_level = min(1.0, current_level + 0.05)
                env.set_curriculum_level(new_level)
                # Reset contatori dopo aggiornamento
```

### Condizioni di Avanzamento

| Condizione | Soglia | Motivazione |
|-----------|--------|-------------|
| `success_rate > 0.85` | 85% successi | Il robot completa il task nella configurazione corrente |
| `grasp_rate > 0.50` | 50% grasps | Il robot sta usando il grasp (non solo pushing) |

### Incremento Graduale
- Step: +0.05 per ogni avanzamento (20 livelli totali)
- Frequenza check: ogni 25,000 step
- Massimo teorico: raggiungimento di `level=1.0` in ~500k step (se sempre in avanzamento)

### Blocco per Pushing Eccessivo
```python
elif sr > 0.85 and gr <= 0.50:
    print("CURRICULUM bloccato: grasp_rate troppo bassa")
```
Se il robot ha 85% successo ma non usa il grasp (sta solo spingendo la porta),  
il curriculum NON avanza. Questo forza l'apprendimento della sequenza di grasp corretta.

### Effetto sul Reset dell'Ambiente

```python
def reset(self):
    p_var = 0.15 * self.curriculum_level  # 0 → 15 cm
    r_var = 0.30 * self.curriculum_level  # 0 → 17.2°
    
    if self.curriculum_level > 0:
        # randomizzazione attiva
```

A `curriculum_level = 0.0`: porta sempre in posizione standard, handle standard → apprendimento base.  
A `curriculum_level = 1.0`: massima variabilità → generalizzazione completa.

---

## 4. Features di Generalizzazione nell'Osservazione

Per permettere alla policy di **adattare il comportamento** alle variazioni, vengono fornite esplicitamente nell'osservazione:

### Features di Contesto della Handle
```python
custom = np.array([
    dist,                          # EEF-handle distance (adatta strategia avvicinamento)
    self._current_handle_radius,   # Geometria handle (adatta apertura gripper)
    self._current_handle_friction, # Frizione handle (adatta forza presa)
    ...
])
```

### Features di Stato FSM (One-Hot)
```python
fsm_reach, fsm_push, fsm_hold, fsm_retreat
```
Questi 4 bit permettono alla policy di avere **comportamenti specializzati per fase**  
pur usando una singola rete neurale.

### Angolo Corrente della Porta (Live)
```python
hinge_qpos = float(sim.data.qpos[hinge_qpos_addr])  # NON obs['hinge_qpos']
```
Essenziale per la Fase 2 (PUSH): la maniglia si muove lungo un arco,  
e la policy deve prevedere dove sarà la maniglia mentre chiude la porta.

---

## 5. Randomizzazione dell'Angolo Iniziale della Porta

**Dove:** `RoboSuiteDoorCloseGymnasiumEnv.reset()` (classe base)

```python
rng  = door_max - door_min
lo   = door_min + init_open_min_fraction * rng  # 70% apertura
hi   = door_min + init_open_max_fraction * rng  # 100% apertura
angle = np.random.uniform(lo, hi)

sim.data.qpos[hinge_qpos_addr] = angle
sim.forward()
```

**Effetto:**
- La porta inizia in posizione casuale tra il 70% e il 100% di apertura
- La policy impara a chiudere la porta da qualsiasi angolo di apertura iniziale
- Intervallo: circa 70%–100% del range `[door_min, door_max]`

---

## 6. Action Smoothing (EMA)

```python
alpha = cfg.action_smooth_alpha  # = 0.8
action = alpha * action_raw + (1 - alpha) * prev_action
```

Non è una forma di generalizzazione dell'ambiente, ma **riduce la sensibilità a piccole perturbazioni**:
- Evita jerk improvvisi dell'azione
- Produce traiettorie più fluide
- Riduce il rischio che rumori stocastici dell'ambiente causino drop del grasp

---

## Sommario: Tutte le Sorgenti di Variabilità

| Variabile | Range | Quando | Scopo |
|-----------|-------|--------|-------|
| Handle radius | ×0.7 → ×1.4 base | Ogni reset | Grip su handle di dimensioni diverse |
| Handle length | ×0.8 → ×1.2 base | Ogni reset | Geometria contatto variabile |
| Handle friction | ×0.3 → ×1.2 base, clip [0.05, 2.0] | Ogni reset | Forza di presa adattiva |
| Posizione porta XY | ±15 cm × curriculum | Ogni reset (level>0) | Invarianza traslazione |
| Yaw porta | ±17.2° × curriculum | Ogni reset (level>0) | Invarianza rotazione |
| Distanza robot-porta | [0.50, 0.60] m | Ogni reset (level>0) | Invarianza distanza |
| Angolo iniziale porta | 70%–100% range | Ogni reset | Partenza da angoli diversi |
| EMA alpha | 0.8 (fisso) | Ogni step | Robustezza al rumore azioni |

---

## Cosa NON viene randomizzato

- **Robot**: sempre Franka Panda
- **Struttura** della porta (frame, cardine): fissa
- **Latch joint physics**: stiffness e damping fissi
- **Control frequency**: sempre 30 Hz
- **Tipo di controller**: sempre BASIC (operational space)
- **Horizon**: 500 step fissi

---

## Invarianze Apprese dalla Policy

Grazie alla randomizzazione, la policy apprende a essere robusta rispetto a:

1. **Invarianza geometrica handle**: usa `_current_handle_radius` dall'osservazione per calibrare l'apertura del gripper
2. **Invarianza alla frizione**: usa `_current_handle_friction` per modulare la forza di presa
3. **Invarianza alla posizione porta**: naviga verso la maniglia indipendentemente dalla posizione
4. **Invarianza all'orientazione porta**: raggiunge la maniglia da angoli diversi
5. **Invarianza alla fase FSM**: comportamento distinto per fase grazie al one-hot encoding
6. **Invarianza all'angolo di apertura**: la porta può essere 70-100% aperta all'inizio
