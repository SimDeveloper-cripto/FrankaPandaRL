# 💰 Reward & Penalità — Meccanismo Completo

## Architettura del Sistema di Reward

Il reward è **strettamente legato alla FSM**: ogni fase ha un sottoinsieme distinto  
di reward e penalità. Il reward totale è la **somma** di tutti i termini attivi nello step.

```python
reward = 0.0
rew_info = {}  # dizionario termine_nome → valore

# Accumulo
for v in rew_info.values():
    reward += v

# Clipping (eccetto terminazione)
if not terminated:
    reward = clip(reward, -100.0, 100.0)
```

---

## Pre-Calcolo Comune (tutte le fasi)

Prima di entrare nei branch FSM, vengono calcolate variabili condivise:

```python
eef_pos    = obs["robot0_eef_pos"]
handle_pos = obs.get("handle_pos", obs.get("door_handle_pos", eef_pos))

dist_handle = ‖eef_pos - handle_pos‖        # distanza 3D EEF-handle
dist_xy     = ‖eef_pos[:2] - handle_pos[:2]‖ # distanza nel piano XY
height_diff = eef_pos[2] - handle_pos[2]     # differenza di quota Z (+ = robot sopra)

door_qpos  = sim.data.qpos[hinge_qpos_addr]  # angolo porta (rad)
is_closed  = |door_qpos| < 0.03              # porta chiusa
latch_qpos = sim.data.qpos[handle_qpos_addr] # angolo latch (rad)

grip_tol   = _current_handle_radius + 0.03   # tolleranza distanza = raggio + 3cm

# Stato gripper
gripper_action = action[-1]
gripper_qpos   = obs["robot0_gripper_qpos"]
gripper_width  = sum(|gripper_qpos|)
handle_diameter = _current_handle_radius * 2.0
is_physically_closed = (gripper_width <= handle_diameter + 0.025) AND (gripper_width >= 0.015)
```

### Allineamento EEF → Handle

```python
delta_pos  = handle_pos - eef_pos
dir_to_handle = delta_pos / ‖delta_pos‖
rmat       = R.from_quat(eef_quat).as_matrix()
eef_z      = rmat[:, 2]   # asse Z del polso (punta del gripper)
eef_x      = rmat[:, 0]   # asse X del polso

alignment      = |dot(eef_z, dir_to_handle)|  # 0=perpendicolare, 1=perfettamente allineato
flat_alignment = |eef_x[2]|                   # 0=polso piatto, 1=polso ruotato verso l'alto
```

### Smoothness (penalità jerk)

```python
jerk = ‖action[:-1] - prev_eef_action‖
rew_info["smoothness"] = -1.0 * jerk
```
Attivo in tutte le fasi (piccolo effetto regolarizzante).

---

## FASE 1: REACH (Raggiungimento Maniglia)

**Condizione:** `not _grasp_phase and not _success_latched`

### Reward Base Modificato

```python
rew_info["base"] = base_reward - _W_GRIPPER_CLOSE
# base_reward: da RoboSuiteDoorCloseGymnasiumEnv (progress, delta, -time_penalty)
# Penalità aggiuntiva: -2.5 (scoraggia chiusura prematura del gripper)
```

Il termine `_W_GRIPPER_CLOSE = 2.5` viene sottratto dalla base per compensare  
il fatto che la classe padre potrebbe incoraggiare la chiusura del gripper,  
mentre in questa fase vogliamo che il gripper rimanga aperto.

### Segnale di Distanza 3D

```python
rew_info["dist_3d"] = -_W_REACH_3D * dist_handle  # -5.0 × dist
```
Spinge il robot verso la maniglia. Lineare nella distanza.

### Segnale di Distanza XY

```python
rew_info["dist_xy"] = -_W_REACH_XY * dist_xy  # -3.0 × dist_xy
```
Enfatizza il raggiungimento in piano (dove la maniglia è più facilmente raggiungibile).

### Segnale di Altezza

```python
rew_info["dist_z"] = -_W_REACH_Z * abs(height_diff)  # -15.0 × |dZ|
```
Il peso **altissimo** (15.0) forza l'EEF ad allinearsi verticalmente alla maniglia.  
Questo è il segnale dominante nella fase di avvicinamento.

### Penalità Approccio da Sotto

```python
if height_diff < -_APPROACH_HEIGHT_TOL:  # EEF è SOTTO la maniglia (height_diff < -0.005)
    rew_info["app_blw"] = -_W_APPROACH_BELOW * abs(height_diff + _APPROACH_HEIGHT_TOL)
    # = -3.0 × (distanza dall'altezza minima consentita)
```
Penalizza approcci dal basso (il gripper non può chiudersi bene se viene da sotto).

### Penalità Push Prematuro dall'Alto

```python
if height_diff > 0.03 and gripper_action > 0.2:
    rew_info["app_top"] = -_W_PUSH_PENALTY * height_diff * gripper_action
    # = -5.0 × height_diff × gripper_action
```
Se il robot è troppo in alto (>3 cm sopra la maniglia) E sta cercando di chiudere  
il gripper, viene penalizzato. Evita che il robot premi sulla maniglia dall'alto.

### Allineamento EEF → Handle (scalato per prossimità)

```python
prox_factor = exp(-10.0 × dist_handle)  # 1.0 se dist=0, ~0 se dist>0.5

if dist_handle < grip_tol × 3.0:        # solo se abbastanza vicino
    rew_info["align"] = -1.0 × (1.0 - alignment) × prox_factor
    # Premio per puntare il gripper verso la maniglia
    rew_info["flat"]  = -0.5 × flat_alignment × prox_factor
    # Premio per mantenere il polso piatto (non ruotato)
```
Il fattore esponenziale fa sì che l'allineamento conti solo quando il robot è vicino.  
Evita segnali contrastanti durante la navigazione lontana.

### Gestione Gripper in REACH

**Caso A — Lontano** (`dist_handle > 0.025` m):
```python
if gripper_action > _GRIPPER_OPEN_THRESH:  # > -0.85 (non completamente aperto)
    rew_info["grip"] = -1.0 × (gripper_action - _GRIPPER_OPEN_THRESH)
    # Penalità: il gripper non dovrebbe chiudersi mentre si avvicina lontano
    
self._grasp_confirm_count = 0   # reset contatore grasp
```

**Caso B — Vicino** (`dist_handle <= 0.025` m):
```python
if gripper_action > _GRIPPER_OPEN_THRESH:  # sta chiudendo
    rew_info["grip"] = _W_GRIPPER_CLOSE × ((gripper_action - _GRIPPER_OPEN_THRESH) / (1.0 - _GRIPPER_OPEN_THRESH))
    # = 2.5 × normalized_grip  → bonus morbido proporzionale all'intensità di chiusura

if gripper_action > 0.65 and is_physically_closed and dist_handle < 0.020:
    _grasp_confirm_count += 1   # accumula conferma grasp

if _grasp_confirm_count >= 5:   # TRANSIZIONE → PUSH
    _grasp_phase = True
    rew_info["phase_trans"] = _W_GRASP_BONUS  # = +50.0 (una-tantum)
    rew_info["base"] = base_reward  # rimuoviamo la penalità _W_GRIPPER_CLOSE
```

### Riepilogo Reward REACH

| Termine | Formula | Segno | Peso | Attivazione |
|---------|---------|-------|------|-------------|
| `base` | `base_reward - 2.5` | ± | - | Sempre |
| `dist_3d` | `-5.0 × dist` | − | 5.0 | Sempre |
| `dist_xy` | `-3.0 × dist_xy` | − | 3.0 | Sempre |
| `dist_z` | `-15.0 × |dZ|` | − | 15.0 | Sempre |
| `app_blw` | `-3.0 × (|dZ|-tol)` | − | 3.0 | `dZ < -0.005` |
| `app_top` | `-5.0 × dZ × grip` | − | 5.0 | `dZ > 0.03 AND grip > 0.2` |
| `align` | `-(1-align) × prox` | − | 1.0 | `dist < 3×grip_tol` |
| `flat` | `-0.5 × flat_align × prox` | − | 0.5 | `dist < 3×grip_tol` |
| `grip` | `-1.0 × (grip + 0.85)` | − | 1.0 | `dist>0.025 AND grip>-0.85` |
| `grip` | `2.5 × norm_grip` | + | 2.5 | `dist<=0.025 AND grip>-0.85` |
| `smoothness` | `-1.0 × jerk` | − | 1.0 | Sempre |
| `phase_trans` | `+50.0` | + | 50.0 | Una-tantum, transizione PUSH |

---

## FASE 2: PUSH (Chiusura della Porta)

**Condizione:** `_grasp_phase and not _success_latched`

### Reward Base

```python
rew_info["base"] = base_reward
# Senza modifiche: la base reward qui è positiva (progresso chiusura)
```

### Segnale di Distanza in PUSH

```python
rew_info["dist_3d"] = -5.0 × dist_handle  # mantieni il gripper sull'handle
rew_info["dist_z"]  = -15.0 × |height_diff|  # mantieni la quota corretta
```
Durante PUSH, il robot deve **seguire la maniglia** mentre la porta ruota.  
Questi segnali lo incentivano a non perdere il contatto.

### Reward Progressione Porta

```python
if self._min_door_angle is None:
    self._min_door_angle = door_angle  # inizializzazione al primo step di PUSH

if gripper_action > _GRIPPER_CLOSE_THRESH:  # gripper chiuso
    door_progress = self._min_door_angle - door_angle  # progresso fatto
    if door_progress > 0:
        rew_info["door_prog"] = _W_PROGRESS_GRASP × door_progress
        # = 2000.0 × door_progress (rad)
        self._min_door_angle = door_angle  # aggiorna minimo
```

Il peso **_W_PROGRESS_GRASP = 2000.0** è enorme: ogni radiante di chiusura vale 2000 punti.  
Questo è il reward dominante in PUSH e garantisce che la porta si chiuda.

**Il tracking del minimo** (`_min_door_angle`) evita il double-counting:  
premia solo la chiusura **incrementale** (distanza dal minimo precedente).

### Penalità Azione Braccio

```python
if action is not None:
    rew_info["act_pen"] = -_W_ACTION_PHASE2 × ‖action[:-1]‖
    # = -0.005 × norm_arm  (penalità minima, solo regolarizzazione)
```

### Penalità Lift del Braccio

```python
if action[2] > 0.05:  # azione positiva in Z = robot si alza
    rew_info["lift_pen"] = -2.0 × action[2]
```
Il robot non deve alzarsi mentre spinge (perderebbe il grip sulla maniglia che si abbassa).

### Logica di Perdita Grasp (con Tolleranza Dinamica)

```python
# Tolleranza dinamica basata su velocità angolare porta
door_speed = |prev_angle - door_angle| × control_freq  # rad/s
effective_lose_tol = clip(0.05 + door_speed × 0.5, 0.05, 0.12)

# Near-latch zone: molto più permissiva
if door_angle < 0.05:
    effective_lose_tol = 0.10  # override a 10 cm

# Controllo perdita gripper
gripper_action_lost = gripper_action < _GRIPPER_CLOSE_THRESH  # 0.65
if near_latch:
    gripper_lost = gripper_action_lost  # solo azione, non stato fisico
else:
    gripper_lost = gripper_action_lost or not is_physically_closed

distance_lost = dist_handle > effective_lose_tol
```

**Se grasp perso:**
```python
if distance_lost:
    rew_info["dist_lost"] = -_W_GRASP_LOST × (dist_handle - effective_lose_tol)
    # = -6.0 × (distanza - tolleranza)
    
if gripper_lost:
    rew_info["grip_lost"] = -5.0 × |min(0.0, gripper_action) - 0.65|

_grasp_phase = False   # torna a REACH
```

**Se grasp mantenuto:**
```python
self._grasp_lose_count = 0
if gripper_action < 1.0:  # non al massimo di chiusura
    rew_info["grip"] = -5.0 × (1.0 - gripper_action)
    # Penalità soft: incentiva a tenere il grip al massimo
```

### Riepilogo Reward PUSH

| Termine | Formula | Segno | Peso | Attivazione |
|---------|---------|-------|------|-------------|
| `base` | `base_reward` | ± | - | Sempre |
| `dist_3d` | `-5.0 × dist` | − | 5.0 | Sempre |
| `dist_z` | `-15.0 × |dZ|` | − | 15.0 | Sempre |
| `door_prog` | `2000.0 × Δangle` | + | 2000 | Gripper chiuso + progresso |
| `act_pen` | `-0.005 × ‖arm_act‖` | − | 0.005 | Sempre |
| `lift_pen` | `-2.0 × action[2]` | − | 2.0 | `action[2] > 0.05` |
| `dist_lost` | `-6.0 × (dist-tol)` | − | 6.0 | `distance_lost` |
| `grip_lost` | `-5.0 × |grip-0.65|` | − | 5.0 | `gripper_lost` |
| `grip` | `-5.0 × (1 - grip)` | − | 5.0 | Grasp ok + grip < 1.0 |
| `smoothness` | `-1.0 × jerk` | − | 1.0 | Sempre |

---

## FASE 3: HOLD (Mantenimento Chiusura)

**Condizione:** `_success_latched and not _ready_to_retreat`

### Reward Base

```python
rew_info["base"] = base_reward
rew_info["hold"] = 0.0  # inizializzato, poi accumulato
```

### Penalità per Perdita Presa Fisica

```python
is_waiting_latch = _hold_closed_duration >= int(control_freq × 2.0)  # 60 step

if not is_physically_closed and not is_waiting_latch:
    rew_info["hold_slip"] = -5.0
```
Penalità fissa -5.0 se il gripper non è fisicamente chiuso (non applicata durante il timeout latch).

### Penalità Velocità Angolare Porta (Damping)

```python
door_qvel = sim.data.qvel[hinge_dof_adr]  # velocità angolare porta (rad/s)
if |door_qvel| > 0.01:
    rew_info["hold_veldamp"] = -25.0 × |door_qvel|
```
Peso **-25.0 molto alto**: spinge il robot a **contrastare attivamente il bounce** della porta.  
Non basta che la porta sia ferma; il robot deve even diminuire la velocità angolare.

### Penalità Bounce (Soft Timer Reset)

```python
if not is_closed:  # porta si è aperta leggermente
    rew_info["hold_bounce"] = -20.0 × |door_qpos|
    # Penalità proporzionale all'apertura
    
    penalty_steps = int(|door_qpos| / 0.03 × 10)
    _hold_closed_duration = max(0, _hold_closed_duration - penalty_steps)
    # Soft reset: bounce piccolo → pochi step persi
```

### Reward per Porta Chiusa

```python
if is_closed:
    rew_info["hold"] += 1.0 - |door_qpos|
    # Premio continuo proporzionale a quanto è chiusa la porta
    # (massimo +1.0 se door_qpos=0, diminuisce se si apre)
```

### Sub-Fase HOLD Attiva (`|door_qpos| < 0.04`, timer in corso)

```python
if _hold_closed_duration < target_hold_steps:
    _hold_closed_duration += 1  # incrementa timer
    _ready_to_retreat = False   # blocca transizione
```

**Reward per Grip Corretto:**
```python
if gripper_action > _GRIPPER_CLOSE_THRESH:  # > 0.65
    rew_info["hold_grip"] = +1.0
else:
    rew_info["hold_grip"] = -2.0 × |gripper_action - 0.65|
```

**Penalità per Drop in HOLD:**
```python
if gripper_action < 0.0:  # gripper aperto durante HOLD
    rew_info["hold_drop_pen"] = -10.0 × |gripper_action|
```
Penalità alta per apertura del gripper: il robot non deve mollare la porta in HOLD.

**Penalità per Velocità Giunti:**
```python
joint_vel = obs["robot0_joint_vel"]
if joint_vel is not None:
    rew_info["hold_jnt_freeze"] = -1.0 × ‖joint_vel‖
```
Il braccio deve essere **fisicamente immobile** (velocità giunti = 0).  
L'override `action[:-1]=0.0` garantisce azioni zero, ma la dinamica fiscia può causare  
piccole velocità residue (compensazione gravità). Questa penalità le minimizza.

**Reward per Azione Braccio Zero:**
```python
action_norm = ‖action[:-1]‖
if action_norm < 0.05:
    rew_info["hold_act"] = +1.0   # robot fermo: bonus massimo
else:
    rew_info["hold_act"] = -2.0 × action_norm
```

**Penalità per Torsione Polso:**
```python
rew_info["hold_flat"] = -2.0 × flat_alignment
# flat_alignment = |eef_x[2]|: penalizza se il polso è ruotato verso l'alto
```

**Penalità per Distanza Handle:**
```python
if dist_handle > 0.06:  # EEF si è spostato > 6 cm dalla maniglia
    rew_info["hold_dist"] = -3.0 × (dist_handle - 0.06)
```

### Riepilogo Reward HOLD

| Termine | Formula | Segno | Peso | Attivazione |
|---------|---------|-------|------|-------------|
| `base` | `base_reward` | ± | - | Sempre |
| `hold` | `1.0 - |door_qpos|` | + | 1.0 | `is_closed` |
| `hold_slip` | `-5.0` | − | 5.0 | `!phys_closed AND !waiting` |
| `hold_veldamp` | `-25.0 × |door_qvel|` | − | 25.0 | `|door_qvel| > 0.01` |
| `hold_bounce` | `-20.0 × |door_qpos|` | − | 20.0 | `!is_closed` |
| `hold_grip` | `+1.0` | + | 1.0 | `grip > 0.65` |
| `hold_grip` | `-2.0 × |grip-0.65|` | − | 2.0 | `grip <= 0.65` |
| `hold_drop_pen` | `-10.0 × |grip|` | − | 10.0 | `grip < 0.0` |
| `hold_jnt_freeze` | `-1.0 × ‖joint_vel‖` | − | 1.0 | Sempre (in HOLD) |
| `hold_act` | `+1.0` | + | 1.0 | `action_norm < 0.05` |
| `hold_act` | `-2.0 × action_norm` | − | 2.0 | `action_norm >= 0.05` |
| `hold_flat` | `-2.0 × flat_align` | − | 2.0 | Sempre (in HOLD) |
| `hold_dist` | `-3.0 × (dist-0.06)` | − | 3.0 | `dist > 0.06` |
| `smoothness` | `-1.0 × jerk` | − | 1.0 | Sempre |

---

## FASE 4: RETREAT (Ritirata)

**Condizione:** `_success_latched and _ready_to_retreat`

### Reward per Apertura Gripper

```python
if gripper_action < _GRIPPER_OPEN_THRESH:  # < -0.85 (completamente aperto)
    rew_info["ret_grip"] = +2.0
else:
    rew_info["ret_grip"] = -1.0 × |gripper_action + 1.0|
```
In RETREAT il robot deve **aprire il gripper** per sfilarsi dalla maniglia.  
Il bonus +2.0 per gripper aperto è il segnale più forte in questa fase.

### Penalità Rotazione Polso

```python
rew_info["ret_rot"] = -3.0 × ‖action[3:6]‖
# Penalizza rotazioni del polso durante il ritiro
```
Il polso non deve ruotare durante il ritiro (rischio di incastrarsi nella maniglia).

### Penalità Movimento Laterale

```python
if dist_handle < 0.12:  # ancora vicino alla maniglia
    rew_info["ret_lat"] = -5.0 × |action[1]|  # no movimenti in Y
    if action[2] < 0:
        rew_info["ret_down"] = -5.0 × |action[2]|  # no movimenti verso il basso
```

### Reward Direzionale verso Retreat Target

```python
dist_to_target = ‖eef_pos - _retreat_pos‖

if dist_to_target > 0.02:
    dir_to_target    = (_retreat_pos - eef_pos) / (dist_to_target + 1e-6)
    action_alignment = dot(action[:3], dir_to_target)  # -1 a +1
    
    rew_info["ret_dir"]  = +3.0 × action_alignment
    # Premio proporzionale a quanto il movimento è nella giusta direzione
    
    perp = action[:3] - action_alignment × dir_to_target
    rew_info["ret_perp"] = -2.0 × ‖perp‖
    # Penalizza le componenti perpendicolari al target

else:  # arrivato al target (< 2 cm)
    rew_info["ret_freeze"] = -20.0 × ‖action[:-1]‖
    # Penalità alta per muoversi quando già al target → freeze
```

### Penalità Progressive sui Giunti (Freeze Progressivo)

```python
joint_vel = obs["robot0_joint_vel"]
if joint_vel is not None:
    freeze_weight = clip(1.0 - dist_to_target / 0.15, 0.1, 1.0)
    # freeze_weight: 0.1 quando lontano (15 cm), 1.0 quando vicino (0 cm)
    
    rew_info["ret_jnt_prog"] = -5.0 × freeze_weight × ‖joint_vel‖
```
Man mano che l'EEF si avvicina al target, la penalità per velocità giunti aumenta.  
Incoraggia il robot a **decelerare progressivamente** verso il punto di stop.

### Reward per Latch Neutro

```python
rew_info["latch_ret"] = -1.0 × |latch_qpos|
```
Penalizza il latch che non è tornato alla posizione neutra (0 rad).  
Nota: questo è solo monitoraggio — il latch torna autonomamente durante il ritiro.

### Riepilogo Reward RETREAT

| Termine | Formula | Segno | Peso | Attivazione |
|---------|---------|-------|------|-------------|
| `base` | `base_reward` | ± | - | Sempre |
| `hold` | `1.0 - |door_qpos|` | + | 1.0 | `is_closed` |
| `ret_grip` | `+2.0` | + | 2.0 | `grip < -0.85` |
| `ret_grip` | `-1.0 × |grip+1.0|` | − | 1.0 | `grip >= -0.85` |
| `ret_rot` | `-3.0 × ‖action[3:6]‖` | − | 3.0 | Sempre |
| `ret_lat` | `-5.0 × |action[1]|` | − | 5.0 | `dist<0.12` |
| `ret_down` | `-5.0 × |action[2]|` | − | 5.0 | `dist<0.12 AND action[2]<0` |
| `ret_dir` | `+3.0 × alignment` | + | 3.0 | `dist_to_target > 0.02` |
| `ret_perp` | `-2.0 × ‖perp‖` | − | 2.0 | `dist_to_target > 0.02` |
| `ret_freeze` | `-20.0 × ‖arm_act‖` | − | 20.0 | `dist_to_target <= 0.02` |
| `ret_jnt_prog` | `-5.0 × w × ‖joint_vel‖` | − | 5.0 | `joint_vel is not None` |
| `latch_ret` | `-1.0 × |latch_qpos|` | − | 1.0 | Sempre |
| `smoothness` | `-1.0 × jerk` | − | 1.0 | Sempre |

---

## Reward di Terminazione

### Terminazione Positiva (Success)

```python
# In RoboSuiteDoorCloseGymnasiumEnv._calculate_reward() (classe base):
if _return_hold >= return_hold_steps:  # 10 step al retreat_pos
    terminated = True
    reward += 500.0   # base success bonus
```

Il bonus +500.0 è la segnalazione finale. Nota: nella modalità diagnostica si vede `base: +500.93`  
(il 0.93 è il reward di base del singolo step sommato al bonus).

### Override Terminazione per Condizioni Non Soddisfatte

```python
# In GeneralizedDoorEnv._calculate_reward():
if terminated:
    latch_is_neutral = |latch_qpos| < 0.08   # latch < 4.6°
    door_is_closed   = |door_qpos| < 0.03    # porta < 1.7°
    
    if not latch_is_neutral or not door_is_closed:
        terminated = False          # ANNULLA terminazione
        reward -= 500.0             # rimuovi success bonus
        reward = clip(reward, -100.0, 100.0)
```

**Questo meccanismo critico** impedisce falsi positivi:  
se il robot raggiunge il retreat_pos ma la porta non è davvero chiusa o il latch non è neutro,  
l'episodio **non** termina con successo.

---

## Reward Base dalla Classe Padre

La chiamata `super()._calculate_reward(...)` in `GeneralizedDoorEnv` fornisce la base reward:

```python
# In RoboSuiteDoorCloseGymnasiumEnv._calculate_reward():
delta_close = max(0.0, prev_angle - door_angle)  # chiusura incrementale
progress    = 1.0 - (door_angle - door_min) / (door_max - door_min)  # progresso totale

reward  = w_progress × progress  # = 0.0 (peso zero in config generalized)
reward += w_delta × delta_close   # = 2.0 × delta_close
reward -= time_penalty            # = 0.5 per step
```

In pratica, con `w_progress=0.0`, il base reward segnala solo:
- **delta_close** × 2.0: quantità di porta chiusa in questo step
- **-0.5**: penalità temporale (incoraggia velocità)

---

## Evoluzione del Reward Totale per Fase (Valori Tipici Osservati)

| Fase | Reward Tipico per Step | Note |
|------|----------------------|------|
| REACH (lontano) | -3 a -8 | Dominato da dist_3d + dist_z |
| REACH (vicino + grip) | -1 a +3 | grip bonus + dist riduce |
| PUSH (progress) | +5 a +50 | door_prog domina |
| HOLD (stabile) | +1 a +3 | hold (+1) + hold_act (+1) + hold_grip (+1) |
| HOLD (bounce) | -10 a -50 | hold_veldamp e hold_bounce |
| RETREAT (verso target) | -1 a +2 | ret_dir bilancia ret_grip e penalità |
| Terminazione | +500 | One-shot bonus finale |

**Reward medio episodio convergenza:** 1050–1140  
(accumulo su ~110-120 step × reward medio positivo + bonus terminazione)
