# 🔄 Macchina a Stati Finiti (FSM) — GeneralizedDoorEnv

## Panoramica FSM

La FSM è il **cuore del controllo del comportamento** dell'agente. Coordina le 4 fasi del task,  
determinando quale segnale di reward fornire e come modificare le azioni prima dell'esecuzione.

```
┌─────────────────────────────────────────────────────────────────────┐
│                          FSM a 4 Stati                              │
│                                                                     │
│  ┌──────────┐    grasp         ┌──────────┐    door chiusa         │
│  │          │─────────────────►│          │────────────────────────►│
│  │ 1:REACH  │                  │  2:PUSH  │                         │
│  │          │◄─────────────────│          │◄───────────────────────┐│
│  └──────────┘    grasp lost    └──────────┘    grasp lost          ││
│                                                                     ││
│                         door chiusa (angle ≤ success_angle)         ││
│                         + gripper_action > 0.80                    ││
│                                    ↓                                ││
│  ┌──────────┐    timer OK     ┌──────────┐                         ││
│  │          │◄────────────────│          │                         ││
│  │ 4:RETREAT│                 │  3:HOLD  │─────────────────────────┘│
│  │          │                 │          │    bounce (door riaperta) │
│  └──────────┘                 └──────────┘                          │
│       │                                                             │
│       │ arrived at retreat_pos                                      │
│       │ (return_hold >= return_hold_steps)                          │
│       ▼                                                             │
│   TERMINATED (success = True)                                       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Variabili di Stato FSM

| Variabile | Tipo | Default | Descrizione |
|-----------|------|---------|-------------|
| `_grasp_phase` | bool | False | True se il robot ha afferrato la maniglia (Fase 2 attiva) |
| `_success_latched` | bool | False | True se la porta è stata chiusa con grip saldo (Fase 3/4) |
| `_ready_to_retreat` | bool | False | True se il hold timer è scaduto (Fase 4 attiva) |
| `_grasp_confirm_count` | int | 0 | Step consecutivi con grasp valido (soglia: 5) |
| `_grasp_lose_count` | int | 0 | Step consecutivi con grasp perso (non usato attivamente) |
| `_hold_closed_duration` | int | 0 | Step accumulati con porta chiusa in HOLD |
| `_return_hold` | int | 0 | Step con EEF al retreat_pos (termination counter) |
| `_prev_door_angle` | float/None | None | Angolo porta step precedente (per calcolo velocità) |
| `_min_door_angle` | float/None | None | Angolo minimo raggiunto in PUSH (tracking progresso) |
| `_retreat_pos` | np.array | None | Posizione target di retreat [EEF_at_hold + [-0.13, 0, 0.04]] |
| `_has_received_grasp_bonus` | bool | False | One-shot flag per bonus REACH→PUSH |
| `_fsm_events` | list[str] | [] | Log eventi FSM per diagnostica |

### Encoding One-Hot nell'Osservazione

```python
fsm_reach   = 1.0 if (not _grasp_phase and not _success_latched) else 0.0
fsm_push    = 1.0 if (_grasp_phase and not _success_latched)     else 0.0
fsm_hold    = 1.0 if (_success_latched and not _ready_to_retreat) else 0.0
fsm_retreat = 1.0 if (_success_latched and _ready_to_retreat)    else 0.0
```

---

## Stato 1: REACH (Raggiungimento Maniglia)

### Condizione di Ingresso
- Stato iniziale ad ogni episodio (reset)
- Ritorno da PUSH (grasp perso)

### Condizione di Uscita (→ PUSH)
Devono essere verificate contemporaneamente per **5 step consecutivi**:
1. `gripper_action > _GRIPPER_CLOSE_THRESH (0.65)` — il gripper sta stringendo
2. `is_physically_closed = True` — il gripper è fisicamente chiuso sull'handle:
   ```python
   gripper_width <= handle_diameter + 0.025  AND  gripper_width >= 0.015
   # dove handle_diameter = 2 × _current_handle_radius
   ```
3. `dist_handle < 0.020` — l'EEF è a meno di 2 cm dall'handle

```python
if gripper_action > _GRIPPER_CLOSE_THRESH and is_physically_closed and dist_handle < 0.020:
    self._grasp_confirm_count += 1
else:
    self._grasp_confirm_count = 0  # reset se anche un solo step fallisce

if self._grasp_confirm_count >= 5:
    self._grasp_phase = True
    self._grasp_lose_count = 0
    # → bonus una-tantum _W_GRASP_BONUS = 50.0
```

### Azione Override in REACH
- Nessun override (policy libera)

### Comportamento Previsto
Il robot deve:
1. Navigare verso la maniglia (reward inversamente proporzionale alla distanza)
2. Approcciare dall'alto (penalità se si avvicina da sotto o troppo in alto)
3. Tenere il gripper **aperto** durante l'avvicinamento lontano
4. Chiudere il gripper quando vicino

---

## Stato 2: PUSH (Chiusura della Porta)

### Condizione di Ingresso
- Da REACH: `_grasp_confirm_count >= 5`

### Condizione di Uscita (→ HOLD)
```python
if door_angle <= self._success_angle and not self._success_latched:
    if action[-1] > 0.80:   # gripper fortemente chiuso alla transizione
        self._success_latched = True
        just_succeeded = True
```
La transizione richiede simultaneamente:
1. `door_angle <= _success_angle` (1.5% del range angolare → quasi completamente chiusa)
2. `action[-1] > 0.80` (gripper all'80% di chiusura massima)

La seconda condizione **previene lo slip** alla transizione: evita che il robot entri in HOLD con grip incerto.

### Condizione di Uscita (→ REACH — grasp lost)
```python
effective_lose_tol = clip(0.05 + door_speed * 0.5, 0.05, 0.12)
# in zona near-latch (door_angle < 0.05): tol = 0.10 (più permissiva)

gripper_lost  = (gripper_action < _GRIPPER_CLOSE_THRESH) or (not is_physically_closed)
# in near-latch: gripper_lost = (gripper_action < _GRIPPER_CLOSE_THRESH) solo

distance_lost = dist_handle > effective_lose_tol

if gripper_lost or distance_lost:
    self._grasp_phase = False
    self._grasp_confirm_count = 0
    # → torna a REACH
```

**Tolleranza dinamica:** La soglia di distanza si espande con la velocità angolare della porta:
- Porta ferma: tol = 0.05 m (5 cm)
- Porta veloce: tol fino a 0.12 m (12 cm), per non droppare mentre la maniglia si sposta sull'arco

**Near-latch zone** (`door_angle < 0.05` rad):
- `is_physically_closed` diventa inaffidabile (maniglia in moto)
- Tolleranza distanza = 0.10 m (10 cm)
- Check grip solo su `gripper_action` (non su stato fisico)

### Azione Override in PUSH
- Nessun override completo (policy libera)
- Solo penalty sull'azione nel reward (piccola)

### Comportamento Previsto
Il robot deve:
1. Mantenere il grip (gripper chiuso, EEF vicino alla maniglia)
2. Spingere la porta verso la posizione chiusa
3. Seguire l'arco della maniglia mentre la porta ruota
4. Non alzare il braccio (penalità se `action[2] > 0.05`)

---

## Stato 3: HOLD (Mantenimento Chiusura)

### Condizione di Ingresso
- Da PUSH: porta chiusa + gripper saldo

### Condizione di Uscita (→ RETREAT)
Timer basato su step con porta chiusa:

```python
target_hold_steps = int(control_freq * 2.0)  # 30 × 2.0 = 60 step = 2.0 secondi

if self._hold_closed_duration >= target_hold_steps:
    self._ready_to_retreat = True
    self._retreat_pos = eef_pos + np.array([-0.13, 0.0, 0.04])
    # retreat target: 13 cm indietro, 4 cm su
```

Il timer avanza solo quando `|door_qpos| < 0.04` rad (porta effettivamente chiusa).  
In caso di bounce (porta si riapre leggermente):
```python
penalty_steps = int(abs(door_qpos) / 0.03 * 10)  # proporzionale al bounce
self._hold_closed_duration = max(0, self._hold_closed_duration - penalty_steps)
```
Soft reset: piccoli bounce perdono pochi step, bounce grandi azzerano quasi tutto.

### Azione Override in HOLD
```python
# In GeneralizedDoorEnv.step() (NON in _calculate_reward):
if self._success_latched:
    is_ready_retreat = getattr(self, "_ready_to_retreat", False)
    if not is_ready_retreat:
        action[:-1] = 0.0   # braccio completamente FERMO
        # action[-1] rimane invariata (gripper continua a stringere)
```

**Questo è il fix chiave documentato in `docs_fin.md`:**  
Congela completamente il braccio per eliminare oscillazioni/vibrazioni del polso durante HOLD.

### Comportamento Previsto
- Braccio completamente fermo (action_norm = 0 garantito dall'override)
- Gripper mantiene la presa (reward positivo se `gripper_action > 0.65`)
- 2 secondi di attesa per permettere al latch di stabilizzarsi

---

## Stato 4: RETREAT (Ritirata)

### Condizione di Ingresso
- Da HOLD: `_hold_closed_duration >= 60` step

### Condizione di Uscita (→ TERMINATED)
```python
if dist_retreat < return_pos_tol  OR  _return_hold >= return_hold_steps:
    action = np.zeros_like(action)  # Freeze totale
    # → terminated = True (dalla base class), reward += 500
```

dove `return_pos_tol = 0.05 m (5 cm)` e `return_hold_steps = 10`.

### Target di Retreat
```python
self._retreat_pos = eef_pos_at_hold_end + np.array([-0.13, 0.0, 0.04])
# 13 cm indietro (X), 0 laterale (Y), 4 cm su (Z)
```

**Importanza cinematica:** Il movimento di ritiro (-13 cm in X) è **esattamente ciò che sblocca il latch**.  
I test di ablazione (`scratch/`) dimostrano che congelando il braccio in RETREAT, il latch non torna neutro.

### Azione Override in RETREAT
```python
if is_ready_retreat:
    if dist_retreat < return_pos_tol or _return_hold >= return_hold_steps:
        action = np.zeros_like(action)  # Freeze: raggiunto il target
    else:
        pass  # Nessun override: scala 1.0, policy libera di muoversi
```

### Comportamento Previsto
- Gripper deve aprirsi (`action[-1] < -0.85` per reward massimo)
- Braccio si muove verso `_retreat_pos` (-13 cm in X, +4 cm in Z)
- Minima rotazione del polso (reward penalizza `‖action[3:6]‖`)
- Nessun movimento laterale eccessivo (penalità su `|action[1]|`)
- Nessun movimento verso il basso (penalità se `action[2] < 0`)

---

## Diagramma Dettagliato delle Transizioni

```
RESET
  │
  ▼
[1:REACH] _grasp_phase=False, _success_latched=False
  │
  │  Condizione: _grasp_confirm_count >= 5
  │  (5 step consecutivi: dist<2cm, grip>0.65, phys_closed)
  ├─────────────────────────────────────────────────────► [2:PUSH]
  │
  │  Return: _grasp_phase=False, _grasp_confirm_count=0
  ◄─────────────────────────────────────────────────────┤
                                                         │
[2:PUSH] _grasp_phase=True, _success_latched=False       │
  │                                                      │
  │  Grasp lost: (gripper_action<0.65 OR dist>tol)       │
  ├────────────────────────────────────────────────────► ┘
  │
  │  Condizione: door_angle<=success_angle AND action[-1]>0.80
  ├─────────────────────────────────────────────────────► [3:HOLD]
  │
[3:HOLD] _success_latched=True, _ready_to_retreat=False
  │  action[:-1] = 0.0 (override braccio)
  │
  │  Timer: _hold_closed_duration >= 60 step (2.0s @ 30Hz)
  │  (con soft-reset in caso di bounce)
  ├─────────────────────────────────────────────────────► [4:RETREAT]
  │
[4:RETREAT] _success_latched=True, _ready_to_retreat=True
  │
  │  Condizione: dist_to_retreat_pos < 0.05 m
  │  OR _return_hold >= 10 step
  ├─────────────────────────────────────────────────────► TERMINATED
                                                           (success=True)
                                                           reward += 500
```

---

## Logging Diagnostico

Ogni 200 step, viene stampata una tabella di stato:

```
┌─────────┬────────┬────────┬───────┬───────────┬───────┬───────┬───────┬───────┐
│  PHASE  │  DIST  │   dZ   │ GRIP  │   PHYS    │ WIDTH │ ALIGN │ DOOR  │ LATCH │
├─────────┼────────┼────────┼───────┼───────────┼───────┼───────┼───────┼───────┤
│ 2:PUSH  │  0.018 │ +0.002 │ +0.92 │ PHYS_OK   │ 0.047 │  0.89 │  0.23 │ +1.31 │
└─────────┴────────┴────────┴───────┴───────────┴───────┴───────┴───────┴───────┘
  ↳ REWARDS │ base: +0.10 │ dist_3d: -0.09 │ door_prog: +2000×0.001 │ TOT: +3.45
```

| Campo | Fonte |
|-------|-------|
| PHASE | `_grasp_phase` + `_success_latched` + `_ready_to_retreat` |
| DIST | `‖eef_pos - handle_pos‖` |
| dZ | `eef_pos[2] - handle_pos[2]` |
| GRIP | `action[-1]` |
| PHYS | `is_physically_closed` (OK/OPEN) |
| WIDTH | `sum(‖gripper_qpos‖)` |
| ALIGN | `dot(eef_z, dir_to_handle)` |
| DOOR | `door_angle` (rad) |
| LATCH | `latch_qpos` (rad) |

---

## Fisica del Latch (MuJoCo)

Il latch joint (`Door_latch_joint`) ha:
- **Stiffness**: costante elastica che lo riporta alla posizione neutra
- **Damping**: smorzamento

**Comportamento:**
1. Durante PUSH: il gripper ruota la maniglia, portando `latch_qpos` a ~1.2–1.5 rad
2. Durante HOLD: il latch è tenuto ruotato dal gripper chiuso
3. Durante RETREAT: il movimento del braccio (-13 cm) sfila il gripper dalla maniglia
4. Dopo RETREAT: il latch torna autonomamente a ~0.0 rad (in ~22 step) grazie alla molla

**Test T1** (da `diag_phase34.py`): conferma che la molla riporta il latch a < 0.1 rad in ~22-90 step (dipende da stiffness/damping specifici).

---

## Confronto FSM Base vs Generalizzata

| Aspetto | `RoboSuiteDoorCloseGymnasiumEnv` (base) | `GeneralizedDoorEnv` (generalizzato) |
|---------|----------------------------------------|--------------------------------------|
| Fasi | 2 (Close + Return) | 4 (REACH, PUSH, HOLD, RETREAT) |
| Grasp tracking | No | Sì (`_grasp_phase`, `_grasp_confirm_count`) |
| HOLD action override | `action[:-1] *= 0.2` (soft) | `action[:-1] = 0.0` (hard freeze) |
| Transizione PUSH→HOLD | Solo angolo porta | Angolo + gripper > 0.80 |
| Bounce detection | No | Sì (soft timer reset) |
| FSM in osservazione | No | Sì (4 one-hot bits) |
| Logging diagnostico | No | Sì (ogni 200 step) |
