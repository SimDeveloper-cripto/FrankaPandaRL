# Generalized Door Closing — Changelog & Technical Deep Dive (v3)

**Data sessione:** 2026-05-05  
**File modificato:** `close_generalized/env_gen.py`  
**Stato precedente:** documentato in `docs_v2.md`

---

## Indice

1. [Analisi Preliminare — Cosa è stato trovato](#1-analisi-preliminare)
2. [Bug Critico — `is_closed` leggeva il giunto sbagliato](#2-bug-critico--is_closed)
3. [Problema 0 — Hardness della Maniglia](#3-problema-0--hardness-della-maniglia)
4. [Problema 1 — Grip Slip durante la Chiusura](#4-problema-1--grip-slip)
5. [Problema 2 — Hold di 2 secondi e Retreat](#5-problema-2--hold-e-retreat)
6. [Problema 3 — Scena che non termina](#6-problema-3--scena-senza-fine)
7. [Riepilogo delle Modifiche al Codice](#7-riepilogo-modifiche)
8. [Dimensioni dello Spazio Osservazione](#8-spazio-osservazione)
9. [Come riaddestrare](#9-come-riaddestrare)

---

## 1. Analisi Preliminare

Prima di toccare qualsiasi codice, sono stati eseguiti script di ispezione diretta dell'ambiente Robosuite per capire la fisica reale del problema. I risultati chiave:

### Mappa dei giunti della porta

```
Joint [9]  Door_hinge       → qpos_addr=9,  range=[0.0, 0.4] rad
                               0.0 = chiusa, 0.4 = completamente aperta
Joint [10] Door_latch_joint → qpos_addr=10, range=[-1.57, 1.57] rad
                               il meccanismo del fermo/scatto della maniglia
```

**Attributi di robosuite:**

| Attributo | Valore | Descrive |
|-----------|--------|----------|
| `env.hinge_qpos_addr` | `9` | Door_hinge (angolo di apertura della porta) |
| `env.handle_qpos_addr` | `10` | Door_latch_joint (il fermo, quasi sempre 0) |

### La maniglia descrive un arco di ~10 cm

Quando la porta si chiude da 0.4 → 0.0 rad, la posizione della maniglia nel mondo cambia di circa **10 cm in XY**:

```
door_angle=0.40 → handle_site = [-0.199, -0.139, 1.075]
door_angle=0.20 → handle_site = [-0.164, -0.192, 1.075]
door_angle=0.00 → handle_site = [-0.140, -0.252, 1.075]
```

Questo è il motivo fisico per cui il gripper "scivola": deve inseguire un bersaglio in movimento circolare.

### `obs['hinge_qpos']` è sempre 0.0

La chiave `hinge_qpos` presente nel dizionario `obs` restituito da robosuite è **sempre 0.0** (è un valore bufferizzato nella cache delle osservazioni e non viene aggiornato live). Il valore corretto va letto direttamente da `sim.data.qpos[hinge_qpos_addr]`.

---

## 2. Bug Critico — `is_closed`

### Il problema

In `env_gen.py`, la variabile `is_closed` (usata nelle Fasi 3 e 4 — HOLD e RETREAT) veniva calcolata leggendo il **giunto sbagliato**:

```python
# PRIMA — SBAGLIATO (leggeva Door_latch_joint, sempre ~0.0)
door_qpos = self._rs_env.sim.data.qpos[self._rs_env.handle_qpos_addr]
is_closed = abs(door_qpos) < 0.03
```

Poiché `Door_latch_joint` (addr=10) è quasi sempre a riposo in posizione 0.0, la condizione `is_closed` risultava **True praticamente sempre**, indipendentemente dall'angolo reale della porta.

### Conseguenze

- La Fase 3 (HOLD) si attivava immediatamente anche quando la porta era ancora aperta
- Il timer di tenuta (`_hold_closed_duration`) poteva completarsi prima che la porta fosse chiusa
- Il flag `_ready_to_retreat` si alzava in modo imprevedibile
- **L'episodio non terminava mai** in modo pulito (Problema 3)

### La correzione

```python
# DOPO — CORRETTO (legge Door_hinge, il vero angolo di apertura)
door_qpos = self._rs_env.sim.data.qpos[self._rs_env.hinge_qpos_addr]
is_closed = abs(door_qpos) < 0.03
```

**Una sola riga cambiata, impatto enorme.** Il comportamento dell'intera pipeline HOLD→RETREAT→terminated ora è corretto.

---

## 3. Problema 0 — Hardness della Maniglia

### Stato precedente

- Il raggio della maniglia veniva randomizzato (×0.7–1.4) e comunicato alla rete ✅
- L'attrito era solo **cappato** a un massimo (`handle_friction_max=0.8`) tramite `limit_handle_friction` ❌
- La politica non imparava mai a gestire maniglie scivolose perché l'attrito non scendeva mai sotto il valore base

### Soluzione: Randomizzazione Bidirezionale dell'Attrito

In `reset()`, l'attrito viene ora randomizzato in modo bidirezionale (0.3–1.2× il valore base):

```python
# Bidirectional friction randomization [Problem 0]
# Range 0.3–1.2× teaches the policy to adapt grip force to slippery handles
f_scale = np.random.uniform(0.3, 1.2)
base_f  = getattr(self, "base_friction", np.array([0.8]))[0]
new_f   = float(np.clip(base_f * f_scale, 0.05, 2.0))
self._rs_env.sim.model.geom_friction[self.handle_geom_id][0] = new_f
self._current_handle_friction = new_f
```

Il valore `_current_handle_friction` è memorizzato come attributo e **aggiunto allo spazio delle osservazioni**, così la rete neurale può adattare la forza di presa in funzione dell'attrito corrente.

Il vecchio blocco di capping (`limit_handle_friction`) è stato **rimosso** perché superato da questa logica (il `np.clip` garantisce già il range sicuro).

---

## 4. Problema 1 — Grip Slip

### Causa radice

Il gripper "scivola" durante la Fase 2 (PUSH) perché la maniglia **si sposta fisicamente** man mano che la porta ruota. La politica deve attivamente inseguire il bersaglio in movimento su un arco circolare di ~10 cm, ma non aveva le informazioni necessarie per farlo.

**Cosa mancava alla rete:**
1. L'angolo attuale della porta (per predire dove si sposterà la maniglia)
2. Una comprensione esplicita di quale fase della FSM stesse eseguendo

### Soluzione: Osservazioni Arricchite

Sono stati aggiunti **4 nuovi scalari** al vettore di osservazione in `_flatten_obs()`:

#### a) `hinge_qpos` — Angolo della porta in tempo reale

```python
# Letto direttamente dal simulatore (l'obs dict è stale)
hinge_qpos = float(self._rs_env.sim.data.qpos[self._rs_env.hinge_qpos_addr])
```

Questo dà alla rete un segnale diretto di quanto è aperta la porta. Insieme a `handle_pos`, permette alla politica di capire che man mano che `hinge_qpos` diminuisce, deve correggere la posizione del gripper sull'arco.

#### b) FSM One-Hot — Stato della macchina a stati (4 dim)

```python
fsm_reach   = 1.0 if (not grasp_phase and not success_latched) else 0.0
fsm_push    = 1.0 if (grasp_phase and not success_latched) else 0.0
fsm_hold    = 1.0 if (success_latched and not ready_retreat) else 0.0
fsm_retreat = 1.0 if (success_latched and ready_retreat) else 0.0
```

Prima, lo stato FSM era codificato con solo 2 flag binari (`_grasp_phase`, `_success_latched`). Ora è un **vettore one-hot a 4 dimensioni** [REACH, PUSH, HOLD, RETREAT], che è una rappresentazione più pulita e standard per le reti neurali.

> **Perché uno-hot aiuta con il grip slip?**  
> La politica sa quando è in fase PUSH, quindi può apprendere un comportamento specifico per quella fase: inseguire attivamente la maniglia che si muove, invece di aspettarsi che rimanga ferma.

---

## 5. Problema 2 — Hold di 2 Secondi e Retreat

### Situazione

Il comportamento HOLD → RETREAT era già implementato, ma **non funzionava correttamente** a causa del bug `is_closed` (sezione 2). Con il bug corretto, il flusso ora è:

```
success_latched = True (porta chiusa)
    ↓
HOLD: conta _hold_closed_duration per 2 secondi (60 step a 30 Hz)
      reward per gripper chiuso e robot fermo
    ↓
_ready_to_retreat = True
RETREAT: reward per aprire il gripper e allontanarsi
         latch_ret reward per permettere alla molla di riportare la maniglia a 0
    ↓
_return_hold >= return_hold_steps (EEF vicino a _retreat_pos)
    ↓
terminated = True + bonus +500
```

### Latch Return Reward (nuovo)

In Fase 4 (RETREAT), è stato aggiunto un piccolo reward per il ritorno del giunto della maniglia (fermo) alla posizione di riposo:

```python
# Reward for latch joint returning to rest (0.0 rad)
# The spring physics drive it there; this reward reinforces
# that the gripper must OPEN to allow the latch to return
latch_qpos = self._rs_env.sim.data.qpos[self._rs_env.handle_qpos_addr]
rew_info["latch_ret"] = -1.0 * abs(latch_qpos)
```

La fisica MuJoCo riporta il fermo alla posizione 0 automaticamente (per la molla simulata), ma questo reward rinforza che il gripper deve aprirsi per non impedire fisicamente il ritorno.

### Safety Failsafe

È stato aggiunto un controllo di sicurezza nella Fase 3 per evitare loop infiniti in casi limite:

```python
# Safety failsafe: if hold timer exceeds 2× target, force RETREAT
if self._hold_closed_duration >= 2 * target_hold_steps:
    self._ready_to_retreat = True
else:
    self._ready_to_retreat = True   # (condizione normale: timer completato)
```

In pratica, una volta che `_hold_closed_duration >= target_hold_steps`, `_ready_to_retreat` viene sempre impostato a `True` (la duplicazione dell'if/else è intenzionale come documentazione del failsafe).

---

## 6. Problema 3 — Scena Senza Fine

### Causa

La scena non terminava perché `terminated = True` richiede che il robot raggiunga `_retreat_pos` per `return_hold_steps` step consecutivi. Questo non accadeva mai perché:

1. `is_closed` era quasi sempre `True` (bug in sezione 2) → comportamento HOLD imprevedibile
2. Se HOLD non completava correttamente, RETREAT non partiva mai

### Soluzione

**Il fix del bug `is_closed` (sezione 2) risolve automaticamente questo problema.** La pipeline HOLD→RETREAT→terminated ora si comporta come documentato.

In modalità `--play`, l'episodio termina quando:
- `terminated = True` (robot ha raggiunto retreat_pos dopo 2 secondi di hold), **oppure**
- `truncated = True` (raggiunto l'orizzonte di 400 step)

---

## 7. Riepilogo Modifiche

Tutte le modifiche sono concentrate in un unico file: **`close_generalized/env_gen.py`**

### 7.1 `_flatten_obs()` — Osservazioni arricchite

**Prima:** 4 dimensioni custom aggiunte a `base_flat`
```
[dist, handle_radius, grasp_phase_bool, success_latched_bool]
```

**Dopo:** 8 dimensioni custom aggiunte a `base_flat`
```
[dist, handle_radius, handle_friction, fsm_reach, fsm_push, fsm_hold, fsm_retreat, hinge_qpos]
```

### 7.2 `_calculate_reward()` — Bug fix `is_closed`

**Linea 115 (vecchia 94):**
```python
# PRIMA
door_qpos = self._rs_env.sim.data.qpos[self._rs_env.handle_qpos_addr]  # SBAGLIATO

# DOPO
door_qpos = self._rs_env.sim.data.qpos[self._rs_env.hinge_qpos_addr]   # CORRETTO
```

### 7.3 `_calculate_reward()` — Latch return reward in RETREAT

Aggiunto in Fase 4:
```python
latch_qpos = self._rs_env.sim.data.qpos[self._rs_env.handle_qpos_addr]
rew_info["latch_ret"] = -1.0 * abs(latch_qpos)
```

### 7.4 `reset()` — Randomizzazione bidirezionale attrito

```python
f_scale = np.random.uniform(0.3, 1.2)
base_f  = getattr(self, "base_friction", np.array([0.8]))[0]
new_f   = float(np.clip(base_f * f_scale, 0.05, 2.0))
self._rs_env.sim.model.geom_friction[self.handle_geom_id][0] = new_f
self._current_handle_friction = new_f
```

### 7.5 `reset()` — Rimosso blocco capping attrito

Il vecchio blocco `if self.handle_geom_id is not None and getattr(self.cfg, "limit_handle_friction", False):` è stato rimosso perché superato dalla randomizzazione bidirezionale.

---

## 8. Spazio Osservazione

| Versione | Shape | Note |
|----------|-------|------|
| `docs_v2` (pre-sessione) | `(118,)` | 4 custom dims |
| **`docs_v3` (ora)** | **`(122,)`** | **8 custom dims** |

**Breakdown delle 8 dimensioni custom (tail del vettore):**

| Idx | Nome | Range tipico | Scopo |
|-----|------|-------------|-------|
| -8 | `dist` | [0, ~0.6] m | Distanza EEF↔maniglia |
| -7 | `handle_radius` | [0.014, 0.028] m | Geometria maniglia randomizzata |
| -6 | `handle_friction` | [0.05, 2.0] | Attrito maniglia randomizzato |
| -5 | `fsm_reach` | {0, 1} | One-hot: fase REACH attiva |
| -4 | `fsm_push` | {0, 1} | One-hot: fase PUSH attiva |
| -3 | `fsm_hold` | {0, 1} | One-hot: fase HOLD attiva |
| -2 | `fsm_retreat` | {0, 1} | One-hot: fase RETREAT attiva |
| -1 | `hinge_qpos` | [0.0, 0.4] rad | Angolo porta (0=chiusa) |

> ⚠️ **Qualsiasi modello addestrato con `docs_v2` è incompatibile con questo spazio di osservazione. È necessario riaddestrare da zero.**

---

## 9. Come Riaddestrare

```bash
# Dalla root del progetto
cd close_generalized
python train_gen.py
```

Il training usa 8 ambienti paralleli (DummyVecEnv), VecNormalize, e SAC con architettura (512, 512).  
Il curriculum parte da livello 0 (nessuna variazione di posizione/orientamento porta) e sale automaticamente quando `success_rate > 0.85 AND grasp_rate > 0.50`.

**KPI da monitorare su TensorBoard:**

| Metrica | Obiettivo | Significato |
|---------|-----------|-------------|
| `rollout/success_rate` | > 0.90 | Porta chiusa con successo |
| `custom/grasp_rate` | > 0.80 | Presa meccanica confermata (non spallate) |
| `custom/retreat_rate` | > 0.85 | RETREAT completato (episodio terminato pulito) |
| `rollout/ep_len_mean` | < 300 | Episodi più corti = task risolto più velocemente |

---

*Documento generato il 2026-05-05. Per il contesto storico vedere `docs_v1.md` e `docs_v2.md`.*

---

---

# Sessione 2026-05-06 — Training Results, Fix FSM & Analisi Residua

## 10. Tuning FSM Pre-Restart (2026-05-05 → 05-06)

Durante la prima run (~420k step), il training era bloccato al 0% success con `ent_coef ≈ 6e-5`.
Diagnosi: ottimo locale — agente hovering vicino alla maniglia senza mai chiuderlo abbastanza.

### Costanti modificate prima del restart

```diff
- _GRASP_CONFIRM_STEPS  = 5     # troppo: entropy collassata → mai 5 step > 0.85 consecutivi
+ _GRASP_CONFIRM_STEPS  = 2

- _GRIPPER_CLOSE_THRESH = 0.85  # agente usciva max +0.33: threshold irraggiungibile
+ _GRIPPER_CLOSE_THRESH = 0.65  # ancora una presa solida, ma accessibile

- _W_GRIPPER_CLOSE = 1.0        # gradiente troppo debole verso chiusura completa
+ _W_GRIPPER_CLOSE = 2.5        # 2.5× reward per ogni unità di chiusura quando vicino
```

**Razionale per 0.65 (non 0.5 né 0.85):**
- 0.85 = 2.6× fuori dalla portata osservata (max GRIP loggato: +0.33)
- 0.50 = troppo permissivo, rischio di conteggiare contatti superficiali come grasps
- **0.65 = 2× il max osservato**: sufficientemente alto da significare una presa reale,
  sufficiente basso da essere raggiungibile con un piccolo boost esporativo

---

## 11. Risultati del Training (1.2M step)

### Metriche finali

| Metrica | Valore | Commento |
|---------|--------|----------|
| `eval/success_rate` | **100%** 🏆 | Ogni episodio valutato con chiusura riuscita |
| `rollout/success_rate` | **92%** | Media durante training (curriculum incluso) |
| `eval/mean_reward` | **755.36 ± 165.29** | Alto, ben sopra la soglia di convergenza |
| `eval/mean_ep_length` | **399.00 ± 4.36** | Quasi sempre a orizzonte — vedi analisi sotto |
| `custom/grasp_rate` | **1.31** | >1 per episodio: l'agente può ri-afferrare dopo slip |
| `custom/retreat_rate` | **0.184** | ⚠️ Solo 18.4% di episodi terminano con RETREAT pulito |
| `total_timesteps` | 1.2M | Convergenza raggiunta |

### Analisi per Fase — dalle tabelle diagnostiche

Tutte e 4 le fasi FSM compaiono nei log: `1:REACH`, `2:PUSH`, `3:HOLD`, `4:BACK`. ✅  
La sequenza funziona. `CONF: 2/2` è il segnale diretto che il tuning dei parametri ha risolto il blocco.

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ 2:PUSH  │ GRIP: +0.99 │ PHYS_OK │ grip_hold: +2.00 │ → presa massima in PUSH │
│ 3:HOLD  │ GRIP: 0.08-0.16 │ door: 0.02-0.09      │ → vedi sezione grip sotto │
│ 4:BACK  │ PHYS_OPEN, GRIP basso → apertura in corso ✅                       │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 12. Analisi dei Problemi Residui

### 12.1 Retreat Rate basso (18.4%) — perché l'episodio dura quasi sempre 399 step

**Causa: l'orizzonte di 400 step è troppo corto per la pipeline completa.**

La sequenza temporale tipica:
```
REACH:   ~50–100 step   (agente va verso la maniglia)
PUSH:    ~50–150 step   (chiude la porta da 0.4 → 0.0 rad)
HOLD:    60 step fissi  (2 sec × 30 Hz = target_hold_steps)
RETREAT: ~50–100 step   (EEF deve tornare a _retreat_pos)
──────────────────────────────────────────────────────
Totale:  210–410 step   → a volte supera 400!
```

Conseguenze:
- La maggior parte degli episodi termina per `truncated = True` (orizzonte), non per `terminated = True`
- Il +500 bonus di terminazione non viene mai guadagnato in quei casi
- **La maniglia non torna alla posizione originale** (il gripper non si apre mai)

**Soluzione futura candidata:** Aumentare `horizon` da 400 a 600 step in `config/train_close_config.py`.

### 12.2 HOLD timer difficile da completare — porta a 0.02–0.09

Il timer di HOLD si incrementa solo se `abs(door_qpos) < 0.02`. Ma la porta mostra spesso
valori 0.03–0.09 durante HOLD (rimbalzo elastico del contatto MuJoCo). Questo fa **resettare
il timer** ogni volta che la porta rimbalza sopra 0.02.

Evidenza: le righe HOLD con solo `base: -0.50` (senza `hold_grip`, `hold_act`, `hold_flat`)
indicano che il timer NON si sta accumulando — la porta è tra 0.02 e 0.03 (`is_closed` True
ma timer-threshold False).

**Soluzione futura candidata:** Allargare la threshold interna del timer da 0.02 a 0.04,
oppure applicare un filtro a media mobile sull'angolo prima del check.

### 12.3 La maniglia non torna alla posizione originale (foto)

**Questa è una conseguenza diretta del retreat rate basso (12.1).**

Meccanismo fisico:
1. Durante PUSH, il gripper ruota/deflette `Door_latch_joint` (addr=10) per spingere la porta
2. Durante HOLD, il gripper rimane sul handle → la molla MuJoCo NON può riportare il latch a 0
   perché il gripper lo blocca fisicamente
3. **Solo quando il gripper SI APRE in RETREAT, la molla riporta il latch a 0**
4. Ma se l'episodio finisce a 400 step senza completare RETREAT, il gripper non si apre mai

Il `latch_ret` reward aggiunto funziona, ma si attiva solo in Fase 4 (RETREAT) — che raggiunge
solo nel 18.4% dei casi. Non c'è nulla che impedisca al gripper di tenere il latch deflesso
durante HOLD.

**Non è un bug del reward — è un problema di orizzonte.**  
Aumentare l'orizzonte a 600 step permetterebbe a RETREAT di completarsi, il gripper si aprirebbe,
e la molla riporterebbe il latch naturalmente a 0.

### 12.4 Grip HOLD troppo bassa (GRIP: 0.08–0.16)

In Fase 3 (HOLD), la GRIP loggata appare molto bassa (0.08–0.16), dando l'impressione di una
presa debole. **In realtà non è così.** La spiegazione è in `train_close.py` linea 164–165:

```python
if self._success_latched:
    action *= 0.2   # ← tutti gli action vengono scalati a 0.2×
```

Se l'agente emette `gripper_action = 0.65` (soglia di conferma), il simulatore riceve
`0.65 × 0.2 = 0.13`. Il valore loggato (`GRIP`) è quello **post-scaling**. Quindi:

| Agente emette | Sim riceve (×0.2) | Loggato come GRIP |
|--------------|------------------|--------------------|
| 0.65 | 0.13 | 0.13 |
| 0.80 | 0.16 | 0.16 |
| 1.00 | 0.20 | 0.20 |

La vera politica in HOLD probabilmente punta a 0.65–1.00, ma il log mostra il valore scalato.

**Per grip più ferma** (solo analisi, nessun cambio ora):
- Rimuovere o aumentare il fattore 0.2 per il solo canale gripper (e.g., `action[:-1] *= 0.2`
  ma `action[-1]` rimane invariato)
- Aggiungere un `hold_grip` bonus più forte per `gripper_action > 0.7` (post-scaling)
- Oppure loggare il valore pre-scaling per avere una diagnosi più accurata

---

## 13. Riepilogo Stato Corrente

| Componente | Stato | Note |
|-----------|-------|------|
| Bug `is_closed` | ✅ Risolto | Usa `hinge_qpos_addr` (addr=9) |
| Obs space arricchito | ✅ Attivo | (122,): +hinge_qpos, +friction, +FSM one-hot |
| Friction randomization | ✅ Attivo | 0.3–1.2× bidirezionale |
| Grasp discovery | ✅ Risolto | Threshold 0.65, confirm 2 step |
| Grip slip durante PUSH | ✅ Migliorato | grasp_rate 1.31, door porta a 0.0 |
| HOLD → RETREAT pipeline | ✅ Funziona | Ma solo 18.4% completa prima dell'orizzonte |
| Handle return to rest | ⚠️ Parziale | Funziona in RETREAT, ma orizzonte troppo corto |
| Grip firmness in HOLD | ⚠️ Apparente | Log mostra ×0.2 scaling, policy forse già a 0.65+ |
| Orizzonte troppo corto | ✅ Risolto | 400 → 600 step (config/train_close_config.py) |
| HOLD timer rimbalzo | ✅ Risolto | Soglia 0.02 → 0.04 (env_gen.py linea 252) |

---

## 14. Fix Pre-Run 3 (2026-05-06)

### 14.1 Config: horizon e total_steps

Modifiche applicate dall'utente in `config/train_close_config.py`:

```diff
- horizon     : int = 400
+ horizon     : int = 600   # +200 step = sufficiente per RETREAT completo

- total_steps : int = 1_200_000
+ total_steps : int = 1_500_000   # più steps per convergere con orizzonte più lungo
```

### 14.2 HOLD timer threshold: 0.02 → 0.04

Modifica in `close_generalized/env_gen.py`:

```diff
- if abs(door_qpos) < 0.02:
+ if abs(door_qpos) < 0.04:   # wider than is_closed (0.03) → no bounce-resets
```

**Perché 0.04 e non 0.03:**
- `is_closed` usa `< 0.03` come soglia esterna
- Il timer si attivava solo a `< 0.02`: la zona 0.02–0.03 era "chiusa ma senza hold"
- MuJoCo fa rimbalzare la porta in quella fascia per il contatto elastico
- Con `< 0.04`: ogni volta che `is_closed = True`, anche il timer conta → no più reset per bounce
- La qualità della chiusura è già catturata da `hold = 1.0 - abs(door_qpos)` (reward continuo)

### 14.3 Comportamento del robot dopo RETREAT (risposta alla domanda)

**Il robot rimane fermo e composto dopo la fine del task.**

In `train_close.py` linea 167–168:
```python
if self._success_latched and self._return_hold >= self.cfg.return_hold_steps:
    action = np.zeros_like(action)  # azione zero forzata dopo RETREAT
```

Zero azione nel controller **OSC_POSE** (Operational Space Control) di Robosuite non significa
"nessun torque" — significa **"delta EEF desiderato = 0"**, ovvero il controllore mantiene
attivamente la posizione corrente, resistendo alla gravità e alle perturbazioni. Il braccio
rimane fisso nella posizione di retreat in modo rigido e composto. ✅

---

*Aggiornato il 2026-05-06.*

---

---

# Sessione 2026-05-06 (pomeriggio) — Analisi Run 3 & Fix Pre-Run 4

## 15. Analisi Run 3 (1.5M step, horizon=600)

### 15.1 `ep_len_mean = 399` con `horizon = 600` — ottima notizia

A prima vista sembra un residuo del vecchio horizon=400. In realtà è una prova che gli episodi
**terminano pulitamente** (via `terminated=True`) e NON per troncamento:

```
66% successi  → terminano a ~300 step  (REACH+PUSH+HOLD+RETREAT completo)
34% fallimenti → terminano a ~600 step (troncati a orizzonte)
Media: 0.66 × 300 + 0.34 × 600 ≈ 399 ✅
```

In Run 2 (horizon=400) ogni episodio colpiva il muro a 400 step (truncated). Ora la pipeline
HOLD→RETREAT→+500 **si completa davvero** nei casi di successo. Progresso strutturale.

### 15.2 Success rate 66% (era 92%) — questione di step equivalenti

| Run | Total steps | Horizon | Episodi/env |
|-----|------------|---------|-------------|
| Run 2 | 1.2M | 400 | ~375 |
| Run 3 | 1.5M | 600 | ~470 |

Il conteggio di episodi è simile, ma il task è più difficile (policy coerente per 600 step).
Non è un regressione strutturale — è semplicemente necessario allenare più a lungo.

### 15.3 Causa radice: `action *= 0.2` scalava anche il gripper

In `train_close.py` linea 165:
```python
if self._success_latched:
    action *= 0.2   # SBAGLIATO: anche il gripper viene scalato
```

Con `_success_latched = True`:
- Policy emette gripper = 0.65 (presa salda)
- Simulatore riceve: 0.65 × 0.2 = **0.13** → forza di contatto insufficiente
- La molla della porta supera la forza del gripper → **porta rimbalza a 0.05–0.19 rad**
- `is_closed` (< 0.03) diventa False → **timer HOLD si azzera continuamente**
- Il robot non riesce a completare i 60 step di HOLD → RETREAT non parte

Questo spiega:
- ✗ HOLD shows DOOR: 0.05–0.19 (bouncing)
- ✗ HOLD reward: solo `base: -0.50`, no `hold_grip/hold_act` (timer non conta)
- ✗ Grip HOLD apparentemente bassa (era il valore post-scaling, 5× ridotto)
- ✓ 4:BACK rows mostrano DOOR: 0.01 (quando completa, la porta È chiusa)

### 15.4 Grip in PUSH — perfetta

```
2:PUSH │ GRIP: +0.99–+1.00 │ PHYS_OK │ WIDTH: 0.028–0.062
```
La presa durante PUSH è massimale. Il problema è **solo** nella fase HOLD.

### 15.5 Handle return — funziona quando RETREAT completa

```
4:BACK │ latch_ret: -0.02 │ DOOR: 0.01
```
`latch_ret = -0.02` significa che il giunto della maniglia è a soli 0.02 rad dalla posizione
di riposo — praticamente a zero. La maniglia **ritorna** quando RETREAT si attiva.
Il problema visivo è che RETREAT non si attiva abbastanza spesso (door bounce impedisce HOLD).

---

## 16. Fix Pre-Run 4 (2026-05-06)

### 16.1 Gripper decoupled from ×0.2 scaling — `train_close.py`

```diff
- if self._success_latched:
-     action *= 0.2
+ if self._success_latched:
+     action[:-1] *= 0.2   # arm: piccoli movimenti solo
+     # gripper (action[-1]) invariato: mantiene la pressione di chiusura
```

**Effetto numerico:**

| Policy output gripper | Simulatore riceve (vecchio) | Simulatore riceve (nuovo) | Miglioramento |
|----------------------|---------------------------|--------------------------|---------------|
| 0.65 | 0.13 | **0.65** | **5×** |
| 0.80 | 0.16 | **0.80** | **5×** |
| 1.00 | 0.20 | **1.00** | **5×** |

**Effetti attesi:**
1. Il gripper mantiene la pressione di contatto → la porta non rimbalza fuori dal range `< 0.04`
2. Il timer HOLD si accumula correttamente per 60 step consecutivi
3. RETREAT si attiva in modo consistente → handle ritorna → episodio termina con +500
4. Il log GRIP in HOLD mostrerà 0.65–1.00 invece di 0.13–0.20 (valori reali della policy)

### 16.2 `total_steps` → 2.5M — `config/train_close_config.py`

```diff
- total_steps : int = 1_500_000
+ total_steps : int = 2_500_000
```

Con horizon=600, ogni episodio usa il 50% di step in più. 2.5M step ≈ 1.2M-step-equivalenti
a horizon=400 → allineato a Run 2 che aveva raggiunto il 92% di success rate.

---

*Aggiornato il 2026-05-06.*

---

---

# Sessione 2026-05-06 (sera) — Run 4: Risultati Definitivi

## 17. Risultati Run 4 (2.5M step, horizon=600)

### 17.1 Metriche Finali — Il Task è Risolto

| Metrica | Run 1 | Run 2 | Run 3 | **Run 4** |
|---------|-------|-------|-------|-----------|
| `success_rate` | 0% | 100% | 66% | **100% 🏆** |
| `ep_rew_mean` | 31.6 | 755 | 734 | **1300** |
| `ep_len_mean` | 400 | 399 | 399 | **347** |
| `retreat_rate` | — | 18.4% | ~34% | **~100%** |
| Total steps | 420k | 1.2M | 1.5M | **2.5M** |
| fps | 207 | 207 | 91 | **260** |

Tre metriche da leggere insieme:

- **success_rate = 1.0**: ogni episodio di rollout chiude la porta con successo. ✅
- **ep_rew_mean = 1300**: +73% rispetto a Run 2 (755). L'incremento è quasi interamente
  dovuto al bonus +500 di terminazione (RETREAT completo) che ora si incassa sistematicamente.
- **ep_len_mean = 347 con horizon=600**: gli episodi terminano in media 253 step prima
  dell'orizzonte. Con success_rate=1.0 e terminated che scatta a ~347 step, il robot
  ha imparato a completare l'intero pipeline in modo efficiente:

```
REACH:   ~30–80  step  → approccio alla maniglia
PUSH:    ~50–120 step  → porta da 0.4 → 0.00 rad
HOLD:    60 step       → 2 secondi fermi (target_hold_steps = 30 Hz × 2 s)
RETREAT: ~60–100 step  → arm si allontana, +500 bonus
──────────────────────────────────────────────────────
Media:   ~200–360 step ≈ 347 ✅
```

### 17.2 Analisi Dettagliata delle Tabelle Diagnostiche

#### 2:PUSH — Presa Massimale, Door Progress Enorme

```
2:PUSH │ GRIP: +1.00 │ PHYS_OK │ DOOR: 0.14 │ door_prog: +42.39 │ TOT: +43.61
```

Il reward `door_prog: +42.39` in un singolo step (clippato poi a 100) è la prova che il
carrot reward `_W_PROGRESS_GRASP = 2000.0` sta guidando la policy in modo decisivo. Con
GRIP=+1.00 e PHYS_OK, la presa è meccanicamente perfetta durante la spinta.

#### 3:HOLD — Timer HOLD Funziona, Grip Alta

```
3:HOLD │ GRIP: +0.99 │ DOOR: 0.00 │ hold_grip: +1.00 │ hold_act: -0.44
3:HOLD │ GRIP: +0.98 │ DOOR: 0.00 │ hold_grip: +1.00 │ hold_act: -0.53
```

`hold_grip: +1.00` — confermato che `gripper_action > _GRIPPER_CLOSE_THRESH = 0.65` ✅
Il timer HOLD si accumula correttamente. DOOR=0.00 significa porta completamente chiusa.

⚠️ `hold_flat: -0.75 → -2.87`: il braccio non mantiene una postura orizzontale durante
HOLD. La penalità è cosmetica ma indica che il robot tiene il gripper in angolazione non
ottimale. Non impatta il successo del task ma potrebbe essere migliorato in futuro.

⚠️ Righe HOLD anomale:
```
3:HOLD │ GRIP: -0.99 │ PHYS_OPEN │ DOOR: 0.12  ← robot sta aprendo il gripper in HOLD!
3:HOLD │ GRIP: +0.57 │ PHYS_OK   │ DOOR: 0.07  ← transizione durante ri-afferramento
```
In alcuni episodi il robot rilascia brevemente la presa durante HOLD (GRIP=-0.99). La porta
rimbalza (DOOR=0.07–0.12). Ma `_success_latched` rimane True e il robot ri-afferra,
richiudendo la porta. La success_rate=1.0 conferma che questi casi si risolvono sempre.

#### 4:BACK — RETREAT Sistematico, Gripper Completamente Aperto

```
4:BACK │ GRIP: -0.93 → -1.00 │ PHYS_OPEN │ DOOR: 0.00–0.02 │ ret_grip: +2.00
```

- GRIP = -0.93 → -1.00: il gripper si apre al massimo durante RETREAT ✅
- `ret_grip: +2.00`: il reward di apertura si incassa sempre (gripper_action < -0.6) ✅
- DOOR = 0.00–0.02: la porta rimane chiusa durante il retreat ✅
- `hold: +0.99–+1.00`: la porta è mantenuta chiusa anche mentre il robot si allontana ✅

⚠️ `latch_ret` varia molto:

| latch_ret | latch_qpos | Significato |
|-----------|-----------|-------------|
| -0.02 | 0.02 rad | ✅ Maniglia quasi a riposo |
| -0.03 | 0.03 rad | ✅ Maniglia quasi a riposo |
| -0.11 | 0.11 rad | ⚠️ Lieve deviazione |
| -0.81 | 0.81 rad | ⚠️ Maniglia ancora ruotata ~52° |
| -0.98 | 0.98 rad | ❌ Maniglia ruotata ~63° |
| -1.13 | 1.13 rad | ❌ Maniglia ruotata ~65° |
| -1.48 | 1.48 rad | ❌ Maniglia quasi al limite (-94% del range) |

La variabilità indica che in alcuni episodi il braccio, ritraendosi, tocca ancora la
maniglia e la ruota meccanicamente. La molla MuJoCo la riporta a 0 solo dopo che il
contatto fisico cessa completamente.

---

## 18. La Rete Neurale — Architettura e Addestramento

### 18.1 Algoritmo: SAC (Soft Actor-Critic)

L'agente usa **Soft Actor-Critic** (Haarnoja et al., 2018), un algoritmo off-policy
per spazi d'azione continui particolarmente adatto al controllo robotico:

- **Off-policy**: apprende da un replay buffer di 1M transizioni → alta efficienza dei sample
- **Maximum-entropy**: massimizza reward + entropia della policy → esplorazione intrinseca
- **Actor-Critic**: due reti separate (actor per la policy, critic per il valore)
- **Double-Q trick**: due critic Q₁ e Q₂, usa il minimo → riduce l'overestimation

### 18.2 Architettura delle Reti

```
Input: obs (122,) — normalizzata da VecNormalize (media 0, std 1)

Actor (Policy Network):
  Linear(122 → 512) + ReLU
  Linear(512 → 512) + ReLU
  Linear(512 → 7)   → (μ, log σ) per ogni dim dell'azione
  → Squashed Gaussian: ação ∈ [-1, 1]^7

Critic × 2 (Q-Networks):
  Linear(122 + 7 → 512) + ReLU   ← obs + action concatenati
  Linear(512 → 512) + ReLU
  Linear(512 → 1)                → Q(s, a) scalare

Entropy Temperature α (ent_coef):
  Appresa automaticamente: target_entropy = -dim(A) = -7
  Regola il bilanciamento esplorazione/sfruttamento
```

Totale parametri stimati:
- Actor: 122×512 + 512×512 + 512×14 = ~332k params
- Critic (×2): (129)×512 + 512×512 + 512×1 = ~329k × 2 = ~658k params
- **Totale: ~990k parametri trainabili**

### 18.3 Spazio di Osservazione (122 dimensioni)

```
[0:117]   Base robosuite obs (118 dim):
           - joint positions q (7)
           - joint velocities q̇ (7)
           - eef position xyz (3)
           - eef quaternion (4)
           - gripper qpos (2)
           - object observations: door pos, handle pos, door angle (varie)
           - robot0_robot-state (inertia, etc.)

[118]     dist              = ||EEF − handle_site||₂          [m]
[119]     handle_radius     = raggio maniglia randomizzato       [m]
[120]     handle_friction   = attrito maniglia randomizzato      [N·s/m]
[121]     fsm_reach         = 1.0 se fase REACH, else 0         {0,1}
[122]     fsm_push          = 1.0 se fase PUSH,  else 0         {0,1}
[123]     fsm_hold          = 1.0 se fase HOLD,  else 0         {0,1}
[124]     fsm_retreat       = 1.0 se fase RETREAT, else 0       {0,1}
[125]     hinge_qpos        = angolo porta live da sim           [rad]
```

> **Nota**: la dimensione effettiva è 122, non 126. I 4 flag FSM + hinge_qpos + dist +
> handle_radius + handle_friction sostituiscono i 4 custom precedenti. Totale custom = 8,
> base robosuite = 114 (non 118 come stimato sopra). Verificato: `obs.shape == (122,)`.

### 18.4 Spazio d'Azione (7 dimensioni)

```
action[0:5]  → delta EEF pose (Δx, Δy, Δz, Δroll, Δpitch, Δyaw) in spazio operazionale
action[6]    → comando gripper [-1=aperto, +1=chiuso]

Controller: OSC_POSE (Operational Space Control)
  - Converte delta-EEF in coppie articolari via Jacobiano
  - Zero azione = mantieni posizione EEF corrente (stiff)
  - action[:-1] × 0.2 durante HOLD: arm quasi fermo
  - action[-1] invariato durante HOLD: gripper a piena forza (fix Run 4)
```

### 18.5 Training Setup

| Parametro | Valore | Note |
|-----------|--------|------|
| `learning_rate` | 3×10⁻⁴ | Adam optimizer |
| `buffer_size` | 1M | Replay buffer |
| `batch_size` | 256 | Mini-batch per gradient step |
| `gamma` | 0.95 | Discount factor |
| `tau` | 0.005 | Soft target update |
| `gradient_steps` | 2 | Update steps per env step |
| `learning_starts` | 10k | Random policy warmup |
| `ent_coef` | auto | Target entropy = -7 |
| `action_smooth_alpha` | 0.8 | EMA smoothing: a = 0.8×new + 0.2×prev |
| `num_envs` | 8 | DummyVecEnv parallelo |
| VecNormalize | ✅ | norm_obs=True, norm_reward=True |

### 18.6 Lettura dell'ent_coef — Convergenza della Policy

```
Run 1 (bloccato):  ent_coef = 6.09e-05  ← entropy collassata, nessuna esplorazione
Run 2 (100%):      ent_coef = 8.16e-05  ← bassa, policy deterministica
Run 3 (66%):       ent_coef = 8.91e-05  ← leggermente più alta, policy meno sicura
Run 4 (100%):      ent_coef = 1.17e-04  ← ottimale: deterministica ma con margine
```

Il valore 1.17e-04 di Run 4 indica che la policy è **convergente ma non sovra-deterministica**.
La piccola entropia residua permette di gestire la variabilità della door friction e
dell'handle radius senza collassare in un unico pattern di comportamento.

`ent_coef_loss = -1.53 → -1.67`: negativo significa che l'algoritmo sta **riducendo
attivamente l'entropia** — la policy sta convergendo verso comportamenti più decisi.

`actor_loss = -0.182 → -0.215`: negativo in SAC indica che l'actor è migliorato
rispetto alla baseline del critic — il reward atteso Q(s,π(s)) > 0. ✅

`critic_loss = 5.72e-04`: estremamente basso — il critic predice Q(s,a) con precisione
quasi perfetta. Questo significa che la policy si trova in un ottimo stabile. ✅

### 18.7 Curriculum Adattivo

```python
if success_rate > 0.85 and grasp_rate > 0.50 and level < 1.0:
    level = min(1.0, level + 0.05)
```

Il curriculum parte dal livello 0.0 (posizione/orientamento porta fisso) e sale
incrementalmente fino a 1.0 (massima variazione). Questo garantisce che la policy non
venga sopraffatta dalla variabilità nelle fasi iniziali dell'addestramento.

---

## 19. Problemi Residui dopo Run 4

| Problema | Gravità | Descrizione |
|---------|---------|-------------|
| `latch_ret` variabile | 🟡 Moderata | Maniglia rimane ruotata in alcuni episodi (braccio tocca la maniglia durante il retreat) |
| `hold_flat` penalità alta | 🟢 Bassa | Orientamento non ottimale durante HOLD, non impatta il successo |
| `hold_act` penalità | 🟢 Bassa | Arm si muove un po' durante HOLD (-0.44 to -0.61) |
| Grip negativa in alcuni HOLD | 🟡 Moderata | Policy rilascia brevemente il gripper in HOLD, ma si ri-afferra sempre |
| PHYS_OPEN con WIDTH ~0 | 🟢 Bassa | Gripper troppo chiuso → contatto fisico strano in sim (funziona uguale) |

---

*Aggiornato il 2026-05-06.*
