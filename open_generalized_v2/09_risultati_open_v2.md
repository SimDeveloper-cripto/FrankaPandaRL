# Risultati — Apertura Generalizzata v2 (curriculum 1)

**Stato:** baseline §1.30 **congelata** ("resta così").
**Run:** SAC, 1.5M step, curriculum 1 (posa variabile ±15 cm / ±17°, fisica randomizzata).
**Documento vivo:** verrà aggiornato nei run successivi.

---

## 1. Risultato headline

| Metrica | Valore |
|---|---|
| **Eval deterministico (is_success)** | **100.0 %** (best 100.0 %) |
| **ep_len di eval** | **98.0** |
| Rollout `success_rate` (fine training) | 0.97–0.99 |
| Rollout `ep_len_mean` | 111–125 |
| Rollout `ep_rew_mean` | da −85 (1.49M) a −43 (1.50M) — in miglioramento |
| `ent_coef` | ~5.1–5.4 × 10⁻⁴, **stabile** (nessun entropy collapse) |
| SR = 1 raggiunto da | ~1.36M step |

Il task è **risolto**: la policy afferra la maniglia (posa/fisica variabili), apre la porta
all'angolo-obiettivo, mantiene l'apertura e si ritira con terminazione pulita, al 100 %
deterministico.

---

## 2. Curva di training (ultimo tratto, 1.49M → 1.50M)

- `success_rate` oscilla 0.97–0.99 in rollout (stocastico); l'eval deterministico è 100 %.
- `ep_rew_mean` sale costantemente (−85 → −43): la policy diventa più efficiente (meno
  time-penalty accumulato, ritiro più diretto).
- `ent_coef` resta ~5.3 × 10⁻⁴ senza collassare: l'`target_entropy = +1.0` (alzato per
  contrastare l'entropy collapse, §1.9.C) tiene la stocasticità necessaria a innescare la
  presa laterale.
- `critic_loss` ~2–5 × 10⁻⁴, `actor_loss` ~0.10: training stabile, nessuna divergenza.

---

## 3. Diagnostico fase — 20 episodi deterministici

**Sintesi:** 20/20 episodi raggiungono **RETREAT** con **terminazione pulita**; la porta
**raggiunge sempre** il goal (`open_error` minimo medio **0.0043**, max 0.0108).

### 3.1 Successo

| Definizione | Valore |
|---|---|
| `is_success` (fase HOLD_OPEN/RETREAT raggiunta) — **metrica di eval** | **20/20 = 100 %** |
| true-final entro tol (`open_error` *finale* ≤ 0.05) | 18/20 = 90 % |
| episodi con finale > tol | Ep19 (goal 0.350 → 0.0502), Ep20 (goal 0.344 → 0.0521) |

### 3.2 `open_error` (rad)

| | minimo | finale |
|---|---|---|
| media | 0.0043 | 0.0327 |
| mediana | — | 0.0382 |
| min | — | 0.0047 |
| max | 0.0108 | 0.0521 |
| std | — | 0.0135 |

Il **minimo** ≈ 0 conferma che la porta **passa sempre** per il goal. Il **finale** più alto
è lo scostamento a fine episodio.

### 3.3 Timing delle fasi (step)

| Fase | media | range |
|---|---|---|
| REACH | 21.2 | [14, 31] |
| PULL | 12.0 | [1, 20] |
| HOLD_OPEN | 34.5 | [30, 44] |
| RETREAT | 31.0 | costante (target 30 + 1) |
| **ep_len** | **98.7** | — (coerente con eval 98) |

---

## 4. Analisi a fondo: da dove viene l'`open_error` finale

Il finale **non** è rumore né un difetto di presa: è **interamente geometrico**, spiegato dal
modello fisico della porta.

**Evidenza quantitativa.** Predicendo `open_error_finale ≈ (0.400 − goal_angle)`:

- **corr( (0.400 − goal), finale ) = 0.975**
- scarto medio |finale − (0.400 − goal)| = **0.0029 rad**

Cioè: a fine episodio la porta sta **al fine-corsa 0.400 rad**, e l'errore è semplicemente la
distanza tra il goal e quel limite. Conferma incrociata: le correlazioni di controllo sono
~0 — `corr(g_thresh, finale) = 0.22`, `corr(handle_radius, finale) = −0.04` — quindi il
residuo **non** dipende da frizione, soglia di presa o raggio maniglia.

**Split per goal:**

| Gruppo | n | finale medio | finale max |
|---|---|---|---|
| goal alti (≥ 0.385) | 3 | 0.0090 | 0.0115 |
| goal bassi (≤ 0.355) | 7 | 0.0450 | 0.0521 |

### 4.1 Perché — la fisica del simulatore (robosuite Door)

Dal modello `robosuite/models/assets/objects/door.xml`:
```xml
<joint name="hinge" axis="0 0 1" range="0.0 0.4"
       damping="1" frictionloss="1" limited="true"/>
```
- `range="0.0 0.4" limited="true"` → la porta è **hard-limitata a [0, 0.4] rad**: lo 0.400 è
  un **fine-corsa fisico** (l'escursione è esattamente 0.400 in ogni episodio).
- **Nessuno `stiffness`** sul cardine → **niente molla di richiamo**: la porta **non** torna
  indietro da sola.
- `frictionloss="1"` → attrito secco: quando la porta si ferma, **resta ferma**.

**Meccanismo del residuo.** Il termine `door_prog` premia l'apertura fino al fine-corsa 0.400;
la porta passa per il goal (`open_error` min ≈ 0) e prosegue fino al limite, dove **resta**
(la tiene il `frictionloss`). Quindi `door_end ≈ 0.400` e `open_error_finale ≈ |0.400 − goal|`:
~0.005 per goal alti, ~0.05 per i goal più bassi. La causa è **nel reward, non nel
simulatore** → è migliorabile (vedi §5), ma non intacca il 100 % di eval.

> Nota: la *molla* esiste solo sul **latch** (la maniglia, che torna a neutro) — è ciò che
> `latch_restore`/`latch_ret` accompagna nel RETREAT. Il **cardine** non ha molla.

---

## 5. Caratteristica nota e lever disponibile (non applicato)

La baseline è **congelata al 100 %**. Resta documentata una sola caratteristica: per i goal
più bassi `door_end` si ferma al fine-corsa 0.400 invece che esattamente al goal (2/20 episodi
con finale appena oltre tol). Se in futuro si vuole `door_end` *preciso* al goal:

- **Lever §1.31 (opt-in, default OFF):** `pull_progress_cap_at_goal = True` fa saturare il
  progresso al goal (`min(door_angle, goal)`), specchio esatto della chiusura → la policy
  lascia la porta al goal e il `frictionloss` la tiene lì. Da validare con A/B (flag ON vs
  OFF, stesso seed), confrontando i 20 `open_error` finali.
- **Lever alternativo (geometrico):** `open_tol_rad = 0.06` copre l'intera coda osservata
  (max 0.0521) — analogo diretto del §1.29.
- **Lever ulteriore (se resta inerzia):** freeze duro del braccio in HOLD_OPEN
  (`action[:-1] = 0`, identico alla chiusura).

---

## 6. Riproduzione

```bash
# training (curriculum 1, 1.5M step)
mjpython open_generalized_v2/train_curriculum_v2.py --total-steps 1500000

# eval deterministico + diagnostico per-episodio
python -m open_generalized_v2.diagnose_phase --episodes 20

# visualizzazione della policy
mjpython open_generalized_v2/train_curriculum_v2.py --play
```

Config chiave: `target_entropy = 1.0`, `gamma = 0.95`, `learning_starts = 20_000`,
`gradient_steps = 1`, `open_tol_rad = 0.05`, `door_open_cap_rad = 0.400`,
`goal_frac ∈ [0.85, 1.00]`, FSM a soglie adattive (§1.30: gate `PULL→HOLD_OPEN` su `g_thresh`).

---

## 7. Tabella per-episodio (diagnostico, 20 ep)

| Ep | goal | g_thresh | REACH | PULL | HOLD | RETREAT | oe_min | oe_fin | entro tol |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.357 | 0.719 | 20 | 13 | 30 | 31 | 0.0041 | 0.0386 | ✓ |
| 2 | 0.389 | 0.720 | 26 | 6 | 30 | 31 | 0.0049 | 0.0115 | ✓ |
| 3 | 0.395 | 0.650 | 27 | 5 | 43 | 31 | 0.0031 | 0.0047 | ✓ |
| 4 | 0.389 | 0.650 | 14 | 19 | 39 | 31 | 0.0108 | 0.0108 | ✓ |
| 5 | 0.374 | 0.683 | 18 | 15 | 30 | 31 | 0.0080 | 0.0209 | ✓ |
| 6 | 0.350 | 0.650 | 20 | 13 | 30 | 31 | 0.0003 | 0.0407 | ✓ |
| 7 | 0.360 | 0.678 | 15 | 20 | 31 | 31 | 0.0039 | 0.0377 | ✓ |
| 8 | 0.357 | 0.737 | 19 | 14 | 36 | 31 | 0.0045 | 0.0433 | ✓ |
| 9 | 0.375 | 0.650 | 17 | 16 | 30 | 31 | 0.0061 | 0.0242 | ✓ |
| 10 | 0.373 | 0.650 | 27 | 4 | 30 | 31 | 0.0026 | 0.0263 | ✓ |
| 11 | 0.380 | 0.650 | 20 | 15 | 36 | 31 | 0.0059 | 0.0202 | ✓ |
| 12 | 0.355 | 0.671 | 16 | 18 | 30 | 31 | 0.0020 | 0.0420 | ✓ |
| 13 | 0.362 | 0.720 | 31 | 1 | 43 | 31 | 0.0009 | 0.0378 | ✓ |
| 14 | 0.371 | 0.730 | 18 | 20 | 36 | 31 | 0.0027 | 0.0244 | ✓ |
| 15 | 0.360 | 0.660 | 18 | 16 | 38 | 31 | 0.0054 | 0.0392 | ✓ |
| 16 | 0.353 | 0.697 | 30 | 2 | 30 | 31 | 0.0048 | 0.0471 | ✓ |
| 17 | 0.350 | 0.650 | 26 | 6 | 41 | 31 | 0.0005 | 0.0409 | ✓ |
| 18 | 0.347 | 0.711 | 17 | 16 | 30 | 31 | 0.0025 | 0.0419 | ✓ |
| 19 | 0.350 | 0.663 | 28 | 6 | 33 | 31 | 0.0071 | 0.0502 | ✗ |
| 20 | 0.344 | 0.693 | 17 | 15 | 44 | 31 | 0.0052 | 0.0521 | ✗ |

Tutti: fase MAX = RETREAT, terminazione **PULITA**, `is_success` = SÌ, presa fisica
confermata, `dist_handle` minima sotto soglia (maniglia sempre raggiunta).
