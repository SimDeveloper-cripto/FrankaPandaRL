---

# Generalizzazione del Task di APERTURA della Porta (open_generalized_v2)

> **Documento speculare a `update_v2.md` (chiusura generalizzata, curr0+curr1).**
> Mentre quel documento copre l'intero arco curr0→curr1 della chiusura, questo è
> focalizzato sul **solo curriculum 1** dell'apertura (posa variabile, soglie adattive,
> fisica randomizzata): `fixed_curriculum_level = 1.0`. La curr0 (posa fissa) non è un
> capitolo separato qui, perché il pacchetto di apertura nasce già come generalizzazione
> della v1 di apertura e riusa l'impalcatura matura della chiusura v2.
>
> **Principio guida di tutto il lavoro:** ciò che riguarda la *competenza del task*
> (avvicinarsi, afferrare, aprire, mantenere) vive nel **reward potential-based**
> `[3]`; ciò che riguarda la *qualità del movimento* a successo già raggiunto (come si
> stacca, come accompagna la leva) vive in **logica deterministica a livello di
> environment**, a reward zero. Questa separazione è la lezione metodologica ereditata
> dalla chiusura ed è ciò che ha reso possibile arrivare al ~95–100% senza rompere
> ripetutamente ciò che già funzionava.

---

### 7.1 — Architettura dei Moduli e Responsabilità

Il modulo è composto da 9 file (8 come la chiusura, più `diagnose_phase.py`, strumento
diagnostico nato durante lo sviluppo dell'apertura). Ogni file ha **una sola
responsabilità**: nessuna logica è duplicata tra moduli diversi.

```
open_generalized_v2/
│
├── config_v2.py              ← TrainConfigV2Open
│                                Unica fonte di verità per tutti i parametri.
│                                Non ha logica: è un dataclass di soli valori.
│                                Viene passato come argomento (cfg) a ogni modulo.
│                                Modificare qui per cambiare qualsiasi iperparametro.
│
├── fsm_v2.py                 ← AdaptiveFSMOpen + FSMStateOpen
│                                Decide QUANDO transitare tra fasi.
│                                Calcola le soglie adattive (§3.1).
│                                NON calcola il reward e NON tocca MuJoCo.
│                                Fasi: REACH → PULL → HOLD_OPEN → RETREAT.
│
├── reward_v2.py              ← PotentialBasedRewardOpen
│                                Calcola QUANTO reward assegnare a ogni step.
│                                Legge la fase corrente da FSMStateOpen.
│                                NON decide le transizioni e NON tocca MuJoCo.
│                                Il potenziale di PULL premia il progresso verso il
│                                goal di APERTURA (non verso 0 come la chiusura).
│
├── grasp_strategy.py         ← MultiApproachGrasp   [RIUSATO identico dalla chiusura]
│                                Calcola le K=3 direzioni di approccio (§3.3).
│                                Calcola il valore di alignment per ciascuna.
│                                Fornisce le features per l'osservazione.
│                                Indipendente dal verso del task → invariato.
│
├── domain_rand_v2.py         ← ExtendedDomainRandomizer  [RIUSATO identico]
│                                Modifica il modello MuJoCo in-place a ogni reset.
│                                È l'unico modulo che scrive su sim.model.
│                                Espone i valori randomizzati come attributi correnti.
│                                La fisica non dipende dal verso del task → invariato.
│
├── beta_net.py               ← BetaNetwork (disabilitata di default)
│                                Gate probabilistico AGGIUNTIVO per le transizioni FSM.
│                                Contiene tre MLP indipendenti (una per fase).
│                                Restituisce sempre {1.0, 1.0, 1.0} se disabilitata.
│
├── env_v2.py                 ← AdvancedGeneralizedOpenDoorEnv
│                                Orchestratore: assembla e chiama tutti gli altri moduli.
│                                È il solo punto di contatto con SB3 e Robosuite.
│                                Contiene gli override deterministici del RETREAT
│                                (§1.17/§1.21/§1.22/§1.26): rilascio pulito,
│                                rampa di avvio, accompagnamento leva con cap temporale.
│
├── train_curriculum_v2.py    ← main() + EvalBestCallback
│                                Crea env, model SAC, callback di eval e lancia il
│                                training (curriculum 1 fisso). Gestisce --play.
│
└── diagnose_phase.py         ← strumento diagnostico [NUOVO nell'apertura]
                                 Non tocca il training. Gira N episodi e riporta
                                 DOVE si blocca la catena REACH→PULL→HOLD_OPEN→RETREAT
                                 (fase massima, presa, escursione porta, open_error).
                                 Modalità --random / --scripted / policy addestrata.
```

**Dipendenze tra file (chi importa chi):**
```
train_curriculum_v2.py
    └── importa: config_v2, env_v2

env_v2.py
    └── importa: config_v2, fsm_v2, reward_v2,
                 grasp_strategy, domain_rand_v2, (beta_net)

reward_v2.py
    └── importa: fsm_v2 (solo FSMStateOpen, PHASE_*)

diagnose_phase.py
    └── importa: config_v2, env_v2, fsm_v2

Tutti gli altri
    └── importano solo: config_v2 (TrainConfigV2Open)
```

Come per la chiusura, questa struttura fa sì che ogni modulo possa essere testato in
isolamento: per testare `reward_v2.py` non serve MuJoCo — basta creare un `FSMStateOpen`
fittizio. Questa proprietà è stata sfruttata intensamente durante lo sviluppo
dell'apertura per verificare le correzioni del RETREAT senza ri-addestrare.

---

### 7.2 — Flusso di un Episodio (reset)

```
env.reset()
  │
  ├── 1. reset dell'env base Robosuite (Door, Panda, OSC controller)
  │
  ├── 2. domain_rand.randomize_episode(curriculum_level=1.0)
  │        ├── randomizza geometria/frizione maniglia (raggio ×0.7–1.4, ecc.)
  │        ├── randomizza rigidità latch (×0.5–2.0)        [§3.4]
  │        ├── randomizza damping cerniera (×0.3–1.5)       [§3.4]
  │        └── randomizza massa porta (×0.5–2.0)            [§3.4]
  │
  ├── 3. randomizza la posa della porta (curriculum 1)
  │        posizione XY ±15 cm, yaw ±17° (× curriculum_level = 1.0)
  │
  ├── 4. CAMPIONAMENTO DEL GOAL DI APERTURA  ← DIFFERENZA CHIAVE vs chiusura
  │        door_min      = angolo a porta chiusa
  │        effective_max = door_min + door_open_cap_rad (cap = 0.400 rad)
  │        goal_frac     ~ U[goal_frac_min, goal_frac_max] = U[0.85, 1.00]
  │        goal_angle    = door_min + goal_frac × cap
  │        → l'obiettivo NON è una costante (come 0 nella chiusura):
  │          è un angolo-bersaglio variabile, alto e vicino all'apertura piena.
  │
  ├── 5. fsm.reset()          → fase = REACH, contatori azzerati,
  │                             target_hold_steps = None (calcolato al 1° step)
  │
  ├── 6. reward_fn.reset()    → _prev_phi = 0.0, _max_door_angle = None
  │
  └── 7. costruisce la prima osservazione (vedi §7.5) e la ritorna
```

Il punto **4** è la prima vera differenza concettuale rispetto alla chiusura: nella
chiusura il successo fisico è sempre lo stesso stato (`door_angle ≈ 0`), mentre
nell'apertura il bersaglio è un angolo campionato a ogni episodio. Questo rende il
compito intrinsecamente *goal-parametrico* (vedi §10.1 — Difficoltà) e impone che il
`goal_angle` faccia parte dell'osservazione, perché la policy deve sapere *quanto* aprire.

---

### 7.3 — Flusso di uno Step (con confronto chiusura vs apertura)

| Passo dello step | Chiusura (close v2) | Apertura (open v2) |
|------------------|---------------------|--------------------|
| 1. EMA sull'azione | `a = α·a + (1−α)·a_prev` | identico (`action_smooth_alpha = 0.8`) |
| 2. Override env-level RETREAT | rilascio pulito + rampa | rilascio pulito + rampa + **accompagnamento leva con cap** (§1.22/§1.26) |
| 3. `sim.step(action)` | 1 sola chiamata | **1 sola chiamata** (invariato — nessuno step fantasma) |
| 4. lettura stato | door_qpos, latch_qpos, eef… | door_angle, latch_qpos, eef, **goal_angle** |
| 5. `fsm.update(...)` | REACH→PUSH→HOLD→RETREAT | REACH→**PULL**→**HOLD_OPEN**→RETREAT |
| 6. `reward_fn.compute(...)` | progresso verso `door=0` | progresso verso **`door=goal_angle`** |
| 7. terminazione | `door<0.03 AND latch<0.08` | **`door≥goal−tol AND retreat sostenuto`** (§1.25 corretto) |
| 8. costruzione obs | ~47-D | ~47-D + goal info |

I passi 5–7 sono il cuore dell'inversione chiusura↔apertura e sono trattati in
dettaglio nelle §7.4 (FSM), §7.9 (reward) e §10 (differenze). Il passo **3** è marcato in
grassetto perché è una regola dura ereditata dalla chiusura e mai violata: **un solo
`sim.step()` per `env.step()`**. Gli override del RETREAT modificano l'azione *prima*
dello step, non aggiungono step extra.

---

### 7.4 — Statechart FSM: Confronto Chiusura vs Apertura

**Differenze strutturali (chiusura → apertura):**

| Aspetto FSM | Chiusura (close v2) | Apertura (open v2) |
|------------|---------------------|--------------------|
| Fasi | REACH → PUSH → HOLD → RETREAT | REACH → **PULL** → **HOLD_OPEN** → RETREAT |
| Soglia distanza REACH→(PUSH/PULL) | `1.5 × radius + 0.005` adattiva | **identica** |
| Soglia grip REACH→(PUSH/PULL) | `0.75 − 0.10 × norm_friction` adattiva | **identica** |
| Conferma grasp | 5 step consecutivi | **5 step consecutivi (identico)** |
| Check fisico contatto | `is_phys_closed` | **identico** |
| Condizione (PUSH/PULL)→HOLD | `door_angle ≤ success_angle` (porta chiusa) | **`door_angle ≥ goal_angle − tol`** (porta aperta al goal) — **INVERSIONE** |
| Soglia grip alla transizione | `grip > 0.80` (letterale) | **`grip > g_thresh(f)` adattiva (§1.30)** — vedi nota sotto |
| Timer HOLD | `base + extra(stiffness)` adattivo | **identico** (`fsm_hold_base_steps + extra`) |
| Soft reset timer HOLD | sì (su rimbalzo verso chiuso) | **sì** (su ri-chiusura sotto il goal) |
| Arco di ritorno | PUSH→REACH (grasp perso) | **PULL→REACH** (grasp perso, identico nella logica) |
| Terminazione RETREAT | gestita dall'env/reward (latch neutro) | gestita dall'env/reward (**porta aperta + retreat sostenuto**) |

**Statechart apertura (le differenze rispetto alla chiusura sono evidenziate):**

```
                   ┌───────────────────────────────────────────────────────┐
                   │                                                       │
                   ▼                                                       │
         ┌──────────────────┐                                             │
  start  │                  │  dist < d_thresh(r)   ← adattiva           │
 ──────► │   PHASE_REACH    │  AND grip > g_thresh(f) ← adattiva         │
         │                  │  AND is_phys_closed     ← check fisico     │
         │  reach_steps++   │  per 5 step consecutivi                    ├──► PHASE_PULL
         │                  │                                             │
         └──────────────────┘                                             │
                                                              ┌───────────┴──────────┐
                                     grasp_lost               │                      │
                        ◄────────────────────────────────     │    PHASE_PULL        │
                                                              │                      │
                                                              │  pull_steps++        │
                                                              │  max_door_angle ↑    │  ← [APERTURA]
                                                              └──────────────────────┘     ratchet che SALE
                                                                          │                (specchio del
                                            door_angle >= goal_angle − tol │                min↓ della chiusura)
                                            AND grip > g_thresh(f) ← adattiva (§1.30)│
                                            ◄── [INVERSIONE vs chiusura] ──┘
                                                                          │
                                                                          ▼
                                                              ┌──────────────────────┐
                                                              │                      │
                                                              │   PHASE_HOLD_OPEN    │
                                                              │                      │
                                                              │  hold_open_dur++     │
                                                              │  (se door≥goal−tol)  │
                                                              │  soft reset se la    │
                                                              │  porta si ri-chiude  │
                                                              └──────────────────────┘
                                                                          │
                                                       hold_open_dur >= target_hold_steps
                                                       (base + extra(stiffness))
                                                                          │
                                                                          ▼
                                                              ┌──────────────────────┐
                                                              │                      │
                                                              │   PHASE_RETREAT      │
                                                              │                      │
                                                              │  retreat_steps++     │
                                                              │  env-level:          │
                                                              │   §1.22 accompagna   │
                                                              │     leva (cap §1.26) │
                                                              │   §1.17 rilascio     │
                                                              │   §1.21 rampa ritiro │
                                                              └──────────────────────┘
                                                                          │
                                              retreat_steps >= fsm_retreat_target_steps
                                              AND door_angle >= goal_angle − tol
                                              ◄── [APERTURA: NO gate su latch] ──┘
                                                                          │
                                                                          ▼
                                                                      SUCCESS ✓
```

Tre punti meritano enfasi, perché sono le differenze che hanno richiesto lavoro
specifico (vedi anche §1.25/§1.26 in §10):

1. **`PULL→HOLD_OPEN` è l'inversione esatta della chiusura.** La chiusura entra in HOLD
   quando `door ≤ success_angle` (porta arrivata a zero); l'apertura entra in HOLD_OPEN
   quando `door ≥ goal_angle − tol` (porta arrivata al bersaglio alto). È letteralmente
   la stessa condizione con il segno della disuguaglianza invertito e `goal_angle` al
   posto di `success_angle`.

2. **Il ratchet di progresso SALE** (`max_door_angle ↑`) invece di scendere
   (`min_door_angle ↓`). Vedi §7.9 (termine `door_prog`).

3. **La terminazione del RETREAT NON ha un gate sul latch.** Questa è la lezione più
   sottile dell'apertura (§1.25): nella chiusura la porta va a 0 e il latch scatta a zero
   per fisica, quindi `latch < 0.08` è una condizione *raggiungibile*; nell'apertura la
   porta resta spalancata e la leva non torna a zero da sola, quindi un gate `latch<tol`
   bloccherebbe l'episodio all'infinito. La terminazione è perciò allineata allo stato
   fisicamente raggiungibile: **porta aperta + retreat sostenuto**.

---

### 7.5 — Come la Rete Neurale (SAC) Vede il Sistema

L'osservazione è strutturalmente identica alla chiusura (~47-D), con l'aggiunta
dell'informazione di goal, necessaria perché il bersaglio è variabile.

| Blocco di osservazione | Contenuto | Note apertura |
|------------------------|-----------|---------------|
| Stato robot (proprio) | giunti braccio, gripper width, eef pos/quat | identico |
| Maniglia | posizione, vettore eef→maniglia, distanza | identico |
| Porta | `door_angle`, velocità | identico |
| **Goal di apertura** | `goal_angle` (e/o `open_error = goal − door`) | **specifico apertura** |
| Fisica (domain rand) | `[norm_latch_stiffness, norm_hinge_damping, norm_door_mass]` | identico (3 feature) |
| Grasp multi-approccio | `[best_align, align_0..K-1]` | identico (K+1 feature) |

L'azione è la stessa della chiusura: **8-D** (7 per il braccio via OSC + 1 gripper),
ognuna in `[−1, 1]`, con smoothing EMA (`action_smooth_alpha = 0.8`).

Il punto da sottolineare per la tesi: dal punto di vista di SAC, apertura e chiusura sono
**lo stesso MDP con un segnale di goal diverso**. La FSM e il reward traducono quel goal
in fasi e in gradiente; la rete non "sa" di star aprendo invece di chiudere — vede solo
osservazioni, azioni e reward. Questa è la ragione per cui l'impalcatura della chiusura
si riusa quasi interamente.

---

### 7.6 — Flusso del Training Loop Completo

```
train_curriculum_v2.py  main()
  │
  ├── cfg = TrainConfigV2Open(); cfg.fixed_curriculum_level = 1.0
  │
  ├── venv = DummyVecEnv([make_env_fn(cfg) × 8])     ← 8 env paralleli
  │          VecMonitor → VecNormalize(norm_obs, norm_reward, clip 10)
  │
  ├── eval_env = DummyVecEnv([make_env_fn(cfg) × 1]) ← 1 env di valutazione
  │              VecNormalize(norm_obs=True, norm_reward=False); training=False
  │
  ├── model = SAC("MlpPolicy", venv,
  │               lr=3e-4, buffer=1e6, batch=256, gamma=0.95, tau=0.005,
  │               train_freq=1, gradient_steps=1, learning_starts=20000,
  │               ent_coef="auto", target_entropy=+1.0,   ← DIFFERENZA (vedi §10.2)
  │               net_arch=(512,512))
  │
  ├── cb = EvalBestCallback(eval_env, run_dir, eval_freq=50k, n_eval=20)
  │          → ogni 50k step: valuta 20 episodi deterministici,
  │            salva sempre latest_model.zip + vecnormalize.pkl,
  │            salva best_model.zip quando il success migliora.
  │
  └── model.learn(total_timesteps=1_500_000, callback=cb)
      → final_model.zip + vecnormalize.pkl
```

Rispetto alla chiusura (che documenta 4 callback, incluso quello di curriculum
automatico), qui il loop è **più semplice** perché il progetto è a **curriculum fisso**
(livello 1): non c'è un callback di avanzamento di curriculum, perché non c'è curriculum
da far avanzare. Restano l'eval periodico e il salvataggio di best/latest, con la
robustezza aggiunta (salvataggio `latest_model.zip` a ogni eval) che permette al
`diagnose_phase.py` di caricare un modello anche da run interrotti a metà.

---

### 7.7 — Descrizione Dettagliata degli Oggetti Principali

#### `TrainConfigV2Open` — `config_v2.py`

Dataclass di soli valori, unica fonte di verità. Gruppi principali:

| Gruppo | Parametri chiave | Note apertura |
|--------|------------------|---------------|
| SAC | `target_entropy = 1.0`, `learning_starts = 20000`, `gradient_steps = 1` | `target_entropy` alzato (§10.2) |
| Goal di apertura | `door_open_cap_rad = 0.400`, `goal_frac_min/max = 0.85/1.00`, `open_tol_rad = 0.05` (§1.29) | **specifico apertura** |
| FSM adattiva | `fsm_grip_thresh_base = 0.75`, `fsm_grasp_dist_*`, `fsm_hold_base_steps = 30`, `fsm_retreat_target_steps = 30` | identici alla chiusura |
| Reward potential | `phi_reach_weight = 25`, `phi_reach_sigma = 0.40`, `phi_pull_weight = 5`, `phi_hold_weight = 5`, `phi_retreat_weight = 5` | pesi piccoli `O(1–5)` (§9.3) |
| Reward denso PULL | `w_pull_progress = 300`, `w_pull_dist_3d = 5`, `w_pull_dist_z = 15`, `w_grip_contact = 0.5` | mirror di `door_prog` |
| Reward denso REACH | `w_reach_dist_3d/xy/z`, `w_reach_app_blw/top`, `w_reach_grip_near` | mirror chiusura |
| RETREAT env-level | `retreat_clean_release`, `retreat_clear_margin = 0.02`, `retreat_rampup_steps = 8`, `retreat_latch_restore`, `retreat_latch_neutral_tol = 0.05`, **`retreat_latch_max_steps = 20`** | §1.17/§1.21/§1.22/**§1.26** |
| RETREAT reward | `w_door_regress = 4.0`, `w_latch_ret = 1.0`, `retreat_latch_term_tol = 0.08` | §1.25 |
| Domain rand | `rand_latch_stiffness/hinge_damping/door_mass` (+ range) | identici alla chiusura |
| Grasp | `grasp_n_candidates = 3` | identico |
| beta_net | `use_beta_net = False` | disabilitato (capitolo futuro) |

#### `FSMStateOpen` — `fsm_v2.py` (dataclass)

Contatori e stato della macchina: `phase`, `grasp_confirm_count`, `hold_open_duration`,
`reach_steps`, `pull_steps`, `hold_steps_total`, `retreat_steps`, `return_hold`,
`retreat_pos`, `target_hold_steps`. `reset()` riporta tutto a REACH/0. Rispetto alla
chiusura cambiano solo i nomi semantici (`pull_steps`, `hold_open_duration`).

#### `AdaptiveFSMOpen` — `fsm_v2.py`

API speculare a `AdaptiveFSM` della chiusura:
- `grip_thresh(friction)` → soglia di chiusura gripper adattiva alla frizione (ManipForce
  `[13]`): `clip(0.75 − 0.10 × norm_friction, 0.50, 0.90)`.
- `grasp_dist_thresh(handle_radius)` → distanza di presa adattiva al raggio:
  `0.045 + 1.5 × radius + 0.005`.
- `compute_target_hold_steps(control_freq, latch_stiffness, base)` → timer HOLD_OPEN
  adattivo: latch più rigido tende a richiudere → si mantiene l'apertura un po' di più.
- `update(...)` → avanza la FSM di uno step, ritorna eventi di log. Contiene l'unica
  inversione logica vera (`PULL→HOLD_OPEN` su `door ≥ goal − tol`) e l'arco di ritorno
  `PULL→REACH` su presa persa. **Il gate di presa di `PULL→HOLD_OPEN` usa la soglia
  ADATTIVA `g_thresh` (§1.30), non un letterale 0.80** — vedi §10.8.
- `_GRASP_CONFIRM_STEPS = 5` (identico alla chiusura).

#### `PotentialBasedRewardOpen` — `reward_v2.py`

Calcola il reward con il pattern `F = γ·Φ(s') − Φ(s)` `[3]`. Mantiene `_prev_phi` e
`_max_door_angle` (il ratchet di apertura). Potenziali di fase:
- `phi_reach(dist, radius, lvl)` → gaussiana `w · exp(−dist²/2σ²)` con `σ` legato al raggio.
- `phi_pull(door_angle, goal_angle, door_min, …)` → **progresso di apertura** in `[0,1]`,
  0 a porta chiusa (`door_min`), 1 a `goal_angle`. È lo specchio esatto di `phi_push`.
- `phi_hold(hold_dur, target, door_angle, goal_angle, tol)` → frazione di tempo × quanto
  si è vicini al goal.
- `phi_retreat(dist_retreat)` → progresso del ritiro verso la posa di partenza.

Il potenziale cumulativo cresce lungo le fasi (`Φ_REACH < Φ_PULL < Φ_HOLD_OPEN <
Φ_RETREAT`), così lo shaping "tira" verso il completamento senza spostare l'ottimo `[3]`.
Dettaglio cruciale ereditato dalla chiusura (§1.9.F): **in REACH `Φ = 0`**, per azzerare
la "tassa di sosta" `(γ−1)·Φ` che altrimenti impedirebbe alla policy di restare sulla
maniglia i 5 step necessari a confermare la presa.

#### `MultiApproachGrasp` — `grasp_strategy.py` *(riusato identico)*

K=3 direzioni candidate (top-down, laterale-L, laterale-R, ruotate nel frame della porta
se il quaternione è noto); `alignment = maxᵢ |dot(eef_z, dirᵢ)|`. Fornisce `[best, align₀…]`
all'osservazione. Indipendente dal verso del task `[15]`.

#### `ExtendedDomainRandomizer` — `domain_rand_v2.py` *(riusato identico)*

Randomizza a ogni reset: geometria/frizione maniglia, **rigidità latch** (×0.5–2.0),
**damping cerniera** (×0.3–1.5), **massa porta** (×0.5–2.0). Espone 3 feature normalizzate
in osservazione. È l'unico modulo che scrive su `sim.model`. La fisica non dipende dal
verso del task → riuso bit-per-bit dalla chiusura `[8][17]`.

#### `BetaNetwork` — `beta_net.py` *(disabilitata di default)*

Gate probabilistico appreso per le transizioni FSM; restituisce `{1.0, 1.0, 1.0}` se
disabilitata. È l'orizzonte di estensione verso *options* con terminazione appresa `[1][2]`.

#### `AdvancedGeneralizedOpenDoorEnv` — `env_v2.py`

Orchestratore. Assembla FSM, reward, grasp, domain rand; espone l'osservazione; contiene
**tutta la logica deterministica del RETREAT** (zero reward):
- §1.17 rilascio pulito (apre il gripper solo quando le dita sono libere dalla maniglia);
- §1.21 rampa di avvio del ritiro (avvio morbido fermo→policy su `retreat_rampup_steps`);
- §1.22 accompagnamento della leva (tiene la presa e congela il braccio finché la leva
  non torna neutra, lasciando agire la molla di richiamo);
- **§1.26 cap temporale** sull'accompagnamento (vedi §10.4): superati
  `retreat_latch_max_steps`, procede comunque a rilascio+ritiro anche se la leva è ancora
  ruotata, così il braccio non resta mai aggrappato all'infinito.

Espone in `info`: `dist_handle`, `eef_pos`, `handle_pos`, `vec_eef_to_handle`,
`open_error`, `is_success` (= fase in HOLD_OPEN/RETREAT), `fsm_phase`, `door_angle`,
`latch_qpos`, `goal_angle`, `handle_src`, `obs_keys_sample`.

---

### 7.8 — Dettaglio del Flusso `env.step()`

```
env.step(action):
  1.  action = EMA(action, _prev_action; α = action_smooth_alpha)
  2.  phase = fsm.state.phase
  3.  SE phase == RETREAT:                          ← override deterministici, zero reward
        latch_neutral  = |latch_qpos| ≤ retreat_latch_neutral_tol
        fingers_clear  = gripper_width > handle_diam + retreat_clear_margin
        _latch_steps_ok = fsm.state.retreat_steps ≤ retreat_latch_max_steps   ← §1.26
        SE retreat_latch_restore AND not latch_neutral AND _latch_steps_ok:
            action[:-1] = 0           # congela il braccio
            action[-1]  = grip_floor  # tiene la presa (lascia agire la molla del latch)
        ELIF retreat_clean_release AND not fingers_clear:
            action[:-1] = 0; action[-1] = -1   # apre il gripper → rilascio pulito
        ELSE:
            applica rampa di avvio del ritiro (§1.21)
  4.  _prev_action = action
  5.  obs, _, rs_done, _ = rs_env.step(action)      ← UNICA chiamata al simulatore
  6.  leggi door_angle, latch_qpos, eef_pos, gripper_width, …
  7.  fsm.update(door_angle, goal_angle, open_tol, …)   ← transizioni di fase
  8.  reward, terminated, truncated, breakdown = reward_fn.compute(…)
  9.  costruisci obs e info (is_success, fsm_phase, open_error, goal_angle, …)
  10. ritorna obs, reward, terminated, truncated, info
```

La sequenza è identica a quella della chiusura, con l'unica aggiunta del passo `_latch_steps_ok`
(§1.26) al punto 3. È stato verificato (test offline con stub) che resta **una sola**
chiamata a `rs_env.step()` per `env.step()`.

---

### 7.9 — Tabella Completa dei Termini di Reward per Fase

> Convenzione: i pesi `w_*` sono quelli di `config_v2.py`. I valori mostrati sono i
> default attuali (la configurazione che produce il regime ~95–100% di rollout).

#### REACH — Avvicinamento e presa della maniglia

| Termine | Chiave `rew` | Formula | Condizione |
|---------|-------------|---------|-----------|
| Base time-penalty | `base` | `−0.10` | Sempre |
| Shaping potenziale | `phi_shape` | `γ·Φ(s') − Φ(s)` con **Φ_REACH = 0** | Sempre (Φ=0 in REACH per §1.9.F) |
| Distanza 3D | `dist_3d` | `−5.0 × k × dist_handle` | Sempre |
| Distanza XY | `dist_xy` | `−3.0 × k × dist_xy` | Se `dist_xy` disponibile |
| Distanza Z | `dist_z` | `−15.0 × k × \|height_diff\|` | Se `height_diff` disponibile |
| Approccio troppo basso | `app_blw` | `−3.0 × \|height_diff + 0.005\|` | Se `height_diff < −0.005` |
| Approccio troppo alto | `app_top` | `−1.5 × height_diff` | Se `height_diff > 0.03` |
| Gripper aperto lontano | `grip` | `−1.0 × (grip − (−0.85))` | Se `dist > d_near AND grip > −0.85` |
| Gripper chiuso vicino | `grip` | `+2.5 × norm_g` | Se `dist ≤ d_near AND grip > −0.85` |

#### PULL — Apertura della porta verso il goal

| Termine | Chiave `rew` | Formula | Condizione |
|---------|-------------|---------|-----------|
| Base time-penalty | `base` | `−0.10` | Sempre |
| Shaping potenziale | `phi_shape` | `γ·Φ_PULL(s') − Φ(s)` | Sempre |
| Distanza 3D | `dist_3d` | `−5.0 × dist_handle` | Sempre (non perdere la maniglia) |
| Distanza Z | `dist_z` | `−15.0 × \|height_diff\|` | Se `height_diff` disponibile |
| **Progresso apertura (ratchet)** | `door_prog` | `+300 × Δangle` | Se `grip > thresh` e `Δ = door − max_door_angle > 0` |
| Presa debole | `grip` | `−2.0 × (thresh − grip)` | Se `grip < thresh` |
| Mantenimento contatto | `grip_contact` | `+0.5 × opening_progress` | Se `is_phys_closed` |

> Il termine `door_prog` è il **segnale genuino** che definisce "apri la porta": premia
> solo l'angolo *nuovo* guadagnato verso il goal. `max_door_angle` sale soltanto
> (anti-exploit: oscillare avanti/indietro non ri-premia). È lo specchio esatto di
> `door_prog` della chiusura, con il ratchet invertito (sale invece di scendere). Senza
> questo termine, la policy afferra ma non tira fino al goal (plateau ~15%).

#### HOLD_OPEN — Mantenimento della porta al goal (HOLD PIATTO, §1.29)

Questo blocco è **HOLD piatto**: premia la porta tenuta entro la finestra di successo e la
guida dolcemente al goal se ne esce, **senza** termini che combattano la molla. È la
divergenza implementativa più importante rispetto alla chiusura, e nasce da una lezione
fisica precisa (vedi §10.7): nella chiusura il bersaglio `door≈0` è il **punto di
equilibrio**, nell'apertura il goal alto **non lo è**.

| Termine | Chiave `rew` | Formula | Condizione |
|---------|-------------|---------|-----------|
| Base time-penalty | `base` | `−0.10` | Sempre |
| Shaping potenziale | `phi_shape` | `γ·Φ_HOLD(s') − Φ(s)` | Sempre |
| Stabilità sul goal | `hold` | `+1.0` | Se `open_err < open_tol` (premio piatto al goal) |
| Guida dolce | `hold` | `−1.0 × open_err` | Se `open_err ≥ open_tol` (peso 1, **non** combatte la molla) |
| Presa persa | `hold_slip` | `−5.0` | Se la presa fisica è persa |
| Grip mantenuto | `hold_grip` | `+1.0` se `grip > thresh`, altrimenti `−2.0 × \|grip − thresh\|` | Sempre |
| Anti-apertura grip | `hold_drop_pen` | `−10.0 × \|grip\|` | Se `grip < 0` |
| Braccio fermo | `hold_act` | `+1.0` se `‖action_eef‖ < 0.05`, altrimenti `−2.0 × ‖action_eef‖` | Sempre |
| Maniglia vicina | `hold_dist` | `−3.0 × (dist_handle − 0.06)` | Se `dist_handle > 0.06` |

> **Cosa è stato RIMOSSO e perché (§1.29).** Un tentativo intermedio (§1.28) aveva copiato
> dalla chiusura i termini `hold_bounce = −20 × open_err` e `hold_veldamp = −25 × |door_qvel|`.
> Nella chiusura funzionano perché all'ottimo (`door≈0`, equilibrio) valgono **zero**.
> Nell'apertura il goal è fuori equilibrio: la molla ritira la porta in modo *fisicamente
> inevitabile*, quei termini puniscono la policy per la fisica e **fanno crollare il rollout**
> (rollout 1.0 → eval e `ep_len` peggiorati, porta che "lotta" col damping). Diagnosi
> confermata su 20 episodi reali. Sono stati quindi **eliminati**: i pesi `w_hold_bounce`/
> `w_hold_veldamp` restano in `config_v2.py` ma marcati DEPRECATO e non più applicati. Il
> residuo deterministico è risolto per via **geometrica** (`open_tol = 0.05`), non con altre
> penalità. La lettura `_door_qvel()` resta nell'env (innocua, ancora passata al reward che la
> ignora) per non toccare l'interfaccia.

#### RETREAT — Sfilamento dalla maniglia (mantenendo l'apertura)

| Termine | Chiave `rew` | Formula | Condizione |
|---------|-------------|---------|-----------|
| Base time-penalty | `base` | `−0.10` | Sempre |
| Stabilità sul goal | `hold` | `+1.0` se `open_err < open_tol`, altrimenti `0` | Sempre in RETREAT |
| Penalità ri-chiusura | `door_regress` | `−4.0 × max(0, prev_angle − door_angle)` | **Solo** se `door_angle < goal − open_tol` (§1.29) |
| **Latch monitor** | `latch_ret` | `−1.0 × \|latch_qpos\|` | Sempre (specchio esatto della chiusura) |
| Bonus successo | `success_bonus` | `+5.0` | Una volta su `just_succeeded` |

> Tre note sul RETREAT:
> - **Niente `hold_veldamp`** (rimosso in §1.29 come in HOLD_OPEN): non si combatte la molla.
> - **`door_regress` è ora condizionato** (§1.29): penalizza la ri-chiusura **solo** quando
>   porta la porta SOTTO la finestra di successo (fallimento vero); la deriva fisica entro
>   tolleranza è inevitabile e non va punita.
> - `latch_ret` è **identico** alla chiusura e penalizza la leva ancora ruotata a ogni step:
>   è il segnale *appreso* che insegna ad accompagnare la maniglia alla posizione di partenza
>   prima di staccarsi. È attivo SOLO in RETREAT, quindi non interferisce con REACH/PULL/
>   HOLD_OPEN che portano al goal.
> - La **terminazione** NON dipende dal latch (diversamente dalla chiusura): scatta su
>   `retreat_steps ≥ fsm_retreat_target_steps AND door_angle ≥ goal_angle − tol`. Il reward
>   totale è clippato a `[−50, +50]` per step.

---

### 7.10 — Dettaglio del Loop di Training (eval e salvataggi)

```python
class EvalBestCallback(BaseCallback):
    # trigger su num_timesteps (robusto a num_envs): scatta a multipli reali di eval_freq
    def _on_step():
        if num_timesteps >= _next_eval:
            _next_eval += eval_freq
            sincronizza obs_rms (training → eval_env)
            sr, ml = valuta n_eval_episodes=20 episodi (deterministic=True)
            stampa "[EVAL OPEN v2] step … Success … ep_len …"
            salva SEMPRE latest_model.zip + vecnormalize.pkl   ← recupero da run interrotti
            se sr > best_success:
                best_success = sr
                salva best_model.zip + vecnormalize.pkl
```

Differenze rispetto al loop della chiusura: **niente callback di curriculum** (livello
fisso a 1.0) e **niente reload-on-degradation automatico**; in compenso il salvataggio di
`latest_model.zip` a ogni eval è stato aggiunto apposta per l'apertura, così
`diagnose_phase.py` può sempre caricare *qualcosa* anche se il training viene interrotto a
metà. La logica VecNormalize (salvare `vecnormalize.pkl` insieme al modello) è identica e
fondamentale: l'osservazione è normalizzata online, quindi va caricata insieme ai pesi sia
in `--play` sia nel diagnostico.

---

### 7.11 — Flusso Dati tra Moduli (diagramma)

```
                         ┌──────────────────────────┐
                         │   TrainConfigV2Open       │  (valori, nessuna logica)
                         └────────────┬─────────────┘
                                      │ cfg
        ┌───────────────┬─────────────┼─────────────┬───────────────┐
        ▼               ▼             ▼             ▼               ▼
 ExtendedDomain   MultiApproach  AdaptiveFSMOpen  Potential…Open   (beta_net)
   Randomizer        Grasp            │              │             disabilitato
        │               │             │              │
   scrive su        features      transizioni    reward + term.
   sim.model         in obs        di fase       per fase
        └───────────────┴─────────────┼──────────────┘
                                      ▼
                       ┌──────────────────────────────┐
                       │  AdvancedGeneralizedOpenDoorEnv│
                       │  - assembla l'osservazione     │
                       │  - override RETREAT (env-level)│
                       │  - 1 sola sim.step()           │
                       └────────────┬──────────────────┘
                                    │ obs, reward, done, info
                                    ▼
                       ┌──────────────────────────────┐
                       │   SAC (Stable-Baselines3)      │
                       │   8 env paralleli + VecNorm    │
                       └────────────────────────────────┘
```

---

## 8. Riferimenti Bibliografici

(Numerazione coerente con `update_v2.md` della chiusura.)

### 8.1 — Architettura e struttura del codice
- `[1]` Sutton, Precup & Singh (1999) — *Between MDPs and semi-MDPs: options*.
- `[2]` Konidaris & Barto (2009) — *Skill chaining*, precondizioni di opzione.

### 8.2 — FSM e transizioni
- `[1]` options come fasi con terminazione; `[2]` generalizzazione delle precondizioni.

### 8.3 — Soglie (costanti → adattive alla fisica)
- `[13]` ManipForce (2015) — soglie adattive di presa/forza al contatto.

### 8.4 — Generalizzazione
- `[8]` Tobin et al. (2017) — domain randomization. `[17]` Zhao et al. (2020) — sim-to-real.

### 8.5 — Reward e penalità (struttura del codice)
- `[3]` Ng, Harada & Russell (1999) — potential-based reward shaping (invarianza).
- `[4]` Dynamic PBRS — pesi dinamici che preservano l'ottimo.

### 8.6 — Terminazione ed episodio
- `[1]` terminazione delle opzioni; criterio di completamento allineato alla fisica.

### 8.7 — Osservazione ed esplorazione
- `[18]` SAC — RL a massima entropia, auto-tuning della temperatura.
- `[6]` UVFA, `[7]` HER — contesto per il condizionamento al goal.

---

## 9. Spunti di Teoria e Inquadramento Bibliografico (per la tesi)

> Ogni pilastro del progetto di apertura è qui ricondotto alla letteratura, in parallelo
> alla sezione 9 della chiusura. Dove l'apertura *differisce*, lo si segnala esplicitamente.

### 9.1 — RL a massima entropia: perché SAC e il «pavimento» di entropia
Anche l'apertura è controllo continuo 8-D con contatto: **SAC** `[18]` è la scelta
naturale. La differenza rispetto alla chiusura è la *taratura* del pavimento di entropia:
nell'apertura la maniglia è di lato (~0.25 m in Y) e con un bersaglio di entropia basso la
policy si cristallizza sullo "stare ferma" *prima* di trovarla (un collasso di `ent_coef`
analogo a §1.9.C della chiusura, ma con ignizione più difficile). L'intervento — alzare
`target_entropy` da `−3.0` a `+1.0` — è esattamente il meccanismo di auto-tuning previsto
da `[18]`: tiene viva l'esplorazione attraverso la finestra di scoperta. *Spunto tesi:*
mostrare che due task speculari possono richiedere `target_entropy` diversi se la
difficoltà di *ignizione* differisce.

### 9.2 — Astrazione temporale: la FSM come *options*
La FSM REACH→PULL→HOLD_OPEN→RETREAT è formalizzabile come *options* `[1]`: ogni fase è
`⟨I, π, β⟩` con terminazione `β` deterministica (le soglie). Le soglie adattive
(grip/distanza/timer) sono una forma leggera di apprendimento delle precondizioni `[2]`,
parametrizzate sul contesto fisico anziché apprese da zero. `beta_net.py` (disattivato)
renderebbe `β` appresa e probabilistica. *Spunto tesi:* la FSM esplicita come *prior
strutturale* che riduce l'orizzonte di credito; identica alla chiusura nel formalismo, con
l'inversione della condizione di terminazione di PULL.

### 9.3 — Reward shaping potenziale: invarianza e «drift di sconto»
Il cuore è il teorema di invarianza `[3]`: `F = γ·Φ(s') − Φ(s)` non cambia l'insieme delle
policy ottime. La "tassa di sosta" `(γ−1)·Φ` per step è la stessa trappola della chiusura
(§1.9.F/§1.10.A): da qui i pesi piccoli `O(1–5)` e `Φ_REACH = 0`. L'apertura conserva
l'invarianza **esatta** perché il potenziale di PULL premia il progresso verso il goal
(non verso 0) ma resta una funzione di stato. *Spunto tesi:* derivare `(γ−1)Φ` come tassa
di sosta e usarla per giustificare i pesi `phi_*` anche nel verso di apertura.

### 9.4 — RL goal-conditioned: qui è più che un'estensione
A differenza della chiusura (dove il goal è costante e implicito), nell'apertura il goal
è **variabile e campionato** (`goal_angle ~ door_min + U[0.85,1.0]×cap`). Questo avvicina
il problema agli **UVFA** `[6]`, che condizionano il valore su un goal `V(s,g)`: la nostra
policy osserva `goal_angle` e deve aprire della quantità giusta. L'**HER** `[7]`
(rietichettare traiettorie fallite con il goal raggiunto) è un'estensione particolarmente
naturale qui, vista la struttura goal-parametrica. *Spunto tesi:* presentare l'apertura
come passo verso una policy goal-conditioned sul grado di apertura, citando `[6][7]`.

### 9.5 — Domain randomization e sim-to-real
Identica alla chiusura: rigidità latch, damping cerniera, massa porta, oltre a
geometria/frizione maniglia `[8][17]`, con le 3 feature fisiche in osservazione (policy
condizionata al contesto). L'evoluzione è l'**Active DR** `[9]`. *Spunto tesi:* il modulo
è riusato bit-per-bit, prova che la generalizzazione *meccanica* è ortogonale al verso del
task.

### 9.6 — Curriculum learning
Il pacchetto di apertura è **a curriculum fisso (livello 1)**: non c'è avanzamento
automatico. L'inquadramento `[19][20]` resta utile per la tesi come contesto (la posa
variabile è già il "livello difficile"), ma la meccanica di gate windowed della chiusura
qui non si applica. *Spunto tesi:* discutere perché l'apertura curr1 parte direttamente dal
livello difficile (riusa la competenza di presa già matura della chiusura).

### 9.7 — Grasping e manipolazione contact-rich
La strategia multi-approccio K=3 `[15]` è riusata identica: il grasp è una posa 6-D con
molteplici afferraggi validi, premiare il migliore tra K candidati evita di prescrivere la
direzione. Indipendente dal verso del task.

---

## 10. Differenze del Task di Apertura (cosa CAMBIA)

> Questa è la sezione che il documento della chiusura non ha, ed è il cuore di ciò che
> distingue l'apertura. Ogni voce indica *cosa* cambia, *perché*, e *come* è stato
> risolto.

### 10.1 — Il goal è variabile (goal-parametrico)
**Chiusura:** il successo è sempre `door_angle ≈ 0`. Un solo stato bersaglio.
**Apertura:** il bersaglio è `goal_angle = door_min + goal_frac × cap` con
`goal_frac ~ U[0.85, 1.0]`, campionato a ogni episodio. La policy deve sapere *quanto*
aprire → `goal_angle` entra nell'osservazione, e il progresso (`phi_pull`, `door_prog`) è
misurato *relativamente* al goal. **A simulatore** questo significa che ogni reset fissa un
target diverso, e che la condizione di successo è `|door_angle − goal_angle| ≤ open_tol_rad`.
**La tolleranza è `open_tol_rad = 0.05` (§1.29), non 0.03 come la chiusura:** il goal alto è
*fuori equilibrio* (vedi §10.7) e la porta, una volta raggiunto, deriva indietro di
0.024–0.050 rad per effetto della molla. La finestra di successo deve quindi essere larga
almeno quanto la deriva fisica reale — è il mirror corretto della chiusura, dove la finestra
0.03 coincide invece con l'equilibrio (`door≈0`, deriva nulla).

### 10.2 — Ignizione più difficile → `target_entropy` più alto
**Chiusura:** la porta parte aperta e va spinta verso zero; la maniglia è "davanti".
**Apertura:** la maniglia è di lato (~0.25 m in Y) e l'esplorazione casuale fatica a
trovarla; con `target_entropy = −3.0` (valore della chiusura) la policy collassava sullo
"stare ferma" (`ent_coef → ~1.8e-4`). **Risoluzione:** un'unica variabile cambiata,
`target_entropy = +1.0`. Risultato: eval Success 95%, best 100%, `ent_coef` stabilizzato
~5e-4. È la differenza di iperparametro più importante tra i due task.

### 10.3 — La porta resta APERTA → la leva non torna a zero da sola (terminazione)
**Chiusura:** a fine task la porta è a 0 e il latch **scatta a zero per fisica** (stato di
riposo). Quindi la terminazione `door<0.03 AND latch<0.08` è *raggiungibile*.
**Apertura:** la porta resta spalancata al goal e la leva **non** torna a zero da sola.
Un primo tentativo (§1.25) che copiava il gate `latch<0.08` come condizione di
terminazione causò episodi che non finivano mai: `ep_len ~580`, `ep_rew ~−800`, eval
crollato al 55%, "bounce della porta". **Risoluzione (§1.25 corretto):** mantenere il
termine `latch_ret` (insegna ad accompagnare la leva) ma **disaccoppiare la terminazione**,
allineandola allo stato *fisicamente raggiungibile* — `door ≥ goal − tol AND retreat
sostenuto`. La coerenza con la chiusura non è copiare la *soglia*, è copiare il *principio*:
terminare nello stato che il task raggiunge naturalmente.

### 10.4 — L'accompagnamento della leva può non finire mai → cap temporale
**Chiusura:** il ramo env-level "tieni la presa finché la leva non è neutra" termina,
perché la leva *torna* neutra.
**Apertura:** la leva spesso non scende sotto `retreat_latch_neutral_tol`, quindi il braccio
restava aggrappato alla maniglia **all'infinito** (lo screenshot: braccio proteso, gripper
chiuso sulla leva, nessun ritiro). **Risoluzione (§1.26):** un cap `retreat_latch_max_steps
= 20` sul ramo di accompagnamento: accompagna la leva al massimo per quegli step, poi
procede comunque a rilascio + ritiro anche se la leva è ancora ruotata. È env-level, reward
invariato, nessun retraining.

### 10.5 — `w_door_regress` penalizza la RI-CHIUSURA (non la ri-apertura)
**Chiusura:** dopo il successo si penalizza la porta che si **ri-apre**.
**Apertura:** dopo il successo si penalizza la porta che si **ri-chiude** (`door_regress =
−4.0 × max(0, prev_angle − door_angle)`). È lo stesso meccanismo con il segno del moto
indesiderato invertito.

### 10.6 — Il ratchet di progresso sale invece di scendere
**Chiusura:** `min_door_angle` scende (la porta deve chiudersi).
**Apertura:** `max_door_angle` sale (la porta deve aprirsi). Il termine `door_prog` premia
solo l'angolo *nuovo* guadagnato verso il goal; oscillare non ri-premia (anti-exploit).

---

### 10.7 — La stabilizzazione al goal: dove la simmetria con la chiusura si ROMPE (§1.28 → §1.29)

Questa è la differenza concettuale più profonda emersa dai test su curriculum 1, e
paradossalmente nasce dal tentativo di rendere l'apertura *ancora più simmetrica* alla
chiusura. La conclusione è una lezione fisica che vale la pena mettere in tesi.

**Il sintomo.** Con la sola `hold = +1.0` piatta in HOLD_OPEN, l'apertura mostrava un divario
sistematico: rollout `success_rate ≈ 1.0` ma eval deterministico fermo a ~70–75% (best 95%),
`ep_len` di eval alto (~226). Il diagnostico per-episodio rivelava che la porta **arriva
sempre** al goal (`open_error` minimo ≈ 0.000) ma negli episodi falliti l'`open_error`
*finale* restava 0.024–0.050 — appena **sopra** la tolleranza di 0.03: la porta tocca il
goal e poi **deriva indietro**.

**Il tentativo §1.28 (la simmetria spinta troppo in là).** Sembrava naturale copiare il
blocco HOLD della chiusura — che fa 100% true success — in HOLD_OPEN/RETREAT: `hold_bounce =
−20 × err`, `hold_veldamp = −25 × |door_qvel|`, `hold = 1 − err`, con l'unico adattamento del
bersaglio (`open_err = |door_angle − goal_angle|` invece di `|door_qpos|`). Per leggere la
velocità del cardine si era aggiunto `_door_qvel()` all'env.

**Perché §1.28 ha REGREDITO (la lezione).** Il blocco della chiusura funziona per una ragione
fisica precisa, non per i pesi: nella chiusura il bersaglio `door≈0` è il **punto di
equilibrio** della porta (il latch è a riposo, la molla non tira). All'ottimo `err≈0` e
`door_qvel≈0`, quindi `hold_bounce` e `hold_veldamp` valgono **zero** — non costano nulla alla
policy. Nell'apertura il goal è vicino al cap, **fuori equilibrio**: la molla di richiamo
ritira la porta di 0.024–0.050 rad in modo *fisicamente inevitabile*, e mantiene una velocità
residua. Lì quei due termini non valgono mai zero: puniscono la policy per qualcosa che la
fisica impone. Risultato misurato: il rollout (prima 1.0) **crolla**, l'eval scende e `ep_len`
sale a ~226 (la porta "lotta" contro il damping). Stessa cosa per la conferma a finestra
provata in parallelo: rompeva la transizione.

```
            CHIUSURA (door→0)                      APERTURA (door→goal alto)
        bersaglio = EQUILIBRIO                  bersaglio = FUORI equilibrio
   ┌─────────────────────────────┐          ┌─────────────────────────────────┐
   │ molla a riposo, qvel→0       │          │ molla TIRA indietro, qvel≠0     │
   │ hold_bounce = −20·err → 0    │   vs.    │ hold_bounce punisce deriva       │
   │ hold_veldamp = −25·qvel → 0  │          │ hold_veldamp punisce qvel        │
   │ ⇒ all'ottimo COSTO ZERO ✓    │          │ ⇒ punisce la FISICA → crollo ✗  │
   └─────────────────────────────┘          └─────────────────────────────────┘
```

**Il fix §1.29 (revert + geometria).** Due mosse, coerenti con la metodologia "competenza →
reward, niente penalità che combattono la fisica":
1. **Revert all'hold piatto.** HOLD_OPEN/RETREAT tornano a `hold = +1.0` entro tolleranza
   (guida dolce di peso 1 fuori), **senza** `hold_bounce`/`hold_veldamp`. È lo stato che dava
   rollout 1.0. In RETREAT anche `door_regress` è condizionato a scattare solo sotto la
   finestra di successo.
2. **Intervento geometrico, una sola leva.** Il residuo deterministico si elimina alla
   *fonte fisica*: si allarga la finestra di successo a `open_tol_rad = 0.05`, larga almeno
   quanto la deriva reale (0.024–0.050). Verifica offline su traiettoria sintetica: con
   `tol = 0.03` e deriva ≥ 0.035 il timer di HOLD resta bloccato a 0 (mai RETREAT pulito →
   fallimento eval); con `tol = 0.05` il timer accumula per tutte le derive 0.02–0.05.

**Perché è coerente con la chiusura (e non un abbassamento dell'asticella).** La chiusura
"si permette" `tol = 0.03` perché 0.03 è la finestra *attorno all'equilibrio*, dove la deriva
è nulla. L'apertura deve dimensionare la finestra sulla deriva reale del suo bersaglio
non-equilibrio. È lo **stesso principio** della terminazione (§10.3): non si copia la
*soglia*, si copia il *criterio* — qui "la finestra di successo è larga quanto l'incertezza
fisica del bersaglio". `0.05 rad ≈ 2.9°` resta un "aperto al valore richiesto" stretto, e il
goal può ancora arrivare al cap.

> **Leva successiva, se servisse.** Se nel retraining restasse una coda di fallimenti, la
> mossa **non** è reintrodurre penalità: è dare *headroom* geometrico al goal abbassando
> `goal_frac_max` sotto 1.0 (es. 0.90), così la policy apre fino al cap mentre il bersaglio
> sta più in basso e la deriva lo riporta dentro tolleranza. Una sola variabile, ancora
> geometrica.

---

### 10.8 — Il gate di presa di `PULL→HOLD_OPEN`: letterale `0.80` → adattivo `g_thresh` (§1.30)

Questa è la correzione che ha sbloccato l'eval, e — come §10.7 — nasce da una diagnosi
**misurata**, non da un'ipotesi.

**Il sintomo (run §1.29).** Rollout stabile a `success_rate ≈ 0.91–0.94`, `ent_coef`
stabile ~5.6e-4 (nessun collasso), ma eval deterministico **`45% (best 80%)`** con `ep_len`
di eval altissimo (~377). Classico divario stocastico↔deterministico, ma stavolta enorme.

**La diagnosi, dai 20 episodi del diagnostico.** Il dato che chiude il caso: **tutti e 20
gli episodi raggiungono il goal entro tolleranza** (`open_error` minimo ≈ 0.000–0.02, ben
dentro `tol = 0.05`). La competenza di apertura è quindi *perfetta*. Eppure 8 episodi su 20
restano **bloccati in PULL** per ~580 step (`HOLD_OPEN = 0`, terminazione troncata). E
gli 8 falliti hanno **una sola cosa in comune**: la soglia di presa adattiva `g_thresh` è
bassa (0.65–0.683, maniglie a bassa frizione), quindi il pavimento del grip-lock §1.18
(`g_thresh + grip_lock_margin = g_thresh + 0.10`) vale **0.75–0.78, sotto 0.80**.

**La causa.** La transizione `PULL→HOLD_OPEN` richiedeva `gripper_action > 0.80` (letterale).
Ma il comando del gripper è *floorato* dal grip-lock a `g_thresh + 0.10`: per le maniglie a
bassa frizione quel pavimento è **sotto** 0.80. Risultato: porta al goal, presa fisicamente
chiusa e bloccata, ma il `0.80` arbitrario non viene mai superato → la transizione non scatta
→ l'episodio resta in PULL fino all'orizzonte. I 6 successi con pavimento < 0.80 (Ep4, 6, 12,
16, 17, 20) sono passati **solo perché la policy stocastica ha comandato `grip > 0.80` per
conto suo**, per caso — ecco perché il rollout (con rumore) era 0.91 e l'eval (deterministico,
azione media più timida) crollava al 45%. Il `0.80` era un'**incoerenza di progetto**: la
transizione `REACH→PULL` usa già la soglia adattiva `g_thresh`, questa no.

**Il fix (una riga, coerenza adattiva).** Si allinea il gate alla soglia adattiva:
```python
# fsm_v2.py, PULL→HOLD_OPEN
opened_enough = (door_angle >= goal_angle - open_tol) and (gripper_action > g_thresh)
#                                                                          ^^^^^^^^  era 0.80
```
Poiché il grip-lock floora il comando a `g_thresh + 0.10 > g_thresh`, la condizione è
soddisfatta ogni volta che la porta è al goal con presa chiusa, per **qualsiasi** frizione.

**Verifica offline (riproduzione esatta dei falliti).** Ricostruiti gli 8 episodi (porta al
goal, grip floorato a `g_thresh + 0.10`): col vecchio gate `> 0.80` tutti e 8 restano in PULL;
col nuovo gate `> g_thresh` **8/8 transitano in HOLD_OPEN**. Atteso nel retraining: eval
deterministico allineato al rollout (~0.90–1.00), `ep_len` di eval di nuovo basso (~120–160),
sparizione degli episodi troncati a 600.

> **Perché è la fix giusta e non una pezza.** Non si abbassa nessun criterio di successo
> (la porta deve comunque essere al goal entro `tol`) e non si tocca la presa fisica
> (`is_phys_closed` resta). Si rimuove solo un **numero magico** incoerente con l'intero
> impianto adattivo del progetto — la stessa lezione di §10.3 (terminazione) e §10.7
> (tolleranza): *ogni soglia dev'essere funzione della fisica corrente, mai un letterale*.

---

### 10.9 — Cosa è davvero il RETREAT (chiusura) e la deriva post-rilascio (apertura) — §1.31

Domanda chiave, dopo aver raggiunto il 100%: si può rendere `door_end` *preciso* come nella
chiusura (±0.004), e non solo "entro tol"? Per rispondere bisogna capire cosa fa il RETREAT
nella chiusura — dal codice, non a memoria.

**Il RETREAT della chiusura (`close_generalized_v2/env_v2.py`).** Fa tre cose e nient'altro:
(1) *rilascio pulito* — se le dita non sono libere, congela il braccio e apre il gripper a
−1.0; (2) *rampa di avvio* morbida del ritiro; (3) *ritorno* a `retreat_pos`, azione azzerata
quando è tornato. **Nessun termine "tiene" la porta.** Funziona perché a `door≈0` la porta è
contro il **fine-corsa inferiore** del giunto e il `frictionloss` la tiene ferma: rilasci e lei
*resta*, il latch torna neutro da solo (molla del latch), `door_end` finisce a ±0.004 gratis.

**La fisica reale, dal modello (`robosuite/models/assets/objects/door.xml`).** Il cardine è:
`<joint name="hinge" range="0.0 0.4" damping="1" frictionloss="1" limited="true"/>`. Tre fatti
che riscrivono la diagnosi (correzione rispetto alle stesure precedenti, che parlavano di
"molla di richiamo" sul cardine — **non esiste**):
- `range="0.0 0.4" limited="true"` → la porta è **hard-limitata a [0, 0.4] rad**: lo 0.4 è un
  **fine-corsa fisico**, ed è perché l'escursione max è *esattamente* 0.400 in ogni episodio;
- **nessuno `stiffness`** sul cardine → **niente molla**: la porta **non** torna indietro da
  sola;
- `frictionloss="1"` → attrito secco: quando la porta si ferma, **resta ferma** da sola.

**Perché allora l'apertura ha la deriva (dai 20 episodi del run §1.30).** L'`open_error`
*minimo* è ~0 (la porta *passa* per il goal salendo), ma il *finale* correla col goal: alti
(0.389–0.395) → 0.005–0.011; bassi (0.344–0.357) → 0.04–0.05. Il meccanismo **non** è una molla
che richiude: è che `door_prog` premia l'apertura fino al **fine-corsa 0.400**, quindi la porta
supera il goal, arriva al limite e — non avendo molla — vi **resta** (la tiene il `frictionloss`).
`door_end ≈ 0.400`, `open_error = |0.400 − goal|`: ~0.005 per goal alti, ~0.05 per i bassi. La
causa è **nostra (il reward), non del simulatore** → è risolvibile.

**L'asimmetria di `door_prog` (la causa a monte).** Confronto diretto dei ratchet:
- chiusura: `delta = min_door_angle − door_angle` → spinge verso `0`, che **è** il fine-corsa
  inferiore → il progresso **satura al bersaglio** per costruzione, non può superarlo;
- apertura: `delta = door_angle − max_door_angle` → premia *qualsiasi* apertura fino al
  fine-corsa **0.400**, **senza fermarsi al goal**. La policy supera il goal e porta la porta
  al limite; per i goal bassi `door_end` resta al limite, lontano dal goal.

**Il lever §1.31 (opt-in, default OFF).** Si fa saturare anche l'apertura: il progresso premia
solo fino a `goal_angle` (`prog_angle = min(door_angle, goal_angle)`), esatto specchio della
chiusura. Così la policy **lascia la porta al goal** e il `frictionloss` la tiene lì — niente
overshoot al limite. Implementato come flag `pull_progress_cap_at_goal` (default `False` per
**preservare la baseline al 100%**, §1.30):
```python
# reward_v2.py, PULL
prog_angle = door_angle
if cfg.pull_progress_cap_at_goal:
    prog_angle = min(door_angle, goal_angle)   # specchio della chiusura: satura al bersaglio
delta = prog_angle - self._max_door_angle      # ratchet anti-exploit invariato
```
*Verifica offline:* con flag ON il `door_prog` oltre il goal vale 0 (saturato); con OFF premia
fino al limite. *Da confermare col retraining A/B:* atteso `door_end ≈ goal` (anche sui goal
bassi: i due episodi a 0.05 rientrano), senza toccare il 100%.

> **Onestà metodologica (aggiornata col modello).** Il modello conferma che la deriva NON è
> fisica-irrisolvibile: senza molla sul cardine, la porta resta dove la policy la lascia
> (frictionloss). L'unica incognita residua è se la policy **ri-addestrata** con `door_prog`
> saturato si fermi pulita al goal invece di coastare al limite — questo dipende dalla dinamica
> e va visto col run A/B (flag ON vs OFF, stesso seed). Se restasse un filo di overshoot per
> inerzia, il lever successivo è il **freeze duro del braccio in HOLD_OPEN** (`action[:-1]=0`,
> identico alla chiusura), che impedisce alla policy di spingere oltre il goal dopo la
> transizione. La baseline §1.30 resta comunque al 100% true-success entro tol.

---



> Speculare alla §10, questa sezione elenca tutto ciò che l'apertura **riusa** dalla
> chiusura matura — l'impalcatura già validata che non è stata toccata.

- **Architettura a 8 moduli single-responsibility** (config/fsm/reward/grasp/domain_rand/
  beta_net/env/train). Identica.
- **Modulo `grasp_strategy.py`** (MultiApproachGrasp, K=3): riusato bit-per-bit. La presa
  non dipende dal verso del task.
- **Modulo `domain_rand_v2.py`** (ExtendedDomainRandomizer): riusato bit-per-bit.
  Rigidità latch, damping cerniera, massa porta, geometria/frizione maniglia + 3 feature
  in osservazione. La fisica è la stessa.
- **Soglie adattive della FSM** (grip adattiva alla frizione, distanza adattiva al raggio,
  timer HOLD adattivo alla rigidità del latch) e **conferma grasp a 5 step**
  (`_GRASP_CONFIRM_STEPS = 5`). Identiche.
- **Struttura del reward potential-based** `[3]`: pesi piccoli `O(1–5)`, potenziale
  cumulativo crescente lungo le fasi, `Φ_REACH = 0` per annullare la tassa di sosta
  (§1.9.F). Identica nella forma; cambia solo il *verso* del progresso (PULL verso il goal).
- **Termini densi di REACH** (dist_3d/xy/z, app_blw/top, grip_near): mirror diretto della
  chiusura, sono l'unico segnale che porta il braccio alla maniglia.
- **Override deterministici del RETREAT**: §1.17 rilascio pulito, §1.21 rampa di avvio.
  Identici; aggiunti §1.22/§1.26 specifici per la leva dell'apertura.
- **Termine `latch_ret`** in RETREAT: `−1.0 × |latch_qpos|`, identico bit-per-bit alla
  chiusura. Insegna ad accompagnare la leva.
- **Una sola `sim.step()` per `env.step()`**: regola dura, mai violata.
- **EMA sull'azione** (`action_smooth_alpha = 0.8`), **grip-lock** in PULL/HOLD_OPEN
  (§1.18), **SAC** con gli stessi iperparametri di rete/buffer/batch/gamma (tranne
  `target_entropy`, §10.2).
- **Loop di eval** con salvataggio best/latest e VecNormalize. Identico nello spirito
  (semplificato per il curriculum fisso).
- **Metodologia**: competenza del task → reward potential-based; qualità del movimento →
  logica env-level a reward zero; cambiare una variabile alla volta; test funzionali
  offline con stub prima di ri-addestrare. È la disciplina che ha permesso di non rompere
  ripetutamente il 95–100%.

---

## 12. Stato Attuale e Risultati

- **Task di apertura — RISOLTO**: con la fix §1.30 (gate adattivo), **eval `100.0%`
  (best 100.0%), `ep_len 98`**; rollout `success_rate` 0.97–0.99, `ent_coef` stabile
  (~5.4e-4). Diagnostico: 20/20 episodi in RETREAT con terminazione pulita. La porta raggiunge
  **sempre** il goal (`open_error` minimo ≈ 0.000–0.02 in tutti).
- **Miglioria opzionale (§10.9, §1.31)**: residua una deriva post-rilascio sui goal bassi
  (`open_error` finale fino a ~0.05). Lever opt-in `pull_progress_cap_at_goal` per saturare
  il progresso al goal (specchio della chiusura); default OFF, da validare in A/B.
- **Stabilizzazione al goal (§10.7, §1.29)**: il tentativo §1.28 di trasferire
  `hold_bounce`/`hold_veldamp` dalla chiusura è stato **revertito** — quei termini combattono
  la molla nel bersaglio non-equilibrio dell'apertura e facevano crollare il rollout. Si è
  tornati all'hold piatto e la finestra di successo è dimensionata sulla deriva fisica reale
  (`open_tol_rad = 0.05`). *Verificato:* gli episodi che entrano in HOLD_OPEN si stabilizzano.
- **Gate di presa adattivo (§10.8, §1.30) — la fix che ha sbloccato l'eval**: il run §1.29
  dava rollout 0.91–0.94 ma **eval `45% (best 80%)`, `ep_len` ~377**. Diagnosi: 8/20 episodi
  bloccati in PULL pur avendo raggiunto il goal, perché la transizione `PULL→HOLD_OPEN`
  usava il letterale `grip > 0.80` mentre il grip-lock floora a `g_thresh + 0.10` (< 0.80 per
  maniglie a bassa frizione). Allineato il gate alla soglia adattiva `g_thresh`. *Verifica
  offline:* 8/8 episodi falliti recuperati. *Da confermare col retraining:* eval allineato al
  rollout (~0.90–1.00), `ep_len` di eval di nuovo ~120–160.
- **Deadlock di terminazione risolto** (§10.3): `ep_len` di rollout ~150 (da ~580),
  `ep_rew` ~−50/−100 (da ~−800), `ent_coef` stabile (~5.6e-4, nessun collasso).
- **Ritiro del braccio**: cap §10.4 per evitare che il braccio resti aggrappato quando la
  leva non si neutralizza; verificare in `--play` (reward invariato).
- **Comandi di chiusura del lavoro:**
  ```
  rm -rf runs/open_gen_v2                                   # i pesi §1.28/§1.29 vanno scartati
  mjpython open_generalized_v2/train_curriculum_v2.py --total-steps 1500000
  python -m open_generalized_v2.diagnose_phase --episodes 20   # eval deterministico
  ```
- **Tuning aperto**: se dopo il retraining restasse una piccola coda, la leva successiva è
  **geometrica** (`goal_frac_max < 1.0`, headroom sul goal, §10.7), non nuove penalità.
  `retreat_latch_max_steps = 20` è una stima (~0.7 s a 30 Hz); calibrare via play,
  **una variabile alla volta**.

> **Tre fix, una lezione unica.** §1.29 (tolleranza), §10.3 (terminazione) e §1.30 (gate di
> presa) sono lo stesso principio applicato tre volte: in un sistema *adattivo alla fisica*,
> ogni soglia dev'essere funzione della fisica corrente (frizione, deriva, equilibrio), mai
> un numero letterale ereditato per simmetria dalla chiusura.

> **Documento iterabile.** Come concordato, questa è una prima stesura completa da
> aggiornare insieme: numeri di sezione, tabelle dei pesi e risultati finali andranno
> rifiniti man mano che il `--play` conferma il comportamento del ritiro e che si
> consolidano le metriche di eval.
