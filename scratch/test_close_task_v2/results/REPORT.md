# Report — suite di test, task di chiusura porta v2

Le sei sezioni vanno dalla fisica di base alla causa dei risultati:
**0–1** verificano ambiente e simulatore, **2** misura la bravura della policy,
**3** ne osserva il comportamento, **4** ne traccia i limiti, **5** dimostra *quali*
componenti la reggono. <br/>
Un promemoria dei termini statistici:

- **True success** = porta chiusa **e** latch neutro a fine episodio (la metrica "vera",
  più severa del solo "ha raggiunto la fase").
- **Wilson CI** = intervallo di confidenza per una proporzione, affidabile anche con pochi
  campioni e vicino al 100%.
- **IQM** = media interquartile (scarta il 25% estremo per lato): robusta agli anomali.
- **CVaR 10%** = media del 10% peggiore dei casi: misura il **rischio**, non la media.

### Esito complessivo

| Suite | Stato | Tempo (s) |
|---|---|---|
| functional | ✅ ok | 0.0 |
| physics | ✅ ok | 8.9 |
| evaluate | ✅ ok | 108.7 |
| phase | ✅ ok | 18.5 |
| robustness | ✅ ok | 80.1 |
| ablation | ✅ ok | 103.2 |

---

## 0. Functional white-box — rampa di ritiro (§1.21)

**Cosa misura.** Legge il ramo RETREAT direttamente dal sorgente di `env_v2.py` e verifica
7 proprietà di sicurezza della rampa di avvio (il braccio riparte morbido, mai a scatto;
la rampa scala solo il braccio, mai il gripper; l'interruttore la disattiva). Non serve né
robosuite né un modello: è deterministico.

**Esito: 7/7 PASS.** Tutte le proprietà attese sono soddisfatte sul codice consegnato.

---

## 1. Physics unit tests

**Cosa misura.** Che il *simulatore* si comporti correttamente prima di introdurre la
policy: molla del latch (T1), rimbalzo cerniera (T2), e parametri di randomization entro i
range dichiarati (T7). Se questi fallissero, ogni risultato successivo sarebbe inaffidabile.

**Esito: 3/3 PASS.**

| Test | Cosa verifica | Esito |
|---|---|---|
| T1 latch spring | la maniglia torna a neutro entro il tempo limite | PASS |
| T2 hinge bounce | il rimbalzo dopo la chiusura resta limitato | PASS |
| T7 domain randomization | i 5 parametri restano nei range | PASS |

Grafici: `results/physics/`.

---

## 2. Valutazione rigorosa (deterministica + stocastica)

**Cosa misura.** Quanto è brava ciascuna policy, con l'incertezza. Il success rate è
affiancato dal **true success** (più severo); la lunghezza episodio è data come **IQM** e
come **CVaR 10%** (il caso peggiore).

**Come leggerlo.** Entrambi i modelli chiudono al **100%** sia col success permissivo sia
col true success, in entrambe le modalità: la metrica "vera" e quella permissiva
**coincidono**, cioè quando la policy raggiunge la fase, la porta è davvero chiusa e
agganciata. Il CI `[96.3, 100.0]` è l'incertezza residua legata alla dimensione del
campione (200 ep), non un segnale di fallimenti. Gli episodi durano ~120–130 step; il
CVaR appena più alto dell'IQM indica una coda benigna (nessun episodio patologicamente lungo).

| Modello | Modo | Success (95% CI) | True success (95% CI) | Len IQM | Len CVaR 10% |
|---|---|---|---|---|---|
| curr0_posa_fissa | det | 100.0% [96.3, 100.0] | 100.0% [96.3, 100.0] | 127.0 | 135.9 |
| curr0_posa_fissa | sto | 100.0% [96.3, 100.0] | 100.0% [96.3, 100.0] | 128.8 | 136.1 |
| curr1_posa_variabile | det | 100.0% [96.3, 100.0] | 100.0% [96.3, 100.0] | 122.3 | 136.9 |
| curr1_posa_variabile | sto | 100.0% [96.3, 100.0] | 100.0% [96.3, 100.0] | 125.2 | 154.1 |

Grafici: `results/evaluate/`.

---

## 3. Diagnostica delle fasi HOLD/RETREAT (T3–T6)

**Cosa misura.** Non *se* la policy riesce, ma *come*: quanto tenta di muovere il braccio
in HOLD (dovrebbe restare fermo), la torsione del polso in RETREAT, lo stato del latch alla
transizione HOLD→RETREAT, gli eventi di bounce in HOLD.

**Come leggerlo.** Il dato più significativo è `latch@transiz = 100.0% > 0.15` in entrambi
i modelli: **al momento in cui inizia il ritiro, il latch è ancora bloccato**. Non è un
difetto — è la firma della FSM v2, che *non* aspetta il latch neutro per iniziare il RETREAT
ma lo lascia liberare *durante* il ritiro. Gli eventi di bounce (48 e 66) sono contenuti e
coerenti con la porta che si assesta in HOLD.

| Modello | HOLD ‖a‖ IQM | RETREAT polso IQM | latch@transiz (% > 0.15) | bounce |
|---|---|---|---|---|
| curr0_posa_fissa | 0.808 | 0.711 | 100.0% | 48 |
| curr1_posa_variabile | 0.767 | 0.772 | 100.0% | 66 |

Grafici: `results/phase/`.

---

## 4. Inviluppo operativo (robustezza)

**Cosa misura.** Dove la policy regge e dove cede: true success stratificato (con Wilson
CI) lungo i 6 assi fisici v2. Riportiamo il **bin peggiore** di ogni asse — la fascia di
parametri più ostile.

**Come leggerlo.** Anche nei bin peggiori di ogni asse il true success resta al **100%**
(n=25 per bin): la policy non ha un "punto di rottura" all'interno dei range testati, né a
posa fissa né a posa variabile. Il CI complessivo `[97.5, 100.0]` riflette solo la
dimensione del campione. In altre parole: la generalizzazione fisica **e** alla posa
regge su tutto l'inviluppo esplorato.

**curr0_posa_fissa** — true success complessivo: **100.0% [97.5, 100.0]**.
- handle_friction: bin peggiore [0.302, 0.447] → 100.0% (n=25)
- handle_radius: bin peggiore [0.014, 0.017] → 100.0% (n=25)
- latch_stiffness_ratio: bin peggiore [0.500, 0.860] → 100.0% (n=25)
- hinge_damping_ratio: bin peggiore [0.330, 0.524] → 100.0% (n=25)
- door_mass_ratio: bin peggiore [0.516, 0.720] → 100.0% (n=25)
- door_x: bin peggiore [−0.130, −0.127] → 100.0% (n=25)

**curr1_posa_variabile** — true success complessivo: **100.0% [97.5, 100.0]**.
- handle_friction: bin peggiore [0.302, 0.447] → 100.0% (n=25)
- handle_radius: bin peggiore [0.014, 0.017] → 100.0% (n=25)
- latch_stiffness_ratio: bin peggiore [0.500, 0.860] → 100.0% (n=25)
- hinge_damping_ratio: bin peggiore [0.330, 0.524] → 100.0% (n=25)
- door_mass_ratio: bin peggiore [0.516, 0.720] → 100.0% (n=25)
- door_x: bin peggiore [−0.130, −0.127] → 100.0% (n=25)

Grafici: `results/robustness/` (curve 1D + heatmap).

---

## 5. Ablazione §1.17/§1.18/§1.21 (baseline vs toggle)

**Cosa misura.** *Quale* meccanismo v2 tiene su i risultati. Si spegne un contributo alla
volta (e tutti insieme) sulla **stessa** policy, a seed condivisi (confronto appaiato). La
differenza di true success è validata con test di **Fisher**, CI di **Newcombe** e p-value
corretti **Holm-Bonferroni** (la correzione tiene conto dei confronti multipli).

**Come leggerlo — la lettura più interessante del report.**

- Su **curr0 (posa fissa)** spegnere i contributi non degrada in modo significativo: tutti
  i Δ sono entro il rumore (p = 1 dopo correzione). A posa fissa la policy ha margine da
  vendere e non ha bisogno di quelle "stampelle" per riuscire.
- Su **curr1 (posa variabile)** cambia tutto. Togliere il **grip-lock** (§1.18) fa crollare
  il true success di **−30 punti** (p ≈ 0.006 corretto), e spegnere **tutti e tre** i
  contributi lo abbatte di **−36.7 punti** (p ≈ 0.0013). Il rilascio pulito (§1.17) e la
  rampa (§1.21) da soli non spostano l'ago: è il grip-lock il meccanismo dominante.

**Interpretazione.** Il grip-lock impedisce le aperture accidentali del gripper dovute al
rumore di esplorazione durante la chiusura. Nella posa variabile — traiettorie più lunghe e
angolate — quelle aperture accidentali sarebbero frequenti e fatali; a posa fissa il
problema quasi non si presenta. L'ablazione **spiega** perché il 100% osservato nella
batteria 2 non è fragile: poggia su un meccanismo deterministico identificabile.

**curr0_posa_fissa** — baseline true success **100.0% [88.6, 100.0]**.

| Variante | Δ true success | 95% CI | Fisher p | p (Holm) |
|---|---|---|---|---|
| no_clean_release | +0.0 pt | [−11.4, +11.4] | 1 | 1 |
| no_grip_lock | −3.3 pt | [−16.7, +8.3] | 1 | 1 |
| no_rampup | +0.0 pt | [−11.4, +11.4] | 1 | 1 |
| no_all_three | −3.3 pt | [−16.7, +8.3] | 1 | 1 |

**curr1_posa_variabile** — baseline true success **100.0% [88.6, 100.0]**.

| Variante | Δ true success | 95% CI | Fisher p | p (Holm) |
|---|---|---|---|---|
| no_clean_release | +0.0 pt | [−11.4, +11.4] | 1 | 1 |
| no_grip_lock | **−30.0 pt** | [−47.9, −12.5] | 0.00194 | **0.00581** |
| no_rampup | +0.0 pt | [−11.4, +11.4] | 1 | 1 |
| no_all_three | **−36.7 pt** | [−54.5, −18.0] | 0.000319 | **0.00128** |

Forest plot: `results/ablation/`.

---

## Bibliografia

**Metodologia di valutazione**
1. Agarwal et al. (2021) *Deep RL at the Edge of the Statistical Precipice.* NeurIPS 34
   (Outstanding Paper). Libreria `rliable` — IQM, stratified bootstrap CI, prob. of improvement.
2. Colas et al. (2018) *How Many Random Seeds?* arXiv:1806.08295 — power analysis, Welch, bootstrap.
3. Colas et al. (2019) *A Hitchhiker's Guide to Statistical Comparisons of RL Algorithms.* arXiv:1904.06979.
4. Henderson et al. (2018) *Deep RL that Matters.* AAAI — riproducibilità, varianza.
5. Chan et al. (2020) *Measuring the Reliability of RL Algorithms.* ICLR — dispersione (IQR) e rischio (CVaR).
6. Patterson et al. (2024) *Empirical Design in RL.* JMLR 25(318) — disegno controllato, confronti appaiati.
7. Wilson (1927); Brown, Cai, DasGupta (2001) — intervallo di Wilson per proporzioni.
8. Newcombe (1998) — CI della differenza di proporzioni.

**Contributi v2 (ablazionati/diagnosticati)**
9.  Ng, Russell & Harada (1999) *Policy Invariance Under Reward Transformations* — potential-based shaping (§3.2).
10. Devlin & Kudenko (2012) *Dynamic Potential-Based Reward Shaping* (§3.2/§3.6).
11. Sutton, Precup & Singh (1999) *Between MDPs and Semi-MDPs* — opzioni, terminazione β (§3.5), avvio morbido (§1.21).
12. Konidaris & Barto (2009) *Skill Chaining* — soglie FSM adattive (§3.1).
13. Tobin et al. (2017) *Domain Randomization*; Zhao et al. (2020) *Sim-to-Real Survey*;
    Mehta et al. (2020) *Active Domain Randomization* — randomization fisica (§3.4) e inviluppo operativo.
14. ten Pas et al. (2017) *Grasp Pose Detection* — multi-approach grasp (§3.3); stile curve esito-vs-parametro.