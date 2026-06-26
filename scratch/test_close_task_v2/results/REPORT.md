# Report suite di test — task di chiusura generalizzato v2

> Generato da `run_all_tests.py`. I numeri sono **stime a intervallo** (rif. 1–8). Le ablazioni toggolano i contributi v2 §1.17/§1.18/§1.21 sulla stessa policy.

- **Data:** 2026-06-26T14:23:24  ·  **git:** `7588c0d`
- **Repo root:** `/Users/simone/workspace/FrankaPandaRL`
- **Python:** 3.10.20  ·  **piattaforma:** macOS-26.5.1-arm64-arm-64bit
- **Versioni:** numpy 2.2.6 · scipy 1.15.3 · SB3 2.7.1 · robosuite 1.5.1 · torch 2.10.0
- **Argomenti:** `{"preset": "standard", "episodes": null, "models": ["curr0", "curr1"], "suites": ["functional", "physics", "evaluate", "phase", "robustness", "ablation"], "no_plots": false}`

## Esito esecuzione

| Suite | Stato | Tempo (s) |
|---|---|---|
| functional | ✅ ok | 0.0 |
| physics | ✅ ok | 8.9 |
| evaluate | ✅ ok | 108.7 |
| phase | ✅ ok | 18.5 |
| robustness | ✅ ok | 80.1 |
| ablation | ✅ ok | 103.2 |

## 0. Functional white-box — rampa di ritiro §1.21

Esito: **7/7 PASS** (no modello, deterministico).

## 1. Physics unit tests

Esito: **3/3 PASS**.

| Test | Esito |
|---|---|
| T1 latch spring | PASS |
| T2 hinge bounce | PASS |
| T7 domain randomization | PASS |

Grafici: `results/physics/`.

## 2. Valutazione rigorosa (det + sto)

| Modello | Modo | Success (95% CI) | True success (95% CI) | Len IQM | Len CVaR 10% |
|---|---|---|---|---|---|
| curr0_posa_fissa | det | 100.0% [96.3, 100.0] | 100.0% [96.3, 100.0] | 127.0 | 135.9 |
| curr0_posa_fissa | sto | 100.0% [96.3, 100.0] | 100.0% [96.3, 100.0] | 128.8 | 136.1 |
| curr1_posa_variabile | det | 100.0% [96.3, 100.0] | 100.0% [96.3, 100.0] | 122.3 | 136.9 |
| curr1_posa_variabile | sto | 100.0% [96.3, 100.0] | 100.0% [96.3, 100.0] | 125.2 | 154.1 |

Grafici: `results/evaluate/`.

## 3. Diagnostica fasi HOLD/RETREAT (T3–T6)

| Modello | HOLD ‖a‖ IQM | RETREAT polso IQM | latch@transiz (% > 0.15) | bounce |
|---|---|---|---|---|
| curr0_posa_fissa | 0.808 | 0.711 | 100.0% | 48 |
| curr1_posa_variabile | 0.767 | 0.772 | 100.0% | 66 |

Grafici: `results/phase/`.

## 4. Inviluppo operativo (robustezza)

True success per regione dei parametri (stratificazione + Wilson CI), 6 assi v2.

**curr0_posa_fissa** — true success complessivo: 100.0% [97.5, 100.0].
  - handle_friction: bin peggiore [0.302,0.447] → 100.0% (n=25).
  - handle_radius: bin peggiore [0.014,0.017] → 100.0% (n=25).
  - latch_stiffness_ratio: bin peggiore [0.500,0.860] → 100.0% (n=25).
  - hinge_damping_ratio: bin peggiore [0.330,0.524] → 100.0% (n=25).
  - door_mass_ratio: bin peggiore [0.516,0.720] → 100.0% (n=25).
  - door_x: bin peggiore [-0.130,-0.127] → 100.0% (n=25).

**curr1_posa_variabile** — true success complessivo: 100.0% [97.5, 100.0].
  - handle_friction: bin peggiore [0.302,0.447] → 100.0% (n=25).
  - handle_radius: bin peggiore [0.014,0.017] → 100.0% (n=25).
  - latch_stiffness_ratio: bin peggiore [0.500,0.860] → 100.0% (n=25).
  - hinge_damping_ratio: bin peggiore [0.330,0.524] → 100.0% (n=25).
  - door_mass_ratio: bin peggiore [0.516,0.720] → 100.0% (n=25).
  - door_x: bin peggiore [-0.130,-0.127] → 100.0% (n=25).

Grafici: `results/robustness/` (curve 1D + heatmap).

## 5. Ablazione §1.17/§1.18/§1.21 (baseline vs toggle)

Confronto **appaiato**, Fisher + Newcombe, p-value Holm-Bonferroni.

**curr0_posa_fissa** — baseline true success 100.0% [88.6, 100.0].

| Variante | Δ true success | 95% CI | Fisher p | p (Holm) |
|---|---|---|---|---|
| no_clean_release | +0.0 pt | [-11.4, +11.4] | 1 | 1 |
| no_grip_lock | -3.3 pt | [-16.7, +8.3] | 1 | 1 |
| no_rampup | +0.0 pt | [-11.4, +11.4] | 1 | 1 |
| no_all_three | -3.3 pt | [-16.7, +8.3] | 1 | 1 |

**curr1_posa_variabile** — baseline true success 100.0% [88.6, 100.0].

| Variante | Δ true success | 95% CI | Fisher p | p (Holm) |
|---|---|---|---|---|
| no_clean_release | +0.0 pt | [-11.4, +11.4] | 1 | 1 |
| no_grip_lock | -30.0 pt | [-47.9, -12.5] | 0.00194 | 0.00581 |
| no_rampup | +0.0 pt | [-11.4, +11.4] | 1 | 1 |
| no_all_three | -36.7 pt | [-54.5, -18.0] | 0.000319 | 0.00128 |

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
