# Report suite di test — task di chiusura v1

> Generato automaticamente da `run_all_tests.py`. I numeri sono **stime a intervallo** (non point estimate): è il punto della metodologia (rif. 1–6).

- **Data:** 2026-06-25T13:43:09
- **Repo root:** `/Users/simone/workspace/FrankaPandaRL`  ·  **git:** `3013551`
- **Python:** 3.10.20  ·  **piattaforma:** macOS-26.5.1-arm64-arm-64bit
- **Versioni:** numpy 2.2.6 · scipy 1.15.3 · SB3 2.7.1 · robosuite 1.5.1 · torch 2.10.0
- **Argomenti:** `{"preset": "full", "episodes": null, "curricula": [0.0, 1.0], "run_dir": "runs/close_gen", "suites": ["physics", "evaluate", "phase", "robustness", "ablation"], "no_plots": false}`

## Esito esecuzione

| Suite | Stato | Tempo (s) |
|-------|-------|-----------|
| physics | ✅ ok | 9.1 |
| evaluate | ✅ ok | 246.6 |
| phase | ✅ ok | 34.3 |
| robustness | ✅ ok | 173.8 |
| ablation | ✅ ok | 319.3 |

## 1. Physics unit tests (deterministici)

Esito: **3/3 PASS**.

| Test | Esito |
|------|-------|
| T1 latch spring | PASS |
| T2 hinge bounce | PASS |
| T7 domain randomization | PASS |

- Latch spring: ritorno a neutro in **14** step (stiffness 1.000, damping 0.000).
- Hinge bounce: velocità massima **0.0673 rad/s** (damping 0.100).

Grafici: `results/physics/`.

## 2. Valutazione rigorosa (det + sto)

Success rate con **intervallo di Wilson**; lunghezza/angolo con **IQM + bootstrap CI**; coda peggiore via **CVaR**.

| Curr | Modo | Success (95% CI) | True success (95% CI) | Len IQM | Len CVaR 10% |
|------|------|------------------|------------------------|---------|--------------|
| 0.0 | det | 99.0% [96.4, 99.7] | 97.0% [93.6, 98.6] | 109.6 | 307.1 |
| 0.0 | sto | 99.0% [96.4, 99.7] | 96.5% [93.0, 98.3] | 110.8 | 336.7 |
| 1.0 | det | 98.5% [95.7, 99.5] | 97.0% [93.6, 98.6] | 110.3 | 296.8 |
| 1.0 | sto | 99.5% [97.2, 99.9] | 98.5% [95.7, 99.5] | 111.0 | 227.9 |

Grafici: `results/evaluate/`.

## 3. Diagnostica fasi HOLD/RETREAT (T3–T6)

| Curr | HOLD ‖a‖ IQM | RETREAT polso IQM | latch@transiz (% > 0.15) | bounce |
|------|--------------|-------------------|---------------------------|--------|
| 0.0 | 1.292 | 0.828 | 100.0% | 100 |
| 1.0 | 1.318 | 0.964 | 100.0% | 81 |

Grafici: `results/phase/`.

## 4. Inviluppo operativo (robustezza)

True success per regione dei parametri di domain randomization (stratificazione + Wilson CI). Risponde a *08_risultati_v2 §2/§4*: quantifica DOVE la policy generalizza.

**Curriculum 0.0** — true success complessivo: 99.0% [97.1, 99.7].
  - frizione: bin peggiore [0.302,0.424] → 98.0% (n=50).
  - raggio: bin peggiore [0.017,0.019] → 98.0% (n=50).
  - distanza porta: bin peggiore [-0.126,-0.123] → 98.0% (n=50).

**Curriculum 1.0** — true success complessivo: 98.7% [96.6, 99.5].
  - frizione: bin peggiore [0.601,0.756] → 96.0% (n=50).
  - raggio: bin peggiore [0.017,0.019] → 98.0% (n=50).
  - distanza porta: bin peggiore [-0.127,-0.123] → 94.0% (n=50).

Grafici: `results/robustness/` (curve 1D + heatmap frizione×raggio).

## 5. Studio di ablazione (baseline vs interventi)

Confronto **appaiato** (stessi seed), test di **Fisher** + CI di **Newcombe**, p-value corretti con **Holm-Bonferroni**.

**Curriculum 0.0** — baseline true success 96.0% [86.5, 98.9].

| Variante | Δ true success | 95% CI | Fisher p | p (Holm) |
|----------|----------------|--------|----------|----------|
| freeze_cond_latch | -90.0 pt | [-94.9, -76.1] | 4.87e-22 | 1.95e-21 |
| hold_freeze | +2.0 pt | [-7.0, +11.6] | 1 | 1 |
| hold_freeze_grip | +0.0 pt | [-9.9, +9.9] | 1 | 1 |
| wait_latch | -92.0 pt | [-96.1, -78.6] | 2.98e-23 | 1.49e-22 |
| override_grip | +2.0 pt | [-7.0, +11.6] | 1 | 1 |

**Curriculum 1.0** — baseline true success 100.0% [92.9, 100.0].

| Variante | Δ true success | 95% CI | Fisher p | p (Holm) |
|----------|----------------|--------|----------|----------|
| freeze_cond_latch | -82.0 pt | [-90.2, -67.3] | 2.49e-19 | 9.96e-19 |
| hold_freeze | +0.0 pt | [-7.1, +7.1] | 1 | 1 |
| hold_freeze_grip | +0.0 pt | [-7.1, +7.1] | 1 | 1 |
| wait_latch | -94.0 pt | [-97.9, -81.5] | 4.64e-25 | 2.32e-24 |
| override_grip | -4.0 pt | [-13.5, +3.7] | 0.495 | 1 |

Forest plot: `results/ablation/`.

---

## Bibliografia (metodologia di valutazione)

1. Agarwal R., Schwarzer M., Castro P. S., Courville A. C., Bellemare M. G. (2021).
   *Deep Reinforcement Learning at the Edge of the Statistical Precipice.*
   Advances in Neural Information Processing Systems 34, pp. 29304–29320 (NeurIPS,
   Outstanding Paper Award). Libreria `rliable`. — IQM, stratified bootstrap CI,
   performance profiles, probability of improvement, optimality gap.
2. Colas C., Sigaud O., Oudeyer P.-Y. (2018). *How Many Random Seeds? Statistical
   Power Analysis in Deep Reinforcement Learning Experiments.* arXiv:1806.08295.
3. Colas C., Sigaud O., Oudeyer P.-Y. (2019). *A Hitchhiker's Guide to Statistical
   Comparisons of Reinforcement Learning Algorithms.* arXiv:1904.06979.
4. Henderson P., Islam R., Bachman P., Pineau J., Precup D., Meger D. (2018).
   *Deep Reinforcement Learning that Matters.* AAAI 2018.
5. Chan S. C. Y., Fishman S., Korattikara A., Canny J., Guadarrama S. (2020).
   *Measuring the Reliability of Reinforcement Learning Algorithms.* ICLR 2020.
   Libreria `rl-reliability-metrics`. — dispersione (IQR) e rischio (CVaR).
6. Patterson A., Neumann S., White M., White A. (2024). *Empirical Design in
   Reinforcement Learning.* Journal of Machine Learning Research 25(318):1–63.
7. Wilson E. B. (1927); Brown L. D., Cai T. T., DasGupta A. (2001). *Interval
   Estimation for a Binomial Proportion.* Statistical Science 16(2):101–133.
8. Newcombe R. G. (1998). *Interval estimation for the difference between independent
   proportions.* Statistics in Medicine 17:873–890.
9. Tobin J. et al. (2017). *Domain Randomization for Transferring Deep Neural Networks
   from Simulation to the Real World.* IROS 2017. — usato per l'inviluppo operativo.
10. Mehta B. et al. (2020). *Active Domain Randomization.* CoRL 2020.
11. Zhao W., Queralta J. P., Westerlund T. (2020). *Sim-to-Real Transfer in Deep
    Reinforcement Learning for Robotics: a Survey.* IEEE SSCI 2020.
12. ten Pas A., Gualtieri M., Saenko K., Platt R. (2017). *Grasp Pose Detection in
    Point Clouds.* IJRR — stile delle curve "esito vs parametro".
