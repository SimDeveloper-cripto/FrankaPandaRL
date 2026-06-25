# Suite di test scientifica — task di chiusura porta (v1)

Riscrittura rigorosa della batteria di test in `scratch/test_close_task_v1/`.
Gli script originali (scritti a mano) misuravano il giusto, ma con *point estimate*
(media di pochi episodi) e senza incertezza. Questa versione adotta le pratiche di
valutazione raccomandate dalla letteratura di Deep RL: **stime a intervallo**, metriche
**robuste**, metriche di **rischio**, **inviluppo operativo** della generalizzazione e
**confronti statistici** appaiati per le ablazioni.

Tutto è eseguibile con **un comando** e produce JSON + grafici + un `REPORT.md` aggregato.

---

## 0. «La comunità scientifica ha prodotto dei test da fare?» — sì

Il riferimento centrale è **Agarwal et al. 2021** (NeurIPS, *Outstanding Paper*), che ha
fatto da spartiacque: valutare un agente con media/mediana di pochi run è inaffidabile;
vanno riportati **intervalli** e metriche robuste (IQM, performance profiles, probability
of improvement). Da lì la libreria `rliable`. A corredo:

| Tema | Riferimento | Cosa impone | Dove lo usiamo |
|------|-------------|-------------|----------------|
| Stime a intervallo, IQM, bootstrap, prob. of improvement | **Agarwal et al. 2021** [1] | non point estimate | `stats_utils`, ovunque |
| Quanti seed/episodi, test t di Welch, bootstrap | **Colas et al. 2018** [2] | power analysis | `required_n_for_proportion`, confronti |
| Confronti corretti (multipli) | **Colas et al. 2019** [3] | Holm-Bonferroni | `ablation_study` |
| Riproducibilità, varianza | **Henderson et al. 2018** [4] | seed fissati, niente claim da pochi run | seeding, metadati |
| Affidabilità: dispersione + **rischio** | **Chan et al. 2020** [5] | IQR, CVaR | `cvar`, distribuzioni per-episodio |
| Disegno sperimentale controllato | **Patterson et al. 2024** [6] | confronto appaiato | ablazioni a seed condivisi |
| CI di proporzione (p≈1) | **Wilson 1927 / Brown 2001** [7] | Wilson ≫ Wald | `wilson_ci` |
| Differenza di proporzioni | **Newcombe 1998** [8] | CI della differenza | `compare_proportions` |
| Caratterizzare la generalizzazione | **Tobin 2017** [9], **Mehta 2020** [10], **Zhao 2020** [11] | sweep dei parametri | `robustness_analysis` |
| Curve «esito vs parametro» | **ten Pas et al. 2017** [12] | inviluppo operativo | `robustness_analysis` |

Riferimenti completi in fondo (e replicati in `REPORT.md`). I codici [9]–[12] sono già
nella bibliografia del progetto (`update_v2.md`).

---

## 1. Mappa: script originale → modulo nuovo

| Originale | Diventa | Cosa cambia |
|-----------|---------|-------------|
| `eval_stats_close.py` | `evaluate_policy.py` | Wilson CI sul success, IQM+bootstrap su lunghezza/angolo, **CVaR**, true-success, 50→**200** episodi motivati, 6 grafici |
| `diag_phase34.py` (T1,T2) | `physics_unit_tests.py` | da stampa diagnostica a **property test PASS/FAIL** con tolleranze (+ T7 sui range di domain randomization), pytest-compatibile |
| `diag_phase34.py` (T3–T6) | `phase_diagnostics.py` | stesse grandezze ma con IQM/CI/Wilson e istogrammi |
| `test_freeze_logic.py`, `test_hold_freeze.py`, `test_hold_freeze_grip.py`, `test_wait_logic.py`, `test_override_grip.py` | `ablation_envs.py` + `ablation_study.py` | gli interventi diventano uno **studio di ablazione** appaiato con Fisher + Newcombe + Holm-Bonferroni e forest plot |
| `inspect_eval_stats.py`, `inspect_test_model.py` | (utility, invariati) | restano comodi per ispezione manuale |
| — (novità) | `robustness_analysis.py` | **inviluppo operativo**: success vs frizione/raggio/posa con CI + heatmap |
| — (novità) | `run_all_tests.py` | esegue tutto e genera `REPORT.md` |
| — (novità) | `stats_utils.py` | nucleo statistico riusabile (testato in isolamento) |
| — (novità) | `_common.py` | path robusti, caricamento env/modello, rollout seedato |

Il modello di riferimento resta lo stesso: `runs/close_gen/best_model.zip` +
`runs/close_gen/vecnormalize.pkl`.

---

## 2. Le cinque batterie

1. **`physics_unit_tests.py`** — proprietà fisiche **deterministiche** dell'ambiente,
   senza modello. T1 (la molla del latch torna a neutro e in quanti step), T2 (il bounce
   della cerniera è contenuto), T7 (i parametri di domain randomization restano nei range
   dichiarati). Esito PASS/FAIL con tolleranze esplicite.

2. **`evaluate_policy.py`** — valutazione della policy, **deterministica** (eval) e
   **stocastica** (train). Riporta: success rate ± **Wilson**; **true success** (porta
   chiusa `|door|<0.03` *e* latch neutro `|latch|<0.08` a fine episodio — risponde a
   `08_risultati_v2 §4.2`); lunghezza e min-door-angle come **IQM + bootstrap CI**;
   **CVaR** del 10% peggiore; breakdown dei fallimenti con CI.

3. **`phase_diagnostics.py`** — diagnostica HOLD/RETREAT (T3–T6) con statistica:
   norma azione braccio in HOLD, torsione polso in RETREAT, `latch_qpos` alla transizione
   HOLD→RETREAT, eventi di bounce.

4. **`robustness_analysis.py`** — **inviluppo operativo**: stratifica gli episodi per
   parametro realizzato (frizione, raggio, distanza porta) e calcola il true-success per
   regione (± Wilson), più heatmap 2D frizione×raggio. È la risposta quantitativa a
   «dove» la policy generalizza (`08_risultati_v2 §2/§4`).

5. **`ablation_study.py`** — confronto **controllato** baseline vs gli interventi
   deterministici degli ex-`test_*.py`, a **seed condivisi** (appaiato), con Fisher +
   Newcombe + Holm-Bonferroni e forest plot.

---

## 3. Come si lancia

Dalla **radice del progetto** (il venv del progetto attivo). I path sono risolti in modo
robusto, ma lanciare da root è la convenzione.

```bash
# tutto, preset "standard" (100 ep eval, 30 phase, 150 robust, 30 ablation), curriculum 1
python scratch/test_close_task_v1/run_all_tests.py --preset standard

# entrambe le pose (fissa e variabile) come in v2
python scratch/test_close_task_v1/run_all_tests.py --preset full --curricula 0 1

# solo alcune batterie
python scratch/test_close_task_v1/run_all_tests.py --suites evaluate ablation --episodes 80

# preset rapido per un primo giro (numeri indicativi, CI larghi)
python scratch/test_close_task_v1/run_all_tests.py --preset quick
```

Singole batterie (tutte con `--run-dir`, default `runs/close_gen`):

```bash
python scratch/test_close_task_v1/physics_unit_tests.py
python scratch/test_close_task_v1/evaluate_policy.py     --episodes 200 --curriculum 1
python scratch/test_close_task_v1/phase_diagnostics.py   --episodes 30  --curriculum 1
python scratch/test_close_task_v1/robustness_analysis.py --episodes 300 --curriculum 1
python scratch/test_close_task_v1/ablation_study.py      --episodes 50  --curriculum 1
```

I test fisici sono anche `pytest`-compatibili: `pytest scratch/test_close_task_v1/physics_unit_tests.py`.

Il nucleo statistico si auto-verifica senza simulatore: `python scratch/test_close_task_v1/stats_utils.py`.

**Dipendenze:** solo `numpy`, `scipy`, `matplotlib` (oltre a quelle già del progetto:
robosuite, stable-baselines3). `rliable` **non** è richiesta — le metriche (IQM, bootstrap,
prob. of improvement) sono implementate nativamente; se preferisci, puoi sostituirle con
`rliable` mantenendo le stesse formule.

**Tempi indicativi** (1 env, CPU): `quick` pochi minuti; `standard` ~20–40 min;
`full` ~1–2 h (soprattutto evaluate 200×2 e robustness 300).

---

## 4. Output

Tutto sotto `scratch/test_close_task_v1/results/`:

```
results/
├── REPORT.md                 # report aggregato con tabelle, CI e bibliografia
├── run_meta.json             # data, git, versioni librerie, argomenti (riproducibilità)
├── all_results.json          # tutti i risultati (array per-episodio esclusi)
├── physics/                  # physics_results.json + plot_latch_spring/hinge_bounce/friction_hist
├── evaluate/                 # metrics_*.json, episodes_*.json, 6 plot (success, fasi, distribuzioni…)
├── phase/                    # phase_diag_*.json + plot T3/T4/T5/T6
├── robustness/               # robustness_*.json + curve 1D + heatmap
└── ablation/                 # ablation_*.json + forest plot
```

Dammi questa cartella (o anche solo `REPORT.md` + i `*.json`) e aggiorniamo i documenti
(`08_risultati`, ecc.) con i numeri.

---

## 5. Nota scientifica onesta sullo scopo (importante per la tesi)

Con **un solo modello addestrato**, gli intervalli qui calcolati quantificano la
**variabilità a livello di episodio** (domain randomization + policy stocastica): rispondono
a «quanto è brava *questa* policy, con quanta incertezza». È la domanda giusta per
validare un modello consegnato.

Per affermazioni sull'**algoritmo / procedura di training** (es. «v2 è meglio di v1»), la
metodologia di Agarwal/Colas/Henderson richiede **più seed di training** (≥5) e l'aggregazione
*tra run*. La suite è già predisposta: addestrando N seed in `runs/close_gen_seedK/` basta
rilanciare con `--run-dir runs/close_gen_seedK` e aggregare gli IQM/CI tra cartelle (oppure
passare gli array a `rliable`). Dichiarare questo confine è esso stesso buona pratica
(Henderson 2018 [4]; Patterson 2024 [6]).

Inoltre, le ablazioni (`test_*` originali) sono **controllori deterministici post-policy**
applicati in valutazione su una policy addestrata sul baseline: lo studio misura il loro
effetto a parità di policy, **non** un ri-addestramento. È esattamente ciò che testavano
gli script originali, ora con la statistica corretta.

---

## Bibliografia

1. R. Agarwal, M. Schwarzer, P. S. Castro, A. C. Courville, M. G. Bellemare. *Deep
   Reinforcement Learning at the Edge of the Statistical Precipice.* NeurIPS 34:29304–29320,
   2021 (Outstanding Paper). Libreria `rliable`.
2. C. Colas, O. Sigaud, P.-Y. Oudeyer. *How Many Random Seeds? Statistical Power Analysis
   in Deep Reinforcement Learning Experiments.* arXiv:1806.08295, 2018.
3. C. Colas, O. Sigaud, P.-Y. Oudeyer. *A Hitchhiker's Guide to Statistical Comparisons of
   Reinforcement Learning Algorithms.* arXiv:1904.06979, 2019.
4. P. Henderson, R. Islam, P. Bachman, J. Pineau, D. Precup, D. Meger. *Deep Reinforcement
   Learning that Matters.* AAAI 2018.
5. S. C. Y. Chan, S. Fishman, A. Korattikara, J. Canny, S. Guadarrama. *Measuring the
   Reliability of Reinforcement Learning Algorithms.* ICLR 2020. Libreria
   `rl-reliability-metrics`.
6. A. Patterson, S. Neumann, M. White, A. White. *Empirical Design in Reinforcement
   Learning.* JMLR 25(318):1–63, 2024.
7. E. B. Wilson, *Probable Inference, the Law of Succession, and Statistical Inference*,
   JASA 1927; L. D. Brown, T. T. Cai, A. DasGupta, *Interval Estimation for a Binomial
   Proportion*, Statistical Science 16(2):101–133, 2001.
8. R. G. Newcombe. *Interval estimation for the difference between independent proportions.*
   Statistics in Medicine 17:873–890, 1998.
9. J. Tobin et al. *Domain Randomization for Transferring Deep Neural Networks from
   Simulation to the Real World.* IROS 2017.
10. B. Mehta, M. Diaz, F. Golemo, C. Pal, L. Paull. *Active Domain Randomization.* CoRL 2020.
11. W. Zhao, J. P. Queralta, T. Westerlund. *Sim-to-Real Transfer in Deep Reinforcement
    Learning for Robotics: a Survey.* IEEE SSCI 2020.
12. A. ten Pas, M. Gualtieri, K. Saenko, R. Platt. *Grasp Pose Detection in Point Clouds.*
    IJRR 2017.
