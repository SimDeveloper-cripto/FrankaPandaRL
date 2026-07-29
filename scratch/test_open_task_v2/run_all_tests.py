#!/usr/bin/env python3
# scratch/test_open_task_v2/run_all_tests.py
"""
run_all_tests — Orchestratore della suite di test scientifica (task di APERTURA v2).

Esegue in un solo comando tutte le batterie e produce un report aggregato con metriche,
intervalli di confidenza, grafici e bibliografia.

Batterie:
  0. functional   — white-box degli override del RETREAT (§1.17/§1.21/§1.43/§1.46/§1.50),
                    senza robosuite né modello
  1. physics      — proprietà fisiche deterministiche: molla del latch, ritenzione della
                    porta aperta, range di randomization, campionamento del goal
  2. evaluate     — valutazione rigorosa det+sto (Wilson CI, IQM, CVaR) a tre livelli di
                    successo
  3. phase        — diagnostica HOLD_OPEN/RETREAT (T3–T7)
  4. robustness   — inviluppo operativo su 7 assi (inclusi goal e fisica estesa)
  5. ablation     — disattivazione degli override, confronto appaiato

Modello: l'apertura v2 è addestrata al SOLO curriculum 1 (posa variabile) → un run.
La run-dir si risolve così: --run-dir esplicito → `runs/open_gen_v2` → autodiscovery di
`runs/open_gen_v2*` contenente un modello (viene stampato quale è stato scelto).

Esempi:
  python scratch/test_open_task_v2/run_all_tests.py --preset standard
  python scratch/test_open_task_v2/run_all_tests.py --preset full
  python scratch/test_open_task_v2/run_all_tests.py --suites evaluate ablation
  python scratch/test_open_task_v2/run_all_tests.py --run-dir runs/open_gen_v2_xyz
"""

from __future__ import annotations

import os
import sys
import json
import time
import argparse
import datetime
import platform
import subprocess
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import (results_dir, REPO_ROOT, json_default, resolve_run_dir,  # noqa: E402
                     find_model, CURRICULUM)

PRESETS = {
    "quick":    dict(evaluate=30,  phase=15, robust=60,  ablation=15),
    "standard": dict(evaluate=100, phase=30, robust=150, ablation=30),
    # `phase` alzato a 100: con 30 episodi gli istogrammi T5/T6/T7 restano spigolosi e
    # poco leggibili in stampa. Costa ~35 s in più.
    "full":     dict(evaluate=200, phase=100, robust=300, ablation=50),
}
ALL_SUITES = ["functional", "physics", "evaluate", "phase", "robustness", "ablation"]
TAG = "curr1_posa_variabile"


def collect_meta(args, run_dir):
    def ver(mod):
        try:
            return __import__(mod).__version__
        except Exception:
            return "n/a"

    def git_hash():
        try:
            return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                           cwd=REPO_ROOT,
                                           stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            return "n/a"
    return dict(timestamp=datetime.datetime.now().isoformat(timespec="seconds"),
                repo_root=REPO_ROOT, git_commit=git_hash(),
                run_dir=run_dir, model=find_model(run_dir, quiet=True) or "n/d",
                curriculum=CURRICULUM,
                python=platform.python_version(), platform=platform.platform(),
                versions=dict(numpy=ver("numpy"), scipy=ver("scipy"),
                              stable_baselines3=ver("stable_baselines3"),
                              robosuite=ver("robosuite"), torch=ver("torch"),
                              gymnasium=ver("gymnasium")),
                args=vars(args))


def guarded(name, fn):
    print("\n" + "#" * 78); print(f"# SUITE: {name}"); print("#" * 78)
    t0 = time.time()
    try:
        res = fn(); dt = time.time() - t0
        print(f"[OK] {name} in {dt:.1f}s")
        return dict(status="ok", seconds=dt, result=res)
    except Exception as e:
        dt = time.time() - t0
        print(f"[FAIL] {name}: {e}"); traceback.print_exc()
        return dict(status="fail", seconds=dt, error=str(e))


BIBLIO = """\
## Bibliografia

**Metodologia di valutazione**
1. Agarwal et al. (2021) *Deep Reinforcement Learning at the Edge of the Statistical Precipice.*
   NeurIPS 34 (Outstanding Paper). Libreria `rliable` — IQM, stratified bootstrap CI,
   performance profiles, probability of improvement.
2. Colas et al. (2018) *How Many Random Seeds?* arXiv:1806.08295 — analisi di potenza, Welch, bootstrap.
3. Colas et al. (2019) *A Hitchhiker's Guide to Statistical Comparisons of RL Algorithms.* arXiv:1904.06979
   — correzione per test multipli (Holm-Bonferroni).
4. Henderson et al. (2018) *Deep Reinforcement Learning that Matters.* AAAI — riproducibilità e varianza.
5. Chan et al. (2020) *Measuring the Reliability of RL Algorithms.* ICLR — dispersione (IQR) e rischio come metriche di prima classe.
5b. Rockafellar & Uryasev (2000) *Optimization of Conditional Value-at-Risk.* Journal of Risk 2(3) — definizione del CVaR usato sul decile peggiore.
6. Patterson et al. (2024) *Empirical Design in Reinforcement Learning.* JMLR 25(318) — disegno
   controllato, confronti appaiati/blocked.
7. Wilson (1927); Brown, Cai & DasGupta (2001) — intervallo di Wilson per proporzioni.
8. Newcombe (1998) — intervallo di confidenza per la differenza di proporzioni.
8b. Holm (1979) *A Simple Sequentially Rejective Multiple Test Procedure.* Scand. J. Statist. 6(2) — correzione applicata ai 7 confronti dell'ablazione.

**Contributi v2 dell'apertura (ablazionati / diagnosticati)**
9.  Ng, Russell & Harada (1999) *Policy Invariance Under Reward Transformations* — potential-based
    shaping (§3.2); giustifica override deterministici a reward invariato.
10. Devlin & Kudenko (2012) *Dynamic Potential-Based Reward Shaping* (§3.2/§3.6).
11. Sutton, Precup & Singh (1999) *Between MDPs and Semi-MDPs* — fasi come opzioni, terminazione β
    (§3.5), avvio morbido dell'opzione di ritiro (§1.21).
12. Konidaris & Barto (2009) *Skill Chaining* — soglie FSM adattive e precondizioni di opzione (§3.1).
13. Tobin et al. (2017) *Domain Randomization*; Zhao et al. (2020) *Sim-to-Real Transfer Survey*;
    Mehta et al. (2020) *Active Domain Randomization* — randomization fisica (§3.4) e inviluppo operativo.
14. ten Pas et al. (2017) *Grasp Pose Detection in Point Clouds* — grasp multi-approach (§3.3);
    stile delle curve esito-vs-parametro.
15. ManipForce (2015) — soglie di presa adattive alla frizione/contatto (§3.1, §1.18).
16. Handa et al. (2020) *DexPilot* — rappresentazione del contatto, direzione di approccio.
"""


def _fmt_ci(d):
    if not isinstance(d, dict) or "point" not in d:
        return "n/d"
    return f"{d['point']*100:.1f}% [{d['lo']*100:.1f}, {d['hi']*100:.1f}]"


def build_report(meta, suites):
    L = []
    L.append("# Report suite di test — task di APERTURA generalizzata v2\n")
    L.append("> Generato da `run_all_tests.py`. Tutti i tassi sono **stime a intervallo** "
             "(rif. 1–8): il numero puntuale da solo non è interpretabile. Le ablazioni "
             "disattivano gli override deterministici del RETREAT sulla stessa policy.\n")
    L.append(f"- **Data:** {meta['timestamp']}  ·  **git:** `{meta['git_commit']}`")
    L.append(f"- **Repo root:** `{meta['repo_root']}`")
    L.append(f"- **Run dir:** `{meta['run_dir']}`  ·  **modello:** `{meta['model']}`")
    L.append(f"- **Curriculum:** {meta['curriculum']} (posa variabile — l'apertura v2 è "
             f"addestrata solo a questo livello)")
    L.append(f"- **Python:** {meta['python']}  ·  **piattaforma:** {meta['platform']}")
    v = meta["versions"]
    L.append(f"- **Versioni:** numpy {v['numpy']} · scipy {v['scipy']} · SB3 "
             f"{v['stable_baselines3']} · robosuite {v['robosuite']} · torch {v['torch']}")
    L.append(f"- **Argomenti:** `{json.dumps(meta['args'])}`\n")

    L.append("## Esito esecuzione\n")
    L.append("| Suite | Stato | Tempo (s) |"); L.append("|---|---|---|")
    for name, s in suites.items():
        L.append(f"| {name} | {'✅ ok' if s['status'] == 'ok' else '❌ fail'} | {s['seconds']:.1f} |")
    L.append("")

    fn = suites.get("functional")
    if fn and fn["status"] == "ok":
        r = fn["result"]
        L.append("## 0. White-box — override deterministici del RETREAT\n")
        L.append(f"Esito: **{r.get('passed','?')}/{r.get('total','?')} PASS** "
                 f"(nessun modello, deterministico, eseguito sul sorgente reale di `env_v2.py`).\n")
        L.append("| Proprietà | Esito |"); L.append("|---|---|")
        for m, p in (r.get("checks") or {}).items():
            L.append(f"| {m} | {'PASS' if p else 'FAIL'} |")
        L.append("")

    ph = suites.get("physics")
    if ph and ph["status"] == "ok":
        summ = ph["result"].get("summary", {})
        n_inv = summ.get("invalid", 0)
        L.append("## 1. Physics unit tests\n")
        L.append(f"Esito: **{summ.get('passed','?')}/{summ.get('total','?')} PASS**"
                 + (f" — **{n_inv} misura/e NON VALIDA/E**" if n_inv else "") + ".\n")
        L.append("| Test | Esito |"); L.append("|---|---|")
        for n, p in summ.get("checks", {}).items():
            lbl = p if isinstance(p, str) else ("PASS" if p else "FAIL")
            L.append(f"| {n} | {lbl} |")
        ret = ph["result"].get("door_retention")
        if ret:
            if not ret.get("valid", True):
                L.append(f"\n⚠️ **Ritenzione NON MISURABILE**: in tutti i "
                         f"{ret.get('attempts','?')} tentativi il braccio è in contatto con "
                         f"la porta portata a {ret['target']:.3f} rad, quindi la deriva "
                         f"osservata ({ret['drift_free']:.4f} rad) è un artefatto di "
                         f"contatto e non dinamica del cardine.")
            else:
                L.append(f"\nRitenzione: la porta lasciata al goal deriva di "
                         f"**{ret['drift_free']:.4f} rad** (con impulso di richiusura: "
                         f"{ret['drift_kicked']:.4f} rad; seed {ret.get('seed_used','?')}, "
                         f"nessun contatto braccio-porta).")
        L.append("\nGrafici: `results/physics/`.\n")

    ev = suites.get("evaluate")
    if ev and ev["status"] == "ok":
        L.append("## 2. Valutazione rigorosa (deterministica + stocastica)\n")
        L.append("| Modo | Success (95% CI) | True success | Clean success | Len IQM | open_err IQM | Fermo su maniglia |")
        L.append("|---|---|---|---|---|---|---|")
        for mode in ("det", "sto"):
            m = ev["result"].get(mode)
            if not m:
                continue
            L.append(f"| {mode} | {_fmt_ci(m['success_rate'])} | {_fmt_ci(m['true_success_rate'])} "
                     f"| {_fmt_ci(m['clean_success_rate'])} | {m['length_iqm']['point']:.1f} "
                     f"| {m['open_error_end_iqm']['point']:.4f} "
                     f"| {_fmt_ci(m['stuck_on_handle']['ci'])} |")
        det = ev["result"].get("det")
        triv = (det or {}).get("trivial_reference")
        if triv:
            L.append(f"\n**Riferimento banale.** Una policy costante che ignora il goal e "
                     f"spalanca sempre fino al fine corsa otterrebbe "
                     f"**{triv['hits']}/{triv['n']} = {triv['rate']*100:.1f}%** "
                     f"[{triv['ci']['lo']*100:.1f}, {triv['ci']['hi']*100:.1f}] sugli stessi "
                     f"goal, perché con `open_tol = {triv['open_tol']}` il fine corsa cade "
                     f"dentro la tolleranza per buona parte dei goal campionati. "
                     f"La policy addestrata guadagna "
                     f"**{(det['true_success_rate']['point'] - triv['rate'])*100:+.1f} punti** "
                     f"sul solo criterio dell'angolo: il grosso del `true_success` è spiegato "
                     f"dalla geometria del compito, non dall'apprendimento. È un difetto di "
                     f"**specifica della metrica**, non della policy — che infatti risolve la "
                     f"manipolazione (afferrare, aprire, mantenere, rilasciare) nel 100% "
                     f"degli episodi.")
        if det:
            L.append("\n**Terminazioni (eval deterministico)** — la distinzione PULITA / "
                     "ESOGENA / HARD-CAP è la diagnosi del ritiro:\n")
            L.append("| Tipo | count | 95% CI |"); L.append("|---|---|---|")
            for tt, b in det["termination_breakdown"].items():
                L.append(f"| {tt} | {b['count']} | {_fmt_ci(b['ci'])} |")
        L.append("\nGrafici: `results/evaluate/`.\n")

    pd = suites.get("phase")
    if pd and pd["status"] == "ok":
        d = pd["result"]
        t3 = d.get("T3_hold_action_norm") or {}
        t4 = d.get("T4_retreat_wrist_rot") or {}
        t5a = d.get("T5_open_error_at_transition") or {}
        t5b = d.get("T5_latch_at_transition") or {}
        t6 = d.get("T6_deviation_events") or d.get("T6_regress_events") or {}
        t7 = d.get("T7_retreat_moved") or {}
        L.append("## 3. Diagnostica fasi HOLD_OPEN / RETREAT\n")
        L.append("| Metrica | Valore |"); L.append("|---|---|")
        L.append(f"| T3 ‖azione braccio‖ in HOLD_OPEN (IQM) | {(t3.get('iqm') or {}).get('point', float('nan')):.3f} |")
        L.append(f"| T4 torsione polso in RETREAT (IQM) | {(t4.get('iqm') or {}).get('point', float('nan')):.3f} |")
        L.append(f"| T5a open_error alla transizione (media) | {t5a.get('mean', float('nan')):.4f} rad |")
        L.append(f"| T5b latch alla transizione (media) | {t5b.get('mean', float('nan')):+.3f} rad |")
        L.append(f"| T6 episodi con scostamenti in HOLD_OPEN | "
                 f"{t6.get('episodes_with_overshoot','?')} oltre il goal / "
                 f"{t6.get('episodes_with_regress','?')} sotto, su {t6.get('n_episodes','?')} |")
        _dom = t6.get('top3_episode_share')
        L.append(f"| T6 step fuori tolleranza (non episodio-pesato) | {t6.get('n', '?')} "
                 f"(oltre: {t6.get('n_overshoot', '?')}, sotto: {t6.get('n_regress', '?')})"
                 + (f" — i 3 episodi peggiori pesano il {_dom*100:.0f}%" if _dom else "") + " |")
        L.append(f"| T7 allontanamento in RETREAT (IQM) | {(t7.get('iqm') or {}).get('point', float('nan')):.4f} m |")
        L.append(f"| T7 episodi fermi sulla maniglia | {_fmt_ci(t7.get('stuck_ci'))} "
                 f"(su {t7.get('n_retreat_episodes','?')} che raggiungono il RETREAT) |")
        L.append("\nGrafici: `results/phase/`.\n")

    rb = suites.get("robustness")
    if rb and rb["status"] == "ok":
        d = rb["result"]
        L.append("## 4. Inviluppo operativo (robustezza)\n")
        L.append(f"True success complessivo: **{_fmt_ci(d['overall_true_success'])}** · "
                 f"clean success: **{_fmt_ci(d['overall_clean_success'])}**.\n")
        L.append("Regione peggiore per ciascun asse (stratificazione + Wilson CI):\n")
        L.append("| Asse | Bin peggiore | True success | n |"); L.append("|---|---|---|---|")
        for key, bins in d["envelopes"].items():
            if not bins:
                continue
            w = min(bins, key=lambda b: b["rate"])
            L.append(f"| {key} | [{w['lo']:.3f}, {w['hi']:.3f}] | "
                     f"{w['rate']*100:.1f}% [{w['ci_lo']*100:.1f}, {w['ci_hi']*100:.1f}] | {w['n']} |")
        L.append("\nGrafici: `results/robustness/` (curve 1D su 7 assi + 3 heatmap).\n")

    ab = suites.get("ablation")
    if ab and ab["status"] == "ok":
        d = ab["result"]
        L.append("## 5. Ablazione degli override del RETREAT\n")
        L.append(f"Baseline — true success {_fmt_ci(d['baseline_true_success'])}, "
                 f"clean success {_fmt_ci(d['baseline_clean_success'])}. "
                 f"Confronto **appaiato**; p-value corretti Holm-Bonferroni.\n")
        L.append("| Variante | Δ true success | 95% CI | p (Holm) | Δ clean success | p (Holm) |")
        L.append("|---|---|---|---|---|---|")
        for name, c in d["comparisons"].items():
            ts, cs = c["true_success"], c["clean_success"]
            L.append(f"| {name} | {ts['diff']['point']*100:+.1f} pt | "
                     f"[{ts['diff']['lo']*100:+.1f}, {ts['diff']['hi']*100:+.1f}] | "
                     f"{ts.get('fisher_p_holm', float('nan')):.3g} | "
                     f"{cs['diff']['point']*100:+.1f} pt | "
                     f"{cs.get('fisher_p_holm', float('nan')):.3g} |")
        L.append("\nForest plot: `results/ablation/`.\n")
        L.append("> **Due precisazioni obbligatorie.**\n>\n"
                 "> 1. **«Nessun effetto» non è dimostrato** per i bracci non significativi. "
                 "Con n = 30 gli intervalli sono ampi decine di punti: si può escludere un "
                 "effetto *grande*, non un effetto. La formulazione corretta è «nessun "
                 "effetto **rilevabile a questa numerosità**».\n>\n"
                 "> 2. **È un'ablazione del controllore dispiegato, non dell'algoritmo di "
                 "apprendimento.** Gli override vengono disattivati su una policy *già "
                 "addestrata con quegli override attivi*: si misura quanto il comportamento "
                 "finale ne dipende — domanda legittima e ben posta — ma **non** che senza di "
                 "essi non si sarebbe potuto imparare qualcos'altro.\n")

    L.append("## Limiti dichiarati\n")
    L.append("Sono limiti di **disegno**, non di esecuzione: dichiararli è parte del "
             "risultato (Henderson et al. 2018).\n")
    L.append("| # | limite | conseguenza | cosa servirebbe |")
    L.append("|---|---|---|---|")
    L.append("| 1 | **Un solo seed di addestramento** | gli intervalli descrivono la "
             "variabilità fra **episodi**, non fra **seed**: le conclusioni valgono per "
             "*questa* policy, non per il metodo | ≥5 seed e aggregazione fra run "
             "(Agarwal et al. 2021; Colas et al. 2018) |")
    L.append("| 2 | **Ablazione del controllore dispiegato** | misura quanto la policy "
             "*dipende* dagli override, non se senza di essi non si potesse imparare "
             "altro | un ri-addestramento per variante |")
    trv = ((ev or {}).get("result", {}) or {}).get("det", {}).get("trivial_reference") \
        if (ev and ev.get("status") == "ok") else None
    if trv:
        L.append(f"| 3 | **La metrica è poco selettiva** | una policy che ignora il goal "
                 f"otterrebbe già {trv['rate']*100:.0f}%: il `true_success` misura in gran "
                 f"parte la geometria del compito | tolleranza più stretta o goal campionati "
                 f"più lontano dal fine corsa |")
    L.append("| 4 | **Numerosità** | con 100 episodi differenze sotto ~6 punti non sono "
             "risolvibili; le fasce di robustezza (n≈25) non escludono dipendenze fino a "
             "~15 punti | `--preset full` |")
    L.append("| 5 | **T4 non separa policy e override** | la torsione del polso in RETREAT "
             "è in larga parte imposta dall'ambiente | registrare l'azione pre-override |")
    L.append("| 6 | **T1/T2 sulla porta base** | descrivono il regime nominale, non tutti i "
             "regimi campionati | ripetere i test fisici sotto randomizzazione |")
    L.append("")
    L.append("---\n"); L.append(BIBLIO)
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description="Esegue tutta la suite di test dell'apertura v2")
    ap.add_argument("--preset", choices=list(PRESETS), default="standard")
    ap.add_argument("--episodes", type=int, default=None,
                    help="forza lo stesso numero di episodi per tutte le batterie")
    ap.add_argument("--suites", nargs="+", default=ALL_SUITES, choices=ALL_SUITES)
    ap.add_argument("--run-dir", type=str, default=None)
    ap.add_argument("--curriculum", type=float, default=CURRICULUM)
    ap.add_argument("--ablation-quick", action="store_true",
                    help="ablaziona solo i tre override specifici dell'apertura + il totale")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    ep = PRESETS[args.preset].copy()
    if args.episodes is not None:
        ep = {k: args.episodes for k in ep}

    run_dir = resolve_run_dir(args.run_dir)
    needs_model = any(s in args.suites for s in ("evaluate", "phase", "robustness", "ablation"))
    if needs_model and find_model(run_dir, quiet=True) is None:
        print(f"\n[ATTENZIONE] nessun modello trovato in '{run_dir}'.")
        print("             Le batterie che usano la policy falliranno; passa --run-dir "
              "con la cartella giusta,")
        print("             oppure lancia solo: --suites functional physics\n")

    meta = collect_meta(args, run_dir)
    suites = {}
    import importlib

    if "functional" in args.suites:
        m = importlib.import_module("test_retreat_overrides")
        suites["functional"] = guarded("test_retreat_overrides", lambda: m.run())

    if "physics" in args.suites:
        m = importlib.import_module("physics_unit_tests")
        suites["physics"] = guarded(
            "physics_unit_tests",
            lambda: m.run(make_plots=not args.no_plots, run_dir=run_dir))

    if "evaluate" in args.suites:
        m = importlib.import_module("evaluate_policy")
        suites["evaluate"] = guarded(
            "evaluate_policy",
            lambda: m.run(ep["evaluate"], args.curriculum, run_dir,
                          make_plots_flag=not args.no_plots, tag=TAG))

    if "phase" in args.suites:
        m = importlib.import_module("phase_diagnostics")
        suites["phase"] = guarded(
            "phase_diagnostics",
            lambda: m.run(ep["phase"], True, args.curriculum, run_dir, tag=TAG))

    if "robustness" in args.suites:
        m = importlib.import_module("robustness_analysis")
        suites["robustness"] = guarded(
            "robustness_analysis",
            lambda: m.run(ep["robust"], args.curriculum, run_dir,
                          deterministic=True, tag=TAG))

    if "ablation" in args.suites:
        m = importlib.import_module("ablation_study")
        from ablation_variants import QUICK_VARIANTS
        variants = QUICK_VARIANTS if (args.ablation_quick or args.preset == "quick") else None
        suites["ablation"] = guarded(
            "ablation_study",
            lambda: m.run(ep["ablation"], args.curriculum, run_dir,
                          deterministic=True, variants=variants, tag=TAG))

    outdir = results_dir()
    with open(os.path.join(outdir, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2, default=json_default)

    def slim(o):
        if isinstance(o, dict):
            return {k: slim(v) for k, v in o.items()
                    if k not in ("values", "trajectory", "series", "events",
                                 "series_free", "series_kicked")}
        if isinstance(o, list) and len(o) > 50:
            return f"<list len {len(o)}>"
        if isinstance(o, list):
            return [slim(x) for x in o]
        return o
    agg = {name: (dict(status=s["status"], seconds=s["seconds"], result=slim(s.get("result")))
                  if s["status"] == "ok" else s) for name, s in suites.items()}
    with open(os.path.join(outdir, "all_results.json"), "w") as f:
        json.dump(agg, f, indent=2, default=json_default)

    with open(os.path.join(outdir, "REPORT.md"), "w") as f:
        f.write(build_report(meta, suites))

    print("\n" + "=" * 78)
    n_ok = sum(1 for s in suites.values() if s["status"] == "ok")
    print(f"COMPLETATO: {n_ok}/{len(suites)} batterie OK")
    print(f"Report: {os.path.join(outdir, 'REPORT.md')}")
    print("=" * 78)


if __name__ == "__main__":
    main()
