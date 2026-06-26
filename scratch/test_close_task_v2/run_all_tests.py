#!/usr/bin/env python3
# scratch/test_close_task_v2/run_all_tests.py

"""
run_all_tests — Orchestratore della suite di test scientifica (task v2).

Esegue, in un solo comando, tutte le batterie e produce un report aggregato con
metriche, intervalli di confidenza, grafici e bibliografia.

Batterie:
  0. functional   — test white-box §1.21 (rampa di ritiro), no modello/robosuite
  1. physics      — proprietà fisiche deterministiche (T1/T2) + range randomization v2 (T7)
  2. evaluate     — valutazione rigorosa det+sto (Wilson CI, IQM, CVaR), per modello
  3. phase        — diagnostica HOLD/RETREAT (T3–T6)
  4. robustness   — inviluppo operativo su 6 assi di randomization v2
  5. ablation     — toggle §1.17/§1.18/§1.21 sullo stesso modello (confronto appaiato)

Modelli (forniti): curr0 = posa fissa, curr1 = posa variabile.

Esempi:
  python scratch/test_close_task_v2/run_all_tests.py --preset standard
  python scratch/test_close_task_v2/run_all_tests.py --preset full
  python scratch/test_close_task_v2/run_all_tests.py --suites evaluate ablation --models curr1
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
from _common import results_dir, REPO_ROOT, MODEL_SPECS, json_default  # noqa: E402

PRESETS = {
    "quick":    dict(evaluate=30,  phase=15, robust=60,  ablation=15),
    "standard": dict(evaluate=100, phase=30, robust=150, ablation=30),
    "full":     dict(evaluate=200, phase=50, robust=300, ablation=50),
}
ALL_SUITES = ["functional", "physics", "evaluate", "phase", "robustness", "ablation"]


def collect_meta(args):
    def ver(mod):
        try:
            return __import__(mod).__version__
        except Exception:
            return "n/a"

    def git_hash():
        try:
            return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                           cwd=REPO_ROOT, stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            return "n/a"
    return dict(timestamp=datetime.datetime.now().isoformat(timespec="seconds"),
                repo_root=REPO_ROOT, git_commit=git_hash(),
                python=platform.python_version(), platform=platform.platform(),
                versions=dict(numpy=ver("numpy"), scipy=ver("scipy"),
                              stable_baselines3=ver("stable_baselines3"),
                              robosuite=ver("robosuite"), torch=ver("torch"),
                              gymnasium=ver("gymnasium")),
                args=vars(args))


def guarded(name, fn):
    print("\n" + "#" * 76); print(f"# SUITE: {name}"); print("#" * 76)
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
"""


def _fmt_ci(d):
    if not isinstance(d, dict) or "point" not in d:
        return "n/a"
    return f"{d['point']*100:.1f}% [{d['lo']*100:.1f}, {d['hi']*100:.1f}]"


def build_report(meta, suites):
    L = []
    L.append("# Report suite di test — task di chiusura generalizzato v2\n")
    L.append("> Generato da `run_all_tests.py`. I numeri sono **stime a intervallo** (rif. 1–8). "
             "Le ablazioni toggolano i contributi v2 §1.17/§1.18/§1.21 sulla stessa policy.\n")
    L.append(f"- **Data:** {meta['timestamp']}  ·  **git:** `{meta['git_commit']}`")
    L.append(f"- **Repo root:** `{meta['repo_root']}`")
    L.append(f"- **Python:** {meta['python']}  ·  **piattaforma:** {meta['platform']}")
    v = meta["versions"]
    L.append(f"- **Versioni:** numpy {v['numpy']} · scipy {v['scipy']} · SB3 {v['stable_baselines3']} "
             f"· robosuite {v['robosuite']} · torch {v['torch']}")
    L.append(f"- **Argomenti:** `{json.dumps(meta['args'])}`\n")

    L.append("## Esito esecuzione\n")
    L.append("| Suite | Stato | Tempo (s) |"); L.append("|---|---|---|")
    for name, s in suites.items():
        L.append(f"| {name} | {'✅ ok' if s['status']=='ok' else '❌ fail'} | {s['seconds']:.1f} |")
    L.append("")

    fn = suites.get("functional")
    if fn and fn["status"] == "ok":
        r = fn["result"]
        L.append("## 0. Functional white-box — rampa di ritiro §1.21\n")
        L.append(f"Esito: **{r.get('passed','?')}/{r.get('total','?')} PASS** (no modello, deterministico).\n")

    ph = suites.get("physics")
    if ph and ph["status"] == "ok":
        r = ph["result"]; summ = r.get("summary", {})
        L.append("## 1. Physics unit tests\n")
        L.append(f"Esito: **{summ.get('passed','?')}/{summ.get('total','?')} PASS**.\n")
        L.append("| Test | Esito |"); L.append("|---|---|")
        for n, p in summ.get("checks", {}).items():
            L.append(f"| {n} | {'PASS' if p else 'FAIL'} |")
        L.append("\nGrafici: `results/physics/`.\n")

    ev = suites.get("evaluate")
    if ev and ev["status"] == "ok":
        L.append("## 2. Valutazione rigorosa (det + sto)\n")
        L.append("| Modello | Modo | Success (95% CI) | True success (95% CI) | Len IQM | Len CVaR 10% |")
        L.append("|---|---|---|---|---|---|")
        for tag, modes in ev["result"].items():
            for mode in ("det", "sto"):
                m = modes.get(mode)
                if not m:
                    continue
                L.append(f"| {tag} | {mode} | {_fmt_ci(m['success_rate'])} | "
                         f"{_fmt_ci(m['true_success_rate'])} | {m['length_iqm']['point']:.1f} "
                         f"| {m['length_cvar_worst10']:.1f} |")
        L.append("\nGrafici: `results/evaluate/`.\n")

    pd = suites.get("phase")
    if pd and pd["status"] == "ok":
        L.append("## 3. Diagnostica fasi HOLD/RETREAT (T3–T6)\n")
        L.append("| Modello | HOLD ‖a‖ IQM | RETREAT polso IQM | latch@transiz (% > 0.15) | bounce |")
        L.append("|---|---|---|---|---|")
        for tag, dd in pd["result"].items():
            t3 = dd.get("T3_hold_action_norm") or {}; t4 = dd.get("T4_retreat_wrist_rot") or {}
            t5 = dd.get("T5_latch_at_transition") or {}; t6 = dd.get("T6_bounce_events") or {}
            i3 = t3.get("iqm", {}).get("point", float("nan")) if t3 else float("nan")
            i4 = t4.get("iqm", {}).get("point", float("nan")) if t4 else float("nan")
            f5 = (t5.get("frac_above_thresh", float("nan")) * 100) if t5 else float("nan")
            L.append(f"| {tag} | {i3:.3f} | {i4:.3f} | {f5:.1f}% | {t6.get('n','?')} |")
        L.append("\nGrafici: `results/phase/`.\n")

    rb = suites.get("robustness")
    if rb and rb["status"] == "ok":
        L.append("## 4. Inviluppo operativo (robustezza)\n")
        L.append("True success per regione dei parametri (stratificazione + Wilson CI), 6 assi v2.\n")
        for tag, dd in rb["result"].items():
            L.append(f"**{tag}** — true success complessivo: {_fmt_ci(dd['overall_true_success'])}.")
            for key in dd["envelopes"]:
                bins = dd["envelopes"][key] or []
                if bins:
                    worst = min(bins, key=lambda b: b["rate"])
                    L.append(f"  - {key}: bin peggiore [{worst['lo']:.3f},{worst['hi']:.3f}] "
                             f"→ {worst['rate']*100:.1f}% (n={worst['n']}).")
            L.append("")
        L.append("Grafici: `results/robustness/` (curve 1D + heatmap).\n")

    ab = suites.get("ablation")
    if ab and ab["status"] == "ok":
        L.append("## 5. Ablazione §1.17/§1.18/§1.21 (baseline vs toggle)\n")
        L.append("Confronto **appaiato**, Fisher + Newcombe, p-value Holm-Bonferroni.\n")
        for tag, dd in ab["result"].items():
            L.append(f"**{tag}** — baseline true success {_fmt_ci(dd['baseline_true_success'])}.\n")
            L.append("| Variante | Δ true success | 95% CI | Fisher p | p (Holm) |")
            L.append("|---|---|---|---|---|")
            for name, c in dd["comparisons"].items():
                ts = c["true_success"]; diff = ts["diff"]
                L.append(f"| {name} | {diff['point']*100:+.1f} pt | "
                         f"[{diff['lo']*100:+.1f}, {diff['hi']*100:+.1f}] | "
                         f"{ts['fisher_p']:.3g} | {ts.get('fisher_p_holm', float('nan')):.3g} |")
            L.append("")
        L.append("Forest plot: `results/ablation/`.\n")

    L.append("---\n"); L.append(BIBLIO)
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description="Esegue tutta la suite di test v2")
    ap.add_argument("--preset", choices=list(PRESETS), default="standard")
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument("--models", nargs="+", default=["curr0", "curr1"],
                    choices=["curr0", "curr1"], help="quali modelli valutare")
    ap.add_argument("--suites", nargs="+", default=ALL_SUITES, choices=ALL_SUITES)
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    ep = PRESETS[args.preset].copy()
    if args.episodes is not None:
        ep = {k: args.episodes for k in ep}

    # filtra i modelli richiesti
    specs = [s for s in MODEL_SPECS if (("curr0" in args.models and s[2] < 0.5)
                                        or ("curr1" in args.models and s[2] >= 0.5))]

    meta = collect_meta(args)
    suites = {}
    import importlib

    if "functional" in args.suites:
        m = importlib.import_module("test_retreat_ramp")
        suites["functional"] = guarded("test_retreat_ramp", lambda: m.run())

    if "physics" in args.suites:
        m = importlib.import_module("physics_unit_tests")
        suites["physics"] = guarded("physics_unit_tests",
                                    lambda: m.run(make_plots=not args.no_plots))

    if "evaluate" in args.suites:
        m = importlib.import_module("evaluate_policy")
        def _eval():
            out = {}
            for tag, run_dir, curr in specs:
                out[tag] = m.run(ep["evaluate"], curr, run_dir,
                                 make_plots_flag=not args.no_plots, tag=tag)
            return out
        suites["evaluate"] = guarded("evaluate_policy", _eval)

    if "phase" in args.suites:
        m = importlib.import_module("phase_diagnostics")
        def _phase():
            out = {}
            for tag, run_dir, curr in specs:
                out[tag] = m.run(ep["phase"], True, curr, run_dir, tag=tag)
            return out
        suites["phase"] = guarded("phase_diagnostics", _phase)

    if "robustness" in args.suites:
        m = importlib.import_module("robustness_analysis")
        def _rob():
            out = {}
            for tag, run_dir, curr in specs:
                out[tag] = m.run(ep["robust"], curr, run_dir, deterministic=True, tag=tag)
            return out
        suites["robustness"] = guarded("robustness_analysis", _rob)

    if "ablation" in args.suites:
        m = importlib.import_module("ablation_study")
        def _abl():
            out = {}
            for tag, run_dir, curr in specs:
                out[tag] = m.run(ep["ablation"], curr, run_dir, deterministic=True, tag=tag)
            return out
        suites["ablation"] = guarded("ablation_study", _abl)

    outdir = results_dir()
    with open(os.path.join(outdir, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2, default=json_default)

    def slim(o):
        if isinstance(o, dict):
            return {k: slim(v) for k, v in o.items()
                    if k not in ("values", "trajectory", "series", "events")}
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

    print("\n" + "=" * 76)
    n_ok = sum(1 for s in suites.values() if s["status"] == "ok")
    print(f"COMPLETATO: {n_ok}/{len(suites)} batterie OK")
    print(f"Report: {os.path.join(outdir, 'REPORT.md')}")
    print("=" * 76)


if __name__ == "__main__":
    main()