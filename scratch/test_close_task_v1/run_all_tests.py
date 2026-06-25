#!/usr/bin/env python3
# scratch/test_close_task_v1/run_all_tests.py
"""
run_all_tests — Orchestratore della suite di test scientifica (task di chiusura v1).

Lancia, in un solo comando, tutte le batterie di test e produce un report aggregato
con metriche, intervalli di confidenza, grafici e bibliografia.

Batterie eseguite:
  1. physics_unit_tests  — proprietà fisiche deterministiche dell'env (no modello)
  2. evaluate_policy      — valutazione rigorosa det+sto (Wilson CI, IQM, CVaR)
  3. phase_diagnostics    — diagnostica HOLD/RETREAT (T3–T6) con statistica
  4. robustness_analysis  — inviluppo operativo (success vs domain randomization)
  5. ablation_study       — confronto controllato baseline vs interventi

Esempi:
  python scratch/test_close_task_v1/run_all_tests.py --preset standard
  python scratch/test_close_task_v1/run_all_tests.py --preset full --curricula 0 1
  python scratch/test_close_task_v1/run_all_tests.py --suites evaluate ablation --episodes 80

Tutti i path sono risolti rispetto alla radice del progetto, indipendentemente dalla
cartella di lancio (riproducibilità — Henderson 2018; Patterson 2024).
Output in scratch/test_close_task_v1/results/ (JSON + PNG + REPORT.md).
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

# import robusto dei moduli locali (funziona da qualunque CWD)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import results_dir, REPO_ROOT, DEFAULT_RUN_DIR  # noqa: E402


PRESETS = {
    "quick":    dict(evaluate=30,  phase=15, robust=60,  ablation=15),
    "standard": dict(evaluate=100, phase=30, robust=150, ablation=30),
    "full":     dict(evaluate=200, phase=50, robust=300, ablation=50),
}

ALL_SUITES = ["physics", "evaluate", "phase", "robustness", "ablation"]


# ─────────────────────────────────────────────────────────────────────────────
# Metadati di esecuzione (riproducibilità)
# ─────────────────────────────────────────────────────────────────────────────
def collect_meta(args) -> dict:
    def ver(mod):
        try:
            return __import__(mod).__version__
        except Exception:
            return "n/a"

    def git_hash():
        try:
            return subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
                stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            return "n/a"

    return dict(
        timestamp=datetime.datetime.now().isoformat(timespec="seconds"),
        repo_root=REPO_ROOT,
        git_commit=git_hash(),
        python=platform.python_version(),
        platform=platform.platform(),
        versions=dict(numpy=ver("numpy"), scipy=ver("scipy"),
                      stable_baselines3=ver("stable_baselines3"),
                      robosuite=ver("robosuite"), torch=ver("torch"),
                      gymnasium=ver("gymnasium")),
        args=vars(args),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Esecuzione protetta di una batteria
# ─────────────────────────────────────────────────────────────────────────────
def guarded(name, fn):
    print("\n" + "#" * 76)
    print(f"# SUITE: {name}")
    print("#" * 76)
    t0 = time.time()
    try:
        res = fn()
        dt = time.time() - t0
        print(f"[OK] {name} in {dt:.1f}s")
        return dict(status="ok", seconds=dt, result=res)
    except Exception as e:
        dt = time.time() - t0
        print(f"[FAIL] {name}: {e}")
        traceback.print_exc()
        return dict(status="fail", seconds=dt, error=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Generazione del report markdown
# ─────────────────────────────────────────────────────────────────────────────
BIBLIO = """\
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
"""


def _fmt_ci(d):
    if not isinstance(d, dict) or "point" not in d:
        return "n/a"
    return f"{d['point']*100:.1f}% [{d['lo']*100:.1f}, {d['hi']*100:.1f}]"


def build_report(meta, suites) -> str:
    L = []
    L.append("# Report suite di test — task di chiusura v1\n")
    L.append("> Generato automaticamente da `run_all_tests.py`. I numeri sono **stime a "
             "intervallo** (non point estimate): è il punto della metodologia (rif. 1–6).\n")
    L.append(f"- **Data:** {meta['timestamp']}")
    L.append(f"- **Repo root:** `{meta['repo_root']}`  ·  **git:** `{meta['git_commit']}`")
    L.append(f"- **Python:** {meta['python']}  ·  **piattaforma:** {meta['platform']}")
    v = meta["versions"]
    L.append(f"- **Versioni:** numpy {v['numpy']} · scipy {v['scipy']} · "
             f"SB3 {v['stable_baselines3']} · robosuite {v['robosuite']} · torch {v['torch']}")
    L.append(f"- **Argomenti:** `{json.dumps(meta['args'])}`\n")

    # Stato batterie
    L.append("## Esito esecuzione\n")
    L.append("| Suite | Stato | Tempo (s) |")
    L.append("|-------|-------|-----------|")
    for name, s in suites.items():
        L.append(f"| {name} | {'✅ ok' if s['status']=='ok' else '❌ fail'} | {s['seconds']:.1f} |")
    L.append("")

    # 1. Physics
    ph = suites.get("physics")
    if ph and ph["status"] == "ok":
        r = ph["result"]; summ = r.get("summary", {})
        L.append("## 1. Physics unit tests (deterministici)\n")
        L.append(f"Esito: **{summ.get('passed','?')}/{summ.get('total','?')} PASS**.\n")
        L.append("| Test | Esito |")
        L.append("|------|-------|")
        for n, p in summ.get("checks", {}).items():
            L.append(f"| {n} | {'PASS' if p else 'FAIL'} |")
        ls = r.get("latch_spring", {})
        hb = r.get("hinge_bounce", {})
        if ls:
            L.append(f"\n- Latch spring: ritorno a neutro in **{ls.get('return_steps')}** step "
                     f"(stiffness {ls.get('stiffness'):.3f}, damping {ls.get('damping'):.3f}).")
        if hb:
            L.append(f"- Hinge bounce: velocità massima **{hb.get('max_bounce_vel'):.4f} rad/s** "
                     f"(damping {hb.get('damping'):.3f}).")
        L.append("\nGrafici: `results/physics/`.\n")

    # 2. Evaluate (per curriculum)
    ev = suites.get("evaluate")
    if ev and ev["status"] == "ok":
        L.append("## 2. Valutazione rigorosa (det + sto)\n")
        L.append("Success rate con **intervallo di Wilson**; lunghezza/angolo con **IQM + "
                 "bootstrap CI**; coda peggiore via **CVaR**.\n")
        L.append("| Curr | Modo | Success (95% CI) | True success (95% CI) | Len IQM | Len CVaR 10% |")
        L.append("|------|------|------------------|------------------------|---------|--------------|")
        for curr, modes in ev["result"].items():
            for mode in ("det", "sto"):
                m = modes.get(mode)
                if not m:
                    continue
                L.append(f"| {curr} | {mode} | {_fmt_ci(m['success_rate'])} | "
                         f"{_fmt_ci(m['true_success_rate'])} | "
                         f"{m['length_iqm']['point']:.1f} | {m['length_cvar_worst10']:.1f} |")
        L.append("\nGrafici: `results/evaluate/`.\n")

    # 3. Phase
    pd = suites.get("phase")
    if pd and pd["status"] == "ok":
        L.append("## 3. Diagnostica fasi HOLD/RETREAT (T3–T6)\n")
        L.append("| Curr | HOLD ‖a‖ IQM | RETREAT polso IQM | latch@transiz (% > 0.15) | bounce |")
        L.append("|------|--------------|-------------------|---------------------------|--------|")
        for curr, dd in pd["result"].items():
            t3 = dd.get("T3_hold_action_norm") or {}
            t4 = dd.get("T4_retreat_wrist_rot") or {}
            t5 = dd.get("T5_latch_at_transition") or {}
            t6 = dd.get("T6_bounce_events") or {}
            i3 = t3.get("iqm", {}).get("point", float("nan")) if t3 else float("nan")
            i4 = t4.get("iqm", {}).get("point", float("nan")) if t4 else float("nan")
            f5 = (t5.get("frac_above_thresh", float("nan")) * 100) if t5 else float("nan")
            L.append(f"| {curr} | {i3:.3f} | {i4:.3f} | {f5:.1f}% | {t6.get('n','?')} |")
        L.append("\nGrafici: `results/phase/`.\n")

    # 4. Robustness
    rb = suites.get("robustness")
    if rb and rb["status"] == "ok":
        L.append("## 4. Inviluppo operativo (robustezza)\n")
        L.append("True success per regione dei parametri di domain randomization "
                 "(stratificazione + Wilson CI). Risponde a *08_risultati_v2 §2/§4*: "
                 "quantifica DOVE la policy generalizza.\n")
        for curr, dd in rb["result"].items():
            L.append(f"**Curriculum {curr}** — true success complessivo: "
                     f"{_fmt_ci(dd['overall_true_success'])}.")
            for key, label in [("handle_friction", "frizione"),
                               ("handle_radius", "raggio"), ("door_x", "distanza porta")]:
                bins = dd["envelopes"].get(key) or []
                if bins:
                    worst = min(bins, key=lambda b: b["rate"])
                    L.append(f"  - {label}: bin peggiore [{worst['lo']:.3f},{worst['hi']:.3f}] "
                             f"→ {worst['rate']*100:.1f}% (n={worst['n']}).")
            L.append("")
        L.append("Grafici: `results/robustness/` (curve 1D + heatmap frizione×raggio).\n")

    # 5. Ablation
    ab = suites.get("ablation")
    if ab and ab["status"] == "ok":
        L.append("## 5. Studio di ablazione (baseline vs interventi)\n")
        L.append("Confronto **appaiato** (stessi seed), test di **Fisher** + CI di **Newcombe**, "
                 "p-value corretti con **Holm-Bonferroni**.\n")
        for curr, dd in ab["result"].items():
            L.append(f"**Curriculum {curr}** — baseline true success "
                     f"{_fmt_ci(dd['baseline_true_success'])}.\n")
            L.append("| Variante | Δ true success | 95% CI | Fisher p | p (Holm) |")
            L.append("|----------|----------------|--------|----------|----------|")
            for name, c in dd["comparisons"].items():
                ts = c["true_success"]; diff = ts["diff"]
                L.append(f"| {name} | {diff['point']*100:+.1f} pt | "
                         f"[{diff['lo']*100:+.1f}, {diff['hi']*100:+.1f}] | "
                         f"{ts['fisher_p']:.3g} | {ts.get('fisher_p_holm', float('nan')):.3g} |")
            L.append("")
        L.append("Forest plot: `results/ablation/`.\n")

    L.append("---\n")
    L.append(BIBLIO)
    return "\n".join(L)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Esegue tutta la suite di test v1")
    ap.add_argument("--preset", choices=list(PRESETS), default="standard")
    ap.add_argument("--episodes", type=int, default=None,
                    help="override del numero di episodi per TUTTE le batterie basate su modello")
    ap.add_argument("--curricula", type=float, nargs="+", default=[1.0],
                    help="livelli di curriculum da valutare (es. 0 1)")
    ap.add_argument("--run-dir", type=str, default=DEFAULT_RUN_DIR)
    ap.add_argument("--suites", nargs="+", default=ALL_SUITES,
                    choices=ALL_SUITES, help="sottoinsieme di batterie")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    ep = PRESETS[args.preset].copy()
    if args.episodes is not None:
        ep = {k: args.episodes for k in ep}

    meta = collect_meta(args)
    suites = {}

    # import differiti (così physics può girare anche se manca il modello, ecc.)
    import importlib

    if "physics" in args.suites:
        m = importlib.import_module("physics_unit_tests")
        suites["physics"] = guarded("physics_unit_tests",
                                    lambda: m.run(make_plots=not args.no_plots,
                                                  run_dir=args.run_dir))

    if "evaluate" in args.suites:
        m = importlib.import_module("evaluate_policy")
        def _eval():
            out = {}
            for c in args.curricula:
                out[c] = m.run(ep["evaluate"], c, args.run_dir,
                               make_plots_flag=not args.no_plots)
            return out
        suites["evaluate"] = guarded("evaluate_policy", _eval)

    if "phase" in args.suites:
        m = importlib.import_module("phase_diagnostics")
        def _phase():
            out = {}
            for c in args.curricula:
                out[c] = m.run(ep["phase"], True, c, args.run_dir)
            return out
        suites["phase"] = guarded("phase_diagnostics", _phase)

    if "robustness" in args.suites:
        m = importlib.import_module("robustness_analysis")
        def _rob():
            out = {}
            for c in args.curricula:
                out[c] = m.run(ep["robust"], c, args.run_dir, deterministic=True)
            return out
        suites["robustness"] = guarded("robustness_analysis", _rob)

    if "ablation" in args.suites:
        m = importlib.import_module("ablation_study")
        def _abl():
            out = {}
            for c in args.curricula:
                out[c] = m.run(ep["ablation"], c, args.run_dir, deterministic=True)
            return out
        suites["ablation"] = guarded("ablation_study", _abl)

    # Salvataggio aggregato + report
    outdir = results_dir()
    with open(os.path.join(outdir, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # results JSON "snello" (senza i grossi array per-episodio)
    def slim(o):
        if isinstance(o, dict):
            return {k: slim(v) for k, v in o.items()
                    if k not in ("values", "trajectory", "series", "events")}
        if isinstance(o, list) and len(o) > 50:
            return f"<list len {len(o)}>"
        if isinstance(o, list):
            return [slim(x) for x in o]
        return o
    agg = {name: (dict(status=s["status"], seconds=s["seconds"],
                       result=slim(s.get("result"))) if s["status"] == "ok"
                  else s) for name, s in suites.items()}
    with open(os.path.join(outdir, "all_results.json"), "w") as f:
        json.dump(agg, f, indent=2, default=str)

    report = build_report(meta, suites)
    with open(os.path.join(outdir, "REPORT.md"), "w") as f:
        f.write(report)

    print("\n" + "=" * 76)
    n_ok = sum(1 for s in suites.values() if s["status"] == "ok")
    print(f"COMPLETATO: {n_ok}/{len(suites)} batterie OK")
    print(f"Report: {os.path.join(outdir, 'REPORT.md')}")
    print("=" * 76)


if __name__ == "__main__":
    main()
