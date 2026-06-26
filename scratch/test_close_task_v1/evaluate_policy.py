#!/usr/bin/env python3
# scratch/test_close_task_v1/evaluate_policy.py

"""
evaluate_policy — Valutazione statisticamente rigorosa della policy (task di chiusura v1).

Sostituisce/estende `eval_stats_close.py`.

  1. STIME A INTERVALLO, non point estimate. Il success rate è riportato con
     intervallo di Wilson; lunghezza ed angolo minimo con IQM + bootstrap CI.
     (Agarwal et al. 2021; Henderson et al. 2018; Colas et al. 2018.)
  2. METRICA DI RISCHIO. CVaR sulla coda peggiore degli episodi — il "caso brutto"
     che conta in robotica. (Chan et al. 2020.)
  3. SUCCESSO "VERO" separato dal permissivo (porta chiusa + latch neutro a fine
     episodio), come richiesto in 08_risultati_v2 §4.2.
  4. RIPRODUCIBILITÀ. Episodi seedati in modo deterministico (base_seed + i): rilanciare
     dà lo stesso risultato; lo stesso set di seed è riusato dalle ablazioni (confronto
     appaiato — Patterson et al. 2024).
  5. NUMEROSITÀ MOTIVATA. Default 200 episodi: con success ~0.95 dà un CI di Wilson di
     semi-ampiezza ~±0.03 (power analysis — Colas et al. 2018). Era 50.

Output (in results/evaluate/):
  • metrics_<mode>_c<curr>.json  — tutte le metriche + intervalli
  • episodes_<mode>_c<curr>.json — record per-episodio (per analisi a valle)
  • plot_success.png, plot_phase_dist.png, plot_phase_time.png,
    plot_metric_distributions.png, plot_failure_breakdown.png, plot_bootstrap.png
"""

from __future__ import annotations

import os
import json
import argparse
from collections import Counter

import numpy as np

from _common import (make_cfg, make_vec_env, load_model, rollout_episode, results_dir, setup_matplotlib, PHASE_NAMES, FAILURE_TYPES, json_default)
import stats_utils as S


def run_eval(n_episodes: int, deterministic: bool, curriculum: float,
             run_dir: str, base_seed: int = 10_000):
    cfg           = make_cfg(run_dir=run_dir)
    venv, raw_env = make_vec_env(cfg, curriculum_level=curriculum)
    model         = load_model(cfg, venv)

    records = []
    for i in range(n_episodes):
        rec = rollout_episode(venv, model, raw_env,
                              deterministic=deterministic, seed=base_seed + i)
        records.append(rec)
        tag = "SUCCESS" if rec.success else rec.failure_type
        print(f"  ep {i+1:>3}/{n_episodes} | succ={rec.success} true={rec.true_success} "
              f"| len={rec.length:>3} | minθ={rec.min_door_angle:.4f} | {tag}")
    venv.close()
    return records


def summarize(records, mode_name: str, curriculum: float) -> dict:
    n         = len(records)
    succ      = sum(r.success for r in records)
    tsucc     = sum(r.true_success for r in records)
    lengths   = [r.length for r in records]
    min_doors = [r.min_door_angle for r in records]

    sr       = S.wilson_ci(succ, n)
    tsr      = S.wilson_ci(tsucc, n)
    len_iqm  = S.bootstrap_ci(lengths, "iqm")
    door_iqm = S.bootstrap_ci(min_doors, "iqm")

    fail_counts       = Counter(r.failure_type for r in records)
    failure_breakdown = {}
    for ft in FAILURE_TYPES:
        c = fail_counts.get(ft, 0)
        failure_breakdown[ft] = dict(count=c, ci=S.wilson_ci(c, n).as_dict())

    phase_counts = Counter(r.max_phase for r in records)

    summary = dict(
        mode=mode_name, curriculum=curriculum, n_episodes=n,
        success_rate=sr.as_dict(),
        true_success_rate=tsr.as_dict(),
        length_iqm=len_iqm.as_dict(),
        length_mean=float(np.mean(lengths)),
        length_cvar_worst10=S.cvar(lengths, 0.1, lower_tail=False),  # episodi più lunghi
        min_door_iqm=door_iqm.as_dict(),
        min_door_cvar_worst10=S.cvar(min_doors, 0.1, lower_tail=False),  # angoli più alti = peggio
        max_phase_distribution={p: phase_counts.get(p, 0) for p in PHASE_NAMES},
        failure_breakdown=failure_breakdown,
    )

    print("\n" + "=" * 72)
    print(f"VALUTAZIONE — {mode_name} (curriculum={curriculum}, {n} episodi)")
    print("=" * 72)
    print(f"  Success rate (permissivo)  : {sr}")
    print(f"  True success (chiusa+latch): {tsr}")
    print(f"  Lunghezza episodio (IQM)   : {len_iqm}")
    print(f"  Lunghezza, CVaR peggior 10%: {summary['length_cvar_worst10']:.1f} step")
    print(f"  Min door angle (IQM)       : {door_iqm}")
    print(f"  Min door, CVaR peggior 10% : {summary['min_door_cvar_worst10']:.4f} rad")
    print("  " + "-" * 68)
    print("  Modo di fallimento                         count   95% CI")
    for ft in FAILURE_TYPES:
        b = failure_breakdown[ft]
        ci = b["ci"]
        print(f"    {ft:<32} {b['count']:>5}   "
              f"[{ci['lo']*100:4.1f}, {ci['hi']*100:4.1f}]%")
    print("=" * 72 + "\n")
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────
def make_plots(det_records, det_summary, sto_records, sto_summary, outdir, curriculum):
    plt = setup_matplotlib()

    def err(ci):
        return [[ci["point"] - ci["lo"]], [ci["hi"] - ci["point"]]]

    # 1) Success rate con CI (permissivo + true), det vs sto
    fig, ax = plt.subplots(figsize=(8, 6))
    groups = ["Eval det", "Train sto"]
    perm = [det_summary["success_rate"], sto_summary["success_rate"]]
    true = [det_summary["true_success_rate"], sto_summary["true_success_rate"]]
    x = np.arange(2)
    w = 0.35
    for j, (vals, lab, col) in enumerate([(perm, "permissivo", "#1f77b4"),
                                          (true, "true success", "#2ca02c")]):
        pts = [v["point"] * 100 for v in vals]
        errs = np.clip([[(v["point"] - v["lo"]) * 100 for v in vals],
                        [(v["hi"] - v["point"]) * 100 for v in vals]], 0, None)
        ax.bar(x + (j - 0.5) * w, pts, w, yerr=errs, capsize=5, label=lab, color=col)
    ax.set_xticks(x); ax.set_xticklabels(groups)
    ax.set_ylabel("Success rate (%)"); ax.set_ylim(0, 105)
    ax.set_title(f"Success rate ± Wilson 95% CI (curriculum {curriculum})")
    ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "plot_success.png"), dpi=130); plt.close(fig)

    # 2) Distribuzione fase massima raggiunta (stacked)
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ["#d62728", "#ff7f0e", "#1f77b4", "#2ca02c"]
    for name, recs in [("Eval det", det_records), ("Train sto", sto_records)]:
        bottom = 0
        cnt = Counter(r.max_phase for r in recs)
        for i, ph in enumerate(PHASE_NAMES):
            v = cnt.get(ph, 0)
            ax.bar(name, v, bottom=bottom, color=colors[i],
                   label=ph if name == "Eval det" else "")
            if v:
                ax.text(name, bottom + v / 2, str(v), ha="center", va="center",
                        color="white", fontweight="bold")
            bottom += v
    ax.set_ylabel("Episodi"); ax.set_title("Fase massima raggiunta"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "plot_phase_dist.png"), dpi=130); plt.close(fig)

    # 3) Tempo medio per fase ± CI
    fig, ax = plt.subplots(figsize=(9, 6))
    xs = np.arange(len(PHASE_NAMES)); w = 0.35
    for j, (name, recs, col) in enumerate([("Eval det", det_records, "#1f77b4"),
                                           ("Train sto", sto_records, "#2ca02c")]):
        means, errs_lo, errs_hi = [], [], []
        for ph in PHASE_NAMES:
            vals = [r.phase_times[ph] for r in recs]
            ci = S.bootstrap_ci(vals, "mean")
            means.append(ci.point); errs_lo.append(max(0.0, ci.point - ci.lo)); errs_hi.append(max(0.0, ci.hi - ci.point))
        ax.bar(xs + (j - 0.5) * w, means, w, yerr=[errs_lo, errs_hi], capsize=4,
               label=name, color=col)
    ax.set_xticks(xs); ax.set_xticklabels(PHASE_NAMES)
    ax.set_ylabel("Step medi"); ax.set_title("Tempo per fase (media ± bootstrap 95% CI)")
    ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "plot_phase_time.png"), dpi=130); plt.close(fig)

    # 4) Distribuzioni per-episodio (box): lunghezza e min door angle
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].boxplot([[r.length for r in det_records], [r.length for r in sto_records]],
                    labels=["det", "sto"], showmeans=True)
    axes[0].set_title("Lunghezza episodio"); axes[0].set_ylabel("step")
    axes[1].boxplot([[r.min_door_angle for r in det_records],
                     [r.min_door_angle for r in sto_records]],
                    labels=["det", "sto"], showmeans=True)
    axes[1].set_title("Min door angle"); axes[1].set_ylabel("rad")
    fig.suptitle("Distribuzioni per-episodio (la dispersione conta — Chan 2020)")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "plot_metric_distributions.png"), dpi=130); plt.close(fig)

    # 5) Breakdown fallimenti con CI (eval det)
    fig, ax = plt.subplots(figsize=(10, 6))
    fb = det_summary["failure_breakdown"]
    types = [t for t in FAILURE_TYPES]
    pts = [fb[t]["ci"]["point"] * 100 for t in types]
    errs = np.clip([[(fb[t]["ci"]["point"] - fb[t]["ci"]["lo"]) * 100 for t in types],
                    [(fb[t]["ci"]["hi"] - fb[t]["ci"]["point"]) * 100 for t in types]], 0, None)
    yy = np.arange(len(types))
    ax.barh(yy, pts, xerr=errs, capsize=4, color="#8c564b")
    ax.set_yticks(yy); ax.set_yticklabels(types)
    ax.set_xlabel("% episodi (± Wilson 95% CI)"); ax.set_title("Breakdown esiti — Eval det")
    ax.invert_yaxis()
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "plot_failure_breakdown.png"), dpi=130); plt.close(fig)

    # 6) Distribuzione bootstrap del success rate (eval det)
    fig, ax = plt.subplots(figsize=(8, 5))
    succ = np.array([1.0 if r.success else 0.0 for r in det_records])
    rng = np.random.default_rng(0)
    boots = rng.choice(succ, size=(10_000, len(succ)), replace=True).mean(axis=1)
    ax.hist(boots * 100, bins=40, color="#1f77b4", alpha=0.8)
    sr = det_summary["success_rate"]
    ax.axvline(sr["lo"] * 100, color="k", ls="--"); ax.axvline(sr["hi"] * 100, color="k", ls="--")
    ax.set_xlabel("Success rate bootstrap (%)"); ax.set_ylabel("frequenza")
    ax.set_title("Distribuzione bootstrap del success rate (Eval det)\n(linee = Wilson 95% CI)")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "plot_bootstrap.png"), dpi=130); plt.close(fig)


def run(episodes: int = 200, curriculum: float = 1.0, run_dir: str = "runs/close_gen",
        seed: int = 10_000, make_plots_flag: bool = True) -> dict:
    """Esegue la valutazione det+sto, salva metriche/episodi/plot, ritorna i summary."""
    outdir = results_dir("evaluate")
    print("Valutazione EVAL (deterministica)…")
    det = run_eval(episodes, True, curriculum, run_dir, seed)
    det_sum = summarize(det, "Eval (deterministica)", curriculum)
    print("Valutazione TRAIN (stocastica)…")
    sto = run_eval(episodes, False, curriculum, run_dir, seed)
    sto_sum = summarize(sto, "Train (stocastica)", curriculum)

    cs = f"c{int(curriculum)}"
    for tag, summ, recs in [("det", det_sum, det), ("sto", sto_sum, sto)]:
        with open(os.path.join(outdir, f"metrics_{tag}_{cs}.json"), "w") as f:
            json.dump(summ, f, indent=2, default=json_default)
        with open(os.path.join(outdir, f"episodes_{tag}_{cs}.json"), "w") as f:
            json.dump([r.as_dict() for r in recs], f, indent=2, default=json_default)
    if make_plots_flag:
        print("Genero i grafici…")
        make_plots(det, det_sum, sto, sto_sum, outdir, curriculum)
    print(f"Fatto. Risultati in {outdir}")
    return dict(det=det_sum, sto=sto_sum)


def main():
    ap = argparse.ArgumentParser(description="Valutazione rigorosa della policy v1")
    ap.add_argument("--episodes", type=int, default=200,
                    help="episodi per modalità (default 200 → Wilson ±~0.03 a p~0.95)")
    ap.add_argument("--curriculum", type=float, default=1.0)
    ap.add_argument("--run-dir", type=str, default="runs/close_gen")
    ap.add_argument("--seed", type=int, default=10_000)
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()
    run(args.episodes, args.curriculum, args.run_dir, args.seed,
        make_plots_flag=not args.no_plots)


if __name__ == "__main__":
    main()