#!/usr/bin/env python3
# scratch/test_close_task_v2/evaluate_policy.py

"""
evaluate_policy — Valutazione statisticamente rigorosa della policy (task v2).

Estende `tests_v2/eval_stats_close_v2.py` con la stessa metodologia della suite v1:

  1. STIME A INTERVALLO: success rate con intervallo di Wilson; lunghezza e angolo
     minimo con IQM + bootstrap CI (Agarwal et al. 2021; Henderson et al. 2018).
  2. METRICA DI RISCHIO: CVaR sulla coda peggiore (Chan et al. 2020).
  3. TRUE SUCCESS separato dal permissivo (porta chiusa |door|<0.03 E latch neutro
     |latch|<0.08 a fine episodio).
  4. RIPRODUCIBILITÀ: episodi seedati (base_seed + i); stessi seed riusati dalle
     ablazioni (confronto appaiato — Patterson et al. 2024).
  5. NUMEROSITÀ MOTIVATA: default 200 episodi (Wilson ±~0.03 a p~0.95). Era 50.

Output (in results/evaluate/): metrics_*.json, episodes_*.json, 6 grafici.
"""

from __future__ import annotations

import os
import json
import argparse
from collections import Counter

import numpy as np

from _common import (make_raw_env, load_obs_rms, load_model, rollout_episode,
                     results_dir, setup_matplotlib, PHASE_NAMES, FAILURE_TYPES,
                     json_default, MODEL_SPECS)
import stats_utils as S


def run_eval(n_episodes, deterministic, curriculum, run_dir, base_seed=10_000):
    env, _cfg = make_raw_env(curriculum_level=curriculum, run_dir=run_dir)
    obs_rms = load_obs_rms(run_dir=run_dir)
    model = load_model(run_dir=run_dir)

    records = []
    for i in range(n_episodes):
        rec = rollout_episode(env, model, obs_rms,
                              deterministic=deterministic, seed=base_seed + i)
        records.append(rec)
        tag = "SUCCESS" if rec.success else rec.failure_type
        print(f"  ep {i+1:>3}/{n_episodes} | succ={rec.success} true={rec.true_success} "
              f"| len={rec.length:>3} | minθ={rec.min_door_angle:.4f} | {tag}")
    return records


def summarize(records, mode_name, curriculum):
    n = len(records)
    succ = sum(r.success for r in records)
    tsucc = sum(r.true_success for r in records)
    lengths = [r.length for r in records]
    min_doors = [r.min_door_angle for r in records]

    sr = S.wilson_ci(succ, n)
    tsr = S.wilson_ci(tsucc, n)
    len_iqm = S.bootstrap_ci(lengths, "iqm")
    door_iqm = S.bootstrap_ci(min_doors, "iqm")

    fail_counts = Counter(r.failure_type for r in records)
    failure_breakdown = {ft: dict(count=fail_counts.get(ft, 0),
                                  ci=S.wilson_ci(fail_counts.get(ft, 0), n).as_dict())
                         for ft in FAILURE_TYPES}
    phase_counts = Counter(r.max_phase for r in records)

    summary = dict(
        mode=mode_name, curriculum=curriculum, n_episodes=n,
        success_rate=sr.as_dict(), true_success_rate=tsr.as_dict(),
        length_iqm=len_iqm.as_dict(), length_mean=float(np.mean(lengths)),
        length_cvar_worst10=S.cvar(lengths, 0.1, lower_tail=False),
        min_door_iqm=door_iqm.as_dict(),
        min_door_cvar_worst10=S.cvar(min_doors, 0.1, lower_tail=False),
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
    print("  Modo di fallimento                          count   95% CI")
    for ft in FAILURE_TYPES:
        b = failure_breakdown[ft]; ci = b["ci"]
        print(f"    {ft:<33} {b['count']:>5}   [{ci['lo']*100:4.1f}, {ci['hi']*100:4.1f}]%")
    print("=" * 72 + "\n")
    return summary


def make_plots(det_records, det_summary, sto_records, sto_summary, outdir, curriculum, tag):
    plt = setup_matplotlib()
    colors = ["#d62728", "#ff7f0e", "#1f77b4", "#2ca02c"]

    # 1) success rate con CI
    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.arange(2); w = 0.35
    for j, (key, lab, col) in enumerate([("success_rate", "permissivo", "#1f77b4"),
                                         ("true_success_rate", "true success", "#2ca02c")]):
        vals = [det_summary[key], sto_summary[key]]
        pts = [v["point"] * 100 for v in vals]
        errs = np.clip([[(v["point"] - v["lo"]) * 100 for v in vals],
                        [(v["hi"] - v["point"]) * 100 for v in vals]], 0, None)
        ax.bar(x + (j - 0.5) * w, pts, w, yerr=errs, capsize=5, label=lab, color=col)
    ax.set_xticks(x); ax.set_xticklabels(["Eval det", "Train sto"])
    ax.set_ylabel("Success rate (%)"); ax.set_ylim(0, 105)
    ax.set_title(f"Success rate ± Wilson 95% CI ({tag})"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(outdir, f"plot_success_{tag}.png"), dpi=130); plt.close(fig)

    # 2) distribuzione fase massima
    fig, ax = plt.subplots(figsize=(9, 6))
    for name, recs in [("Eval det", det_records), ("Train sto", sto_records)]:
        bottom = 0; cnt = Counter(r.max_phase for r in recs)
        for i, ph in enumerate(PHASE_NAMES):
            v = cnt.get(ph, 0)
            ax.bar(name, v, bottom=bottom, color=colors[i], label=ph if name == "Eval det" else "")
            if v:
                ax.text(name, bottom + v / 2, str(v), ha="center", va="center", color="white", fontweight="bold")
            bottom += v
    ax.set_ylabel("Episodi"); ax.set_title(f"Fase massima raggiunta ({tag})"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(outdir, f"plot_phase_dist_{tag}.png"), dpi=130); plt.close(fig)

    # 3) tempo per fase ± CI
    fig, ax = plt.subplots(figsize=(9, 6))
    xs = np.arange(len(PHASE_NAMES)); w = 0.35
    for j, (name, recs, col) in enumerate([("Eval det", det_records, "#1f77b4"),
                                           ("Train sto", sto_records, "#2ca02c")]):
        means, elo, ehi = [], [], []
        for ph in PHASE_NAMES:
            ci = S.bootstrap_ci([r.phase_times[ph] for r in recs], "mean")
            means.append(ci.point); elo.append(max(0.0, ci.point - ci.lo)); ehi.append(max(0.0, ci.hi - ci.point))
        ax.bar(xs + (j - 0.5) * w, means, w, yerr=[elo, ehi], capsize=4, label=name, color=col)
    ax.set_xticks(xs); ax.set_xticklabels(PHASE_NAMES)
    ax.set_ylabel("Step medi"); ax.set_title(f"Tempo per fase (media ± bootstrap 95% CI) ({tag})"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(outdir, f"plot_phase_time_{tag}.png"), dpi=130); plt.close(fig)

    # 4) distribuzioni per-episodio
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].boxplot([[r.length for r in det_records], [r.length for r in sto_records]],
                    labels=["det", "sto"], showmeans=True)
    axes[0].set_title("Lunghezza episodio"); axes[0].set_ylabel("step")
    axes[1].boxplot([[r.min_door_angle for r in det_records], [r.min_door_angle for r in sto_records]],
                    labels=["det", "sto"], showmeans=True)
    axes[1].set_title("Min door angle"); axes[1].set_ylabel("rad")
    fig.suptitle(f"Distribuzioni per-episodio ({tag}) — la dispersione conta (Chan 2020)")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, f"plot_metric_distributions_{tag}.png"), dpi=130); plt.close(fig)

    # 5) breakdown fallimenti con CI (eval det)
    fig, ax = plt.subplots(figsize=(10, 6))
    fb = det_summary["failure_breakdown"]; types = list(FAILURE_TYPES)
    pts = [fb[t]["ci"]["point"] * 100 for t in types]
    errs = np.clip([[(fb[t]["ci"]["point"] - fb[t]["ci"]["lo"]) * 100 for t in types],
                    [(fb[t]["ci"]["hi"] - fb[t]["ci"]["point"]) * 100 for t in types]], 0, None)
    yy = np.arange(len(types))
    ax.barh(yy, pts, xerr=errs, capsize=4, color="#8c564b")
    ax.set_yticks(yy); ax.set_yticklabels(types); ax.invert_yaxis()
    ax.set_xlabel("% episodi (± Wilson 95% CI)"); ax.set_title(f"Breakdown esiti — Eval det ({tag})")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, f"plot_failure_breakdown_{tag}.png"), dpi=130); plt.close(fig)

    # 6) distribuzione bootstrap del success rate
    fig, ax = plt.subplots(figsize=(8, 5))
    succ = np.array([1.0 if r.success else 0.0 for r in det_records])
    rng = np.random.default_rng(0)
    boots = rng.choice(succ, size=(10_000, len(succ)), replace=True).mean(axis=1)
    ax.hist(boots * 100, bins=40, color="#1f77b4", alpha=0.8)
    sr = det_summary["success_rate"]
    ax.axvline(sr["lo"] * 100, color="k", ls="--"); ax.axvline(sr["hi"] * 100, color="k", ls="--")
    ax.set_xlabel("Success rate bootstrap (%)"); ax.set_ylabel("frequenza")
    ax.set_title(f"Distribuzione bootstrap del success rate — Eval det ({tag})")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, f"plot_bootstrap_{tag}.png"), dpi=130); plt.close(fig)


def run(episodes=200, curriculum=1.0, run_dir=None, seed=10_000, make_plots_flag=True, tag=None):
    run_dir = run_dir or (MODEL_SPECS[1][1] if curriculum >= 0.5 else MODEL_SPECS[0][1])
    tag = tag or f"c{int(curriculum)}"
    outdir = results_dir("evaluate")

    print(f"[{tag}] Valutazione EVAL (deterministica)…")
    det = run_eval(episodes, True, curriculum, run_dir, seed)
    det_sum = summarize(det, "Eval (deterministica)", curriculum)
    print(f"[{tag}] Valutazione TRAIN (stocastica)…")
    sto = run_eval(episodes, False, curriculum, run_dir, seed)
    sto_sum = summarize(sto, "Train (stocastica)", curriculum)

    for t, summ, recs in [("det", det_sum, det), ("sto", sto_sum, sto)]:
        with open(os.path.join(outdir, f"metrics_{t}_{tag}.json"), "w") as f:
            json.dump(summ, f, indent=2, default=json_default)
        with open(os.path.join(outdir, f"episodes_{t}_{tag}.json"), "w") as f:
            json.dump([r.as_dict() for r in recs], f, indent=2, default=json_default)
    if make_plots_flag:
        print("Genero i grafici…")
        make_plots(det, det_sum, sto, sto_sum, outdir, curriculum, tag)
    print(f"Fatto. Risultati in {outdir}")
    return dict(det=det_sum, sto=sto_sum)


def main():
    ap = argparse.ArgumentParser(description="Valutazione rigorosa della policy v2")
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--curriculum", type=float, default=1.0)
    ap.add_argument("--run-dir", type=str, default=None,
                    help="default: dir del modello in base al curriculum")
    ap.add_argument("--seed", type=int, default=10_000)
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()
    run(args.episodes, args.curriculum, args.run_dir, args.seed, make_plots_flag=not args.no_plots)


if __name__ == "__main__":
    main()