#!/usr/bin/env python3
# scratch/test_open_task_v2/evaluate_policy.py
"""
evaluate_policy — Valutazione statisticamente rigorosa della policy (task APERTURA v2).

Speculare a `evaluate_policy.py` della chiusura; estende il vecchio
`diagnose_phase.py` (che stampava conteggi grezzi su 5 episodi) con:

  1. STIME A INTERVALLO invece di percentuali nude: ogni tasso ha l'intervallo di
     Wilson (Wilson 1927; Brown, Cai & DasGupta 2001), che a differenza di
     p ± 1.96·√(p(1−p)/n) resta valido anche vicino a 0% e 100% — il regime in cui
     l'apertura si trova spesso durante il debug.
  2. METRICHE ROBUSTE: IQM + bootstrap CI su lunghezza episodio ed errore al goal
     (Agarwal et al. 2021): l'IQM scarta il 25% delle code, quindi non è dominato né
     dai pochi episodi perfetti né dai pochi catastrofici.
  3. RISCHIO: CVaR sul 10% peggiore (Chan et al. 2020) — quanto è brutto il caso brutto.
  4. TRE LIVELLI DI SUCCESSO, perché nell'apertura "successo" è ambiguo:
       success        — permissivo: la FSM ha raggiunto HOLD_OPEN/RETREAT (info is_success)
       true_success   — + porta ANCORA al goal e leva NEUTRA a fine episodio
       clean_success  — + ritiro reale (braccio allontanato ≥6 cm, §1.55) e uscita PULITA
     Il divario fra i tre è la diagnosi: è esattamente la distinzione
     PULITA / ESOGENA / HARD-CAP introdotta in diagnose_phase.py §1.43/§1.45.
  5. RIPRODUCIBILITÀ: episodi seedati (base_seed + i), riusati identici dalle
     ablazioni (confronto appaiato — Patterson et al. 2024; Henderson et al. 2018).
  6. NUMEROSITÀ MOTIVATA: default 200 episodi → semiampiezza di Wilson ≈ ±0.03 a
     p ≈ 0.9 (vs ±0.07 con i 30 episodi degli script originali).

Valutazione in modalità deterministica (policy di deployment) e stocastica (la
distribuzione vista in training): la differenza fra le due è informativa in SAC.

Output in results/evaluate/: metrics_*.json, episodes_*.json, 6 grafici.
"""

from __future__ import annotations

import os
import json
import argparse
from collections import Counter

import numpy as np

from _common import (make_raw_env, load_obs_rms, load_model, rollout_episode,
                     results_dir, setup_matplotlib, safe_hist, PHASE_NAMES, FAILURE_TYPES,
                     TERMINATION_TYPES, json_default, resolve_run_dir, CURRICULUM,
                     STUCK_MOVE_THRESH)
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
        print(f"  ep {i+1:>3}/{n_episodes} | succ={rec.success} true={rec.true_success} "
              f"clean={rec.clean_success} | len={rec.length:>3} "
              f"| open_err={rec.open_error_end:.4f} | latch={rec.latch_end:+.3f} "
              f"| mosso={rec.retreat_moved_max:.3f}m | {rec.termination_type} | {rec.failure_type}")
    try:
        env.close()
    except Exception:
        pass
    return records


def summarize(records, mode_name, curriculum):
    n = len(records)
    succ = sum(r.success for r in records)
    tsucc = sum(r.true_success for r in records)
    csucc = sum(r.clean_success for r in records)
    lengths = [r.length for r in records]
    open_end = [r.open_error_end for r in records]
    signed_end = [r.door_angle_end - r.goal_angle for r in records]
    open_min = [r.min_open_error for r in records]
    moved = [r.retreat_moved_max for r in records]

    sr = S.wilson_ci(succ, n)
    tsr = S.wilson_ci(tsucc, n)
    csr = S.wilson_ci(csucc, n)
    len_iqm = S.bootstrap_ci(lengths, "iqm")
    oe_iqm = S.bootstrap_ci(open_end, "iqm")
    om_iqm = S.bootstrap_ci(open_min, "iqm")
    mv_iqm = S.bootstrap_ci(moved, "iqm")

    fail_counts = Counter(r.failure_type for r in records)
    failure_breakdown = {ft: dict(count=fail_counts.get(ft, 0),
                                  ci=S.wilson_ci(fail_counts.get(ft, 0), n).as_dict())
                         for ft in FAILURE_TYPES}
    term_counts = Counter(r.termination_type for r in records)
    termination_breakdown = {tt: dict(count=term_counts.get(tt, 0),
                                      ci=S.wilson_ci(term_counts.get(tt, 0), n).as_dict())
                             for tt in TERMINATION_TYPES}
    phase_counts = Counter(r.max_phase for r in records)
    n_stuck = sum(1 for r in records if r.retreat_moved_max < STUCK_MOVE_THRESH
                  and r.max_phase == "RETREAT")
    n_ret = max(1, sum(1 for r in records if r.max_phase == "RETREAT"))

    summary = dict(
        mode=mode_name, curriculum=curriculum, n_episodes=n,
        success_rate=sr.as_dict(), true_success_rate=tsr.as_dict(),
        clean_success_rate=csr.as_dict(),
        length_iqm=len_iqm.as_dict(), length_mean=float(np.mean(lengths)),
        length_cvar_worst10=S.cvar(lengths, 0.1, lower_tail=False),
        open_error_end_iqm=oe_iqm.as_dict(),
        signed_error_end=dict(mean=float(np.mean(signed_end)),
                              median=float(np.median(signed_end)),
                              frac_overshoot=float(np.mean(np.array(signed_end) > 0))),
        open_error_end_cvar_worst10=S.cvar(open_end, 0.1, lower_tail=False),
        min_open_error_iqm=om_iqm.as_dict(),
        retreat_moved_iqm=mv_iqm.as_dict(),
        retreat_moved_cvar_worst10=S.cvar(moved, 0.1, lower_tail=True),
        stuck_on_handle=dict(count=n_stuck, of_retreat_episodes=n_ret,
                             ci=S.wilson_ci(n_stuck, n_ret).as_dict()),
        max_phase_distribution={p: phase_counts.get(p, 0) for p in PHASE_NAMES},
        failure_breakdown=failure_breakdown,
        termination_breakdown=termination_breakdown,
        goal_angle=dict(mean=float(np.mean([r.goal_angle for r in records])),
                        min=float(np.min([r.goal_angle for r in records])),
                        max=float(np.max([r.goal_angle for r in records]))),
    )

    print("\n" + "=" * 76)
    print(f"VALUTAZIONE APERTURA — {mode_name} (curriculum={curriculum}, {n} episodi)")
    print("=" * 76)
    print(f"  Success (permissivo, fase≥HOLD_OPEN) : {sr}")
    print(f"  True success (al goal + leva neutra) : {tsr}")
    print(f"  Clean success (+ ritiro reale)       : {csr}")
    print(f"  Lunghezza episodio (IQM)             : {len_iqm}")
    print(f"  Lunghezza, CVaR peggior 10%          : {summary['length_cvar_worst10']:.1f} step")
    print(f"  open_error finale (IQM)              : {oe_iqm}")
    print(f"  errore CON SEGNO (fine-goal), media   : {np.mean(signed_end):+.4f} rad "
          f"({np.mean(np.array(signed_end) > 0)*100:.0f}% oltre il goal)")
    print(f"  open_error finale, CVaR peggior 10%  : {summary['open_error_end_cvar_worst10']:.4f} rad")
    print(f"  open_error minimo (IQM)              : {om_iqm}")
    print(f"  Allontanamento in RETREAT (IQM)      : {mv_iqm} m")
    print(f"  Braccio fermo sulla maniglia         : {S.wilson_ci(n_stuck, n_ret)} (sui RETREAT)")
    print("  " + "-" * 72)
    print("  Tipo di terminazione                        count   95% CI")
    for tt in TERMINATION_TYPES:
        b = termination_breakdown[tt]; ci = b["ci"]
        print(f"    {tt:<33} {b['count']:>5}   [{ci['lo']*100:4.1f}, {ci['hi']*100:4.1f}]%")
    print("  " + "-" * 72)
    print("  Modo di fallimento                          count   95% CI")
    for ft in FAILURE_TYPES:
        b = failure_breakdown[ft]; ci = b["ci"]
        print(f"    {ft:<33} {b['count']:>5}   [{ci['lo']*100:4.1f}, {ci['hi']*100:4.1f}]%")
    print("=" * 76 + "\n")
    return summary


def make_plots(det_records, det_summary, sto_records, sto_summary, outdir, curriculum, tag):
    plt = setup_matplotlib()
    colors = ["#d62728", "#ff7f0e", "#1f77b4", "#2ca02c"]

    # 1) i tre livelli di successo con CI
    fig, ax = plt.subplots(figsize=(9, 6))
    x = np.arange(2); w = 0.26
    series = [("success_rate", "permissivo", "#1f77b4"),
              ("true_success_rate", "true success", "#2ca02c"),
              ("clean_success_rate", "clean success", "#9467bd")]
    for j, (key, lab, col) in enumerate(series):
        vals = [det_summary[key], sto_summary[key]]
        pts = [v["point"] * 100 for v in vals]
        errs = np.clip([[(v["point"] - v["lo"]) * 100 for v in vals],
                        [(v["hi"] - v["point"]) * 100 for v in vals]], 0, None)
        ax.bar(x + (j - 1) * w, pts, w, yerr=errs, capsize=5, label=lab, color=col)
    ax.set_xticks(x); ax.set_xticklabels(["Eval det", "Train sto"])
    ax.set_ylabel("Tasso (%)"); ax.set_ylim(0, 105)
    ax.set_title(f"Successo a tre livelli ± Wilson 95% CI ({tag})\n"
                 "il divario permissivo→true→clean è la diagnosi del ritiro")
    ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(outdir, f"plot_success_{tag}.png"), dpi=130); plt.close(fig)

    # 2) distribuzione fase massima
    fig, ax = plt.subplots(figsize=(9, 6))
    for name, recs in [("Eval det", det_records), ("Train sto", sto_records)]:
        bottom = 0; cnt = Counter(r.max_phase for r in recs)
        for i, ph in enumerate(PHASE_NAMES):
            v = cnt.get(ph, 0)
            ax.bar(name, v, bottom=bottom, color=colors[i], label=ph if name == "Eval det" else "")
            if v:
                ax.text(name, bottom + v / 2, str(v), ha="center", va="center",
                        color="white", fontweight="bold")
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
    ax.set_ylabel("Step medi"); ax.set_title(f"Tempo per fase (media ± bootstrap 95% CI) ({tag})")
    ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(outdir, f"plot_phase_time_{tag}.png"), dpi=130); plt.close(fig)

    # 4) distribuzioni per-episodio
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].boxplot([[r.length for r in det_records], [r.length for r in sto_records]],
                    labels=["det", "sto"], showmeans=True)
    axes[0].set_title("Lunghezza episodio"); axes[0].set_ylabel("step")
    axes[1].boxplot([[r.open_error_end for r in det_records], [r.open_error_end for r in sto_records]],
                    labels=["det", "sto"], showmeans=True)
    axes[1].set_title("open_error finale"); axes[1].set_ylabel("rad")
    axes[2].boxplot([[r.retreat_moved_max for r in det_records],
                     [r.retreat_moved_max for r in sto_records]],
                    labels=["det", "sto"], showmeans=True)
    axes[2].axhline(STUCK_MOVE_THRESH, ls="--", color="k")
    axes[2].set_title("Allontanamento in RETREAT"); axes[2].set_ylabel("m")
    fig.suptitle(f"Distribuzioni per-episodio ({tag}) — la dispersione conta (Chan et al. 2020)")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, f"plot_metric_distributions_{tag}.png"), dpi=130); plt.close(fig)

    # 5) breakdown esiti + terminazioni (eval det)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fb = det_summary["failure_breakdown"]; types = list(FAILURE_TYPES)
    pts = [fb[t]["ci"]["point"] * 100 for t in types]
    errs = np.clip([[(fb[t]["ci"]["point"] - fb[t]["ci"]["lo"]) * 100 for t in types],
                    [(fb[t]["ci"]["hi"] - fb[t]["ci"]["point"]) * 100 for t in types]], 0, None)
    yy = np.arange(len(types))
    axes[0].barh(yy, pts, xerr=errs, capsize=4, color="#8c564b")
    axes[0].set_yticks(yy); axes[0].set_yticklabels(types); axes[0].invert_yaxis()
    axes[0].set_xlabel("% episodi (± Wilson 95% CI)"); axes[0].set_title("Esiti")
    tb = det_summary["termination_breakdown"]; tt = list(TERMINATION_TYPES)
    pts2 = [tb[t]["ci"]["point"] * 100 for t in tt]
    errs2 = np.clip([[(tb[t]["ci"]["point"] - tb[t]["ci"]["lo"]) * 100 for t in tt],
                     [(tb[t]["ci"]["hi"] - tb[t]["ci"]["point"]) * 100 for t in tt]], 0, None)
    yy2 = np.arange(len(tt))
    axes[1].barh(yy2, pts2, xerr=errs2, capsize=4, color="#17becf")
    axes[1].set_yticks(yy2); axes[1].set_yticklabels(tt); axes[1].invert_yaxis()
    axes[1].set_xlabel("% episodi (± Wilson 95% CI)"); axes[1].set_title("Terminazioni")
    fig.suptitle(f"Breakdown — Eval deterministico ({tag})")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, f"plot_failure_breakdown_{tag}.png"), dpi=130); plt.close(fig)

    # 6) distribuzione bootstrap del true success
    fig, ax = plt.subplots(figsize=(8, 5))
    succ = np.array([1.0 if r.true_success else 0.0 for r in det_records])
    rng = np.random.default_rng(0)
    boots = rng.choice(succ, size=(10_000, len(succ)), replace=True).mean(axis=1)
    safe_hist(ax, boots * 100, 40, color="#2ca02c", alpha=0.85)
    sr = det_summary["true_success_rate"]
    ax.axvline(sr["lo"] * 100, color="k", ls="--"); ax.axvline(sr["hi"] * 100, color="k", ls="--")
    ax.set_xlabel("True success bootstrap (%)"); ax.set_ylabel("frequenza")
    ax.set_title(f"Distribuzione bootstrap del true success — Eval det ({tag})")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, f"plot_bootstrap_{tag}.png"), dpi=130); plt.close(fig)


def run(episodes=200, curriculum=CURRICULUM, run_dir=None, seed=10_000,
        make_plots_flag=True, tag=None):
    run_dir = resolve_run_dir(run_dir)
    tag = tag or "curr1_posa_variabile"
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
    ap = argparse.ArgumentParser(description="Valutazione rigorosa della policy di apertura v2")
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--curriculum", type=float, default=CURRICULUM)
    ap.add_argument("--run-dir", type=str, default=None)
    ap.add_argument("--seed", type=int, default=10_000)
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()
    run(args.episodes, args.curriculum, args.run_dir, args.seed,
        make_plots_flag=not args.no_plots)


if __name__ == "__main__":
    main()
