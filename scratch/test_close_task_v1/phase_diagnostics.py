#!/usr/bin/env python3
# scratch/test_close_task_v1/phase_diagnostics.py
"""
phase_diagnostics — Diagnostica delle fasi HOLD e RETREAT (task di chiusura v1).

Rifattorizza T3–T6 di `diag_phase34.py` su un rollout seedato e con statistiche a
intervallo invece di sole medie:

  T3 — ||action[:-1]|| durante HOLD: il braccio è davvero fermo? (distribuzione + IQM/CI)
  T4 — ||action[3:6]|| (torsione del polso) durante RETREAT
  T5 — latch_qpos all'istante della transizione HOLD→RETREAT (istogramma + % > soglia)
  T6 — eventi di bounce in HOLD (door_qvel > 0.05 rad/s): frequenza e severità

Strumenti statistici: IQM + bootstrap CI sulle grandezze continue; intervallo di Wilson
sulle frazioni (es. % transizioni con latch non neutro). Rif. Agarwal 2021; Chan 2020.

Output (in results/phase/): phase_diag_<mode>_c<curr>.json + 4 grafici.
"""

from __future__ import annotations

import os
import json
import argparse
import numpy as np

from _common import (make_cfg, make_vec_env, load_model, rollout_episode,
                     results_dir, setup_matplotlib)
import stats_utils as S

LATCH_TRANSITION_NEUTRAL = 0.15


def run(n_episodes: int, deterministic: bool, curriculum: float, run_dir: str,
        base_seed: int = 30_000):
    cfg = make_cfg(run_dir=run_dir)
    venv, raw_env = make_vec_env(cfg, curriculum_level=curriculum)
    model = load_model(cfg, venv)

    hold_norms, wrist_rots, latch_trans, bounces = [], [], [], []
    succ = 0
    for i in range(n_episodes):
        rec = rollout_episode(venv, model, raw_env, deterministic=deterministic,
                              seed=base_seed + i, collect_trace=True)
        succ += int(rec.success)
        hold_norms += rec.hold_action_norms
        wrist_rots += rec.retreat_wrist_rots
        bounces += rec.bounce_events
        if rec.latch_at_transition is not None:
            latch_trans.append(rec.latch_at_transition)
        print(f"  ep {i+1:>3} | succ={rec.success} | "
              f"latch@transiz={rec.latch_at_transition if rec.latch_at_transition is None else round(rec.latch_at_transition,3)} | "
              f"HOLD_norms={len(rec.hold_action_norms)} RET_rots={len(rec.retreat_wrist_rots)}")
    venv.close()

    out = dict(mode=("det" if deterministic else "sto"), curriculum=curriculum,
               n_episodes=n_episodes, success_rate=S.wilson_ci(succ, n_episodes).as_dict())

    print("\n" + "=" * 72)
    print(f"DIAGNOSTICA FASI — {'det' if deterministic else 'sto'} "
          f"(curriculum={curriculum}, {n_episodes} ep)")
    print("=" * 72)

    # T3
    if hold_norms:
        a = np.asarray(hold_norms)
        out["T3_hold_action_norm"] = dict(
            n=len(a), iqm=S.bootstrap_ci(a, "iqm").as_dict(),
            frac_below_005=float((a < 0.05).mean()),
            frac_above_03=float((a > 0.3).mean()),
            values=a.tolist())
        print(f"  T3 HOLD ||action|| : IQM {S.bootstrap_ci(a,'iqm')} | "
              f"%<0.05={ (a<0.05).mean()*100:.1f}% %>0.30={(a>0.3).mean()*100:.1f}%")
    else:
        out["T3_hold_action_norm"] = None
        print("  T3 HOLD ||action|| : nessun dato (nessun episodio in HOLD)")

    # T4
    if wrist_rots:
        a = np.asarray(wrist_rots)
        out["T4_retreat_wrist_rot"] = dict(
            n=len(a), iqm=S.bootstrap_ci(a, "iqm").as_dict(),
            frac_above_01=float((a > 0.1).mean()), values=a.tolist())
        print(f"  T4 RETREAT wrist   : IQM {S.bootstrap_ci(a,'iqm')} | "
              f"%>0.1={(a>0.1).mean()*100:.1f}%")
    else:
        out["T4_retreat_wrist_rot"] = None
        print("  T4 RETREAT wrist   : nessun dato")

    # T5
    if latch_trans:
        a = np.asarray(latch_trans)
        n_bad = int((np.abs(a) > LATCH_TRANSITION_NEUTRAL).sum())
        out["T5_latch_at_transition"] = dict(
            n=len(a), mean=float(a.mean()), std=float(a.std()),
            min=float(a.min()), max=float(a.max()),
            frac_above_thresh=float(n_bad / len(a)),
            frac_above_thresh_ci=S.wilson_ci(n_bad, len(a)).as_dict(),
            values=a.tolist())
        print(f"  T5 latch@transiz   : mean={a.mean():+.3f} rad | "
              f"% |latch|>{LATCH_TRANSITION_NEUTRAL}: {S.wilson_ci(n_bad,len(a))}")
    else:
        out["T5_latch_at_transition"] = None
        print("  T5 latch@transiz   : nessuna transizione osservata")

    # T6
    out["T6_bounce_events"] = dict(
        n=len(bounces),
        max_vel=float(max((abs(v) for _, _, v in bounces), default=0.0)),
        severe=int(sum(1 for _, _, v in bounces if abs(v) > 0.15)),
        events=[(int(s), float(p), float(v)) for s, p, v in bounces])
    print(f"  T6 bounce HOLD     : {len(bounces)} eventi, "
          f"max {out['T6_bounce_events']['max_vel']:.3f} rad/s, "
          f"severi {out['T6_bounce_events']['severe']}")
    print("=" * 72 + "\n")

    # ── plot ──
    plt = setup_matplotlib()
    outdir = results_dir("phase")
    cs = f"c{int(curriculum)}"; m = "det" if deterministic else "sto"

    if hold_norms:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(hold_norms, bins=40, color="#1f77b4")
        ax.axvline(0.05, ls="--", color="k")
        ax.set_title("T3 — ||action[:-1]|| durante HOLD"); ax.set_xlabel("norma azione braccio")
        fig.tight_layout(); fig.savefig(os.path.join(outdir, f"plot_T3_hold_norm_{m}_{cs}.png"), dpi=130); plt.close(fig)
    if wrist_rots:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(wrist_rots, bins=40, color="#ff7f0e")
        ax.set_title("T4 — torsione polso ||action[3:6]|| in RETREAT"); ax.set_xlabel("norma rotazione")
        fig.tight_layout(); fig.savefig(os.path.join(outdir, f"plot_T4_wrist_{m}_{cs}.png"), dpi=130); plt.close(fig)
    if latch_trans:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(latch_trans, bins=30, color="#2ca02c")
        ax.axvline(LATCH_TRANSITION_NEUTRAL, ls="--", color="k")
        ax.set_title("T5 — latch_qpos alla transizione HOLD→RETREAT"); ax.set_xlabel("latch_qpos (rad)")
        fig.tight_layout(); fig.savefig(os.path.join(outdir, f"plot_T5_latch_{m}_{cs}.png"), dpi=130); plt.close(fig)
    if bounces:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist([abs(v) for _, _, v in bounces], bins=30, color="#d62728")
        ax.axvline(0.15, ls="--", color="k")
        ax.set_title("T6 — severità bounce in HOLD"); ax.set_xlabel("|door_qvel| (rad/s)")
        fig.tight_layout(); fig.savefig(os.path.join(outdir, f"plot_T6_bounce_{m}_{cs}.png"), dpi=130); plt.close(fig)

    with open(os.path.join(outdir, f"phase_diag_{m}_{cs}.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"Risultati in {outdir}")
    return out


def main():
    ap = argparse.ArgumentParser(description="Diagnostica fasi HOLD/RETREAT v1")
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--curriculum", type=float, default=1.0)
    ap.add_argument("--run-dir", type=str, default="runs/close_gen")
    ap.add_argument("--stochastic", action="store_true")
    args = ap.parse_args()
    run(args.episodes, not args.stochastic, args.curriculum, args.run_dir)


if __name__ == "__main__":
    main()
