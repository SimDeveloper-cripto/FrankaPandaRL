#!/usr/bin/env python3
# scratch/test_close_task_v2/phase_diagnostics.py

"""
phase_diagnostics — Diagnostica delle fasi HOLD e RETREAT (task v2).

Porta `tests_v2/diag_phase34_v2.py` (e parte di `smoke_retreat_log_v2.py`) a statistica
a intervallo, su rollout seedati:

  T3 — ||action[:-1]|| richiesta dalla policy in HOLD (l'env la azzera comunque; misura
       se la policy "tenta" di muoversi). Distribuzione + IQM/CI.
  T4 — ||action[3:6]|| (torsione del polso) in RETREAT.
  T5 — latch_qpos all'istante della transizione HOLD→RETREAT (la FSM v2 NON aspetta il
       latch neutro: utile vedere quanto è lontano da 0). Istogramma + % > soglia.
  T6 — eventi di bounce in HOLD (door_qvel > 0.05 rad/s).

Rif.: Agarwal et al. 2021 (IQM/bootstrap); Chan et al. 2020 (rischio/dispersione).
Output in results/phase/.
"""

from __future__ import annotations

import os
import json
import argparse
import numpy as np

from _common import (make_raw_env, load_obs_rms, load_model, rollout_episode,
                     results_dir, setup_matplotlib, json_default, MODEL_SPECS)
import stats_utils as S

LATCH_TRANSITION_NEUTRAL = 0.15


def run(n_episodes, deterministic, curriculum, run_dir, base_seed=30_000, tag=None):
    tag = tag or f"c{int(curriculum)}"
    env, _cfg = make_raw_env(curriculum_level=curriculum, run_dir=run_dir)
    obs_rms = load_obs_rms(run_dir=run_dir)
    model = load_model(run_dir=run_dir)

    hold_norms, wrist_rots, latch_trans, bounces = [], [], [], []
    succ = 0
    for i in range(n_episodes):
        rec = rollout_episode(env, model, obs_rms, deterministic=deterministic,
                              seed=base_seed + i, collect_trace=True)
        succ += int(rec.success)
        hold_norms += rec.hold_action_norms
        wrist_rots += rec.retreat_wrist_rots
        bounces += rec.bounce_events
        if rec.latch_at_transition is not None:
            latch_trans.append(rec.latch_at_transition)
        lt = "n/a" if rec.latch_at_transition is None else f"{rec.latch_at_transition:+.3f}"
        print(f"  ep {i+1:>3} | succ={rec.success} | latch@transiz={lt} | "
              f"HOLD_n={len(rec.hold_action_norms)} RET_n={len(rec.retreat_wrist_rots)}")

    out = dict(mode=("det" if deterministic else "sto"), curriculum=curriculum,
               n_episodes=n_episodes, success_rate=S.wilson_ci(succ, n_episodes).as_dict())

    print("\n" + "=" * 72)
    print(f"DIAGNOSTICA FASI v2 — {'det' if deterministic else 'sto'} ({tag}, {n_episodes} ep)")
    print("=" * 72)

    if hold_norms:
        a = np.asarray(hold_norms)
        out["T3_hold_action_norm"] = dict(n=len(a), iqm=S.bootstrap_ci(a, "iqm").as_dict(),
                                          frac_below_005=float((a < 0.05).mean()),
                                          frac_above_03=float((a > 0.3).mean()), values=a.tolist())
        print(f"  T3 HOLD ||action|| : IQM {S.bootstrap_ci(a,'iqm')} | "
              f"%<0.05={(a<0.05).mean()*100:.1f}% %>0.30={(a>0.3).mean()*100:.1f}%")
    else:
        out["T3_hold_action_norm"] = None; print("  T3 HOLD ||action|| : nessun dato")

    if wrist_rots:
        a = np.asarray(wrist_rots)
        out["T4_retreat_wrist_rot"] = dict(n=len(a), iqm=S.bootstrap_ci(a, "iqm").as_dict(),
                                           frac_above_01=float((a > 0.1).mean()), values=a.tolist())
        print(f"  T4 RETREAT wrist   : IQM {S.bootstrap_ci(a,'iqm')} | %>0.1={(a>0.1).mean()*100:.1f}%")
    else:
        out["T4_retreat_wrist_rot"] = None; print("  T4 RETREAT wrist   : nessun dato")

    if latch_trans:
        a = np.asarray(latch_trans); n_bad = int((np.abs(a) > LATCH_TRANSITION_NEUTRAL).sum())
        out["T5_latch_at_transition"] = dict(n=len(a), mean=float(a.mean()), std=float(a.std()),
                                             min=float(a.min()), max=float(a.max()),
                                             frac_above_thresh=float(n_bad / len(a)),
                                             frac_above_thresh_ci=S.wilson_ci(n_bad, len(a)).as_dict(),
                                             values=a.tolist())
        print(f"  T5 latch@transiz   : mean={a.mean():+.3f} rad | "
              f"% |latch|>{LATCH_TRANSITION_NEUTRAL}: {S.wilson_ci(n_bad,len(a))}")
    else:
        out["T5_latch_at_transition"] = None; print("  T5 latch@transiz   : nessuna transizione")

    out["T6_bounce_events"] = dict(n=len(bounces),
                                   max_vel=float(max((abs(v) for _, _, v in bounces), default=0.0)),
                                   severe=int(sum(1 for _, _, v in bounces if abs(v) > 0.15)),
                                   events=[(int(s), float(p), float(v)) for s, p, v in bounces])
    print(f"  T6 bounce HOLD     : {len(bounces)} eventi, max {out['T6_bounce_events']['max_vel']:.3f} rad/s, "
          f"severi {out['T6_bounce_events']['severe']}")
    print("=" * 72 + "\n")

    plt = setup_matplotlib(); outdir = results_dir("phase")
    m = "det" if deterministic else "sto"
    if hold_norms:
        fig, ax = plt.subplots(figsize=(8, 5)); ax.hist(hold_norms, bins=40, color="#1f77b4")
        ax.axvline(0.05, ls="--", color="k"); ax.set_title(f"T3 — ||action[:-1]|| in HOLD ({tag})")
        ax.set_xlabel("norma azione braccio (richiesta policy)")
        fig.tight_layout(); fig.savefig(os.path.join(outdir, f"plot_T3_hold_norm_{m}_{tag}.png"), dpi=130); plt.close(fig)
    if wrist_rots:
        fig, ax = plt.subplots(figsize=(8, 5)); ax.hist(wrist_rots, bins=40, color="#ff7f0e")
        ax.set_title(f"T4 — torsione polso in RETREAT ({tag})"); ax.set_xlabel("norma rotazione")
        fig.tight_layout(); fig.savefig(os.path.join(outdir, f"plot_T4_wrist_{m}_{tag}.png"), dpi=130); plt.close(fig)
    if latch_trans:
        fig, ax = plt.subplots(figsize=(8, 5)); ax.hist(latch_trans, bins=30, color="#2ca02c")
        ax.axvline(LATCH_TRANSITION_NEUTRAL, ls="--", color="k")
        ax.set_title(f"T5 — latch_qpos alla transizione HOLD→RETREAT ({tag})"); ax.set_xlabel("latch_qpos (rad)")
        fig.tight_layout(); fig.savefig(os.path.join(outdir, f"plot_T5_latch_{m}_{tag}.png"), dpi=130); plt.close(fig)
    if bounces:
        fig, ax = plt.subplots(figsize=(8, 5)); ax.hist([abs(v) for _, _, v in bounces], bins=30, color="#d62728")
        ax.axvline(0.15, ls="--", color="k"); ax.set_title(f"T6 — severità bounce in HOLD ({tag})")
        ax.set_xlabel("|door_qvel| (rad/s)")
        fig.tight_layout(); fig.savefig(os.path.join(outdir, f"plot_T6_bounce_{m}_{tag}.png"), dpi=130); plt.close(fig)

    with open(os.path.join(outdir, f"phase_diag_{m}_{tag}.json"), "w") as f:
        json.dump(out, f, indent=2, default=json_default)
    print(f"Risultati in {outdir}")
    return out


def main():
    ap = argparse.ArgumentParser(description="Diagnostica fasi HOLD/RETREAT v2")
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--curriculum", type=float, default=1.0)
    ap.add_argument("--run-dir", type=str, default=None)
    ap.add_argument("--stochastic", action="store_true")
    args = ap.parse_args()
    run_dir = args.run_dir or (MODEL_SPECS[1][1] if args.curriculum >= 0.5 else MODEL_SPECS[0][1])
    run(args.episodes, not args.stochastic, args.curriculum, run_dir)


if __name__ == "__main__":
    main()