#!/usr/bin/env python3
# scratch/test_close_task_v2/robustness_analysis.py

"""
robustness_analysis — Inviluppo operativo della policy (task v2).

Come nella suite v1, ma sfrutta gli assi di randomization aggiuntivi della v2
(rigidità latch, smorzamento cerniera, massa porta — domain_rand_v2 §3.4). Risponde
alla domanda «dove» generalizza la policy, non con un singolo numero ma con curve
esito-vs-parametro (Tobin et al. 2017; Mehta et al. 2020; Zhao et al. 2020;
stile ten Pas et al. 2017, Fig. 5).

Metodo: STRATIFICAZIONE (robusto). Si raccolgono molti episodi con randomization
naturale, si registra il parametro realizzato e il true_success, si raggruppa in bin e
si calcola il success ± intervallo di Wilson per bin. Più due heatmap 2D.

Output in results/robustness/.
"""

from __future__ import annotations

import os
import json
import argparse
import numpy as np

from _common import (make_raw_env, load_obs_rms, load_model, rollout_episode,
                     results_dir, setup_matplotlib, json_default, MODEL_SPECS)
import stats_utils as S

AXES = {
    "handle_friction": "Frizione maniglia",
    "handle_radius": "Raggio maniglia (m)",
    "latch_stiffness_ratio": "Rigidità latch (×base)",
    "hinge_damping_ratio": "Smorzamento cerniera (×base)",
    "door_mass_ratio": "Massa porta (×base)",
    "door_x": "Distanza porta (x, m)",
}


def collect(n_episodes, curriculum, run_dir, deterministic, base_seed=40_000):
    env, _cfg = make_raw_env(curriculum_level=curriculum, run_dir=run_dir)
    obs_rms = load_obs_rms(run_dir=run_dir)
    model = load_model(run_dir=run_dir)
    recs = []
    for i in range(n_episodes):
        recs.append(rollout_episode(env, model, obs_rms, deterministic=deterministic, seed=base_seed + i))
        if (i + 1) % 25 == 0:
            print(f"  raccolti {i+1}/{n_episodes} episodi")
    return recs


def stratify_1d(recs, key, n_bins=6):
    vals = np.array([getattr(r, key) for r in recs], float)
    out = np.array([1 if r.true_success else 0 for r in recs], int)
    finite = np.isfinite(vals); vals, out = vals[finite], out[finite]
    if len(vals) == 0:
        return []
    edges = np.unique(np.quantile(vals, np.linspace(0, 1, n_bins + 1)))
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (vals >= lo) & (vals <= hi if hi == edges[-1] else vals < hi)
        n = int(mask.sum())
        if n == 0:
            continue
        s = int(out[mask].sum()); ci = S.wilson_ci(s, n)
        bins.append(dict(lo=float(lo), hi=float(hi), center=float((lo + hi) / 2),
                         n=n, successes=s, rate=ci.point, ci_lo=ci.lo, ci_hi=ci.hi))
    return bins


def heatmap_2d(recs, kx, ky, nb=4):
    vx = np.array([getattr(r, kx) for r in recs], float)
    vy = np.array([getattr(r, ky) for r in recs], float)
    oc = np.array([1 if r.true_success else 0 for r in recs], int)
    ok = np.isfinite(vx) & np.isfinite(vy); vx, vy, oc = vx[ok], vy[ok], oc[ok]
    if len(vx) == 0:
        return None
    ex = np.unique(np.quantile(vx, np.linspace(0, 1, nb + 1)))
    ey = np.unique(np.quantile(vy, np.linspace(0, 1, nb + 1)))
    rate = np.full((len(ey) - 1, len(ex) - 1), np.nan); count = np.zeros_like(rate)
    for j in range(len(ey) - 1):
        for i in range(len(ex) - 1):
            mx = (vx >= ex[i]) & (vx <= ex[i + 1] if i == len(ex) - 2 else vx < ex[i + 1])
            my = (vy >= ey[j]) & (vy <= ey[j + 1] if j == len(ey) - 2 else vy < ey[j + 1])
            m = mx & my
            if m.sum() > 0:
                rate[j, i] = oc[m].mean(); count[j, i] = m.sum()
    return dict(ex=ex.tolist(), ey=ey.tolist(), rate=rate.tolist(), count=count.tolist(),
               kx=kx, ky=ky)


def run(n_episodes, curriculum, run_dir, deterministic=True, tag=None):
    tag = tag or f"c{int(curriculum)}"
    print("=" * 72)
    print(f"ROBUSTEZZA / INVILUPPO OPERATIVO v2 ({tag}, {n_episodes} ep)")
    print("=" * 72)
    recs = collect(n_episodes, curriculum, run_dir, deterministic)

    out = dict(curriculum=curriculum, n_episodes=n_episodes,
               mode=("det" if deterministic else "sto"),
               overall_true_success=S.wilson_ci(sum(r.true_success for r in recs), len(recs)).as_dict(),
               envelopes={})
    for key, label in AXES.items():
        bins = stratify_1d(recs, key); out["envelopes"][key] = bins
        if bins:
            print(f"\n  {label}:")
            for b in bins:
                print(f"    [{b['lo']:.3f},{b['hi']:.3f}]  n={b['n']:>3}  "
                      f"true={b['rate']*100:5.1f}% [{b['ci_lo']*100:4.1f},{b['ci_hi']*100:4.1f}]")
    out["heatmaps"] = {
        "friction_radius": heatmap_2d(recs, "handle_friction", "handle_radius"),
        "latch_mass": heatmap_2d(recs, "latch_stiffness_ratio", "door_mass_ratio"),
    }
    print(f"\n  True success complessivo: {S.wilson_ci(sum(r.true_success for r in recs), len(recs))}")
    print("=" * 72)

    outdir = results_dir("robustness")
    with open(os.path.join(outdir, f"robustness_{tag}.json"), "w") as f:
        json.dump(out, f, indent=2, default=json_default)

    plt = setup_matplotlib()
    # curve 1D (griglia 2×3)
    fig, axs = plt.subplots(2, 3, figsize=(16, 8)); axs = axs.ravel()
    for ax, (key, label) in zip(axs, AXES.items()):
        bins = out["envelopes"][key]
        if not bins:
            ax.set_title(f"{label}\n(no data)"); continue
        c = [b["center"] for b in bins]; r = [b["rate"] * 100 for b in bins]
        lo = [max(0.0, (b["rate"] - b["ci_lo"]) * 100) for b in bins]
        hi = [max(0.0, (b["ci_hi"] - b["rate"]) * 100) for b in bins]
        ax.errorbar(c, r, yerr=[lo, hi], marker="o", capsize=4, color="#1f77b4")
        ax.set_ylim(0, 105); ax.set_xlabel(label); ax.set_ylabel("True success (%)"); ax.set_title(label)
    fig.suptitle(f"Inviluppo operativo v2 — true success vs parametri ({tag})")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, f"plot_envelope_1d_{tag}.png"), dpi=130); plt.close(fig)

    # heatmap
    for name, hm in out["heatmaps"].items():
        if hm is None:
            continue
        fig, ax = plt.subplots(figsize=(7, 6))
        rate = np.array(hm["rate"], float) * 100
        im = ax.imshow(rate, origin="lower", aspect="auto", cmap="RdYlGn", vmin=0, vmax=100,
                       extent=[hm["ex"][0], hm["ex"][-1], hm["ey"][0], hm["ey"][-1]])
        ax.set_xlabel(AXES.get(hm["kx"], hm["kx"])); ax.set_ylabel(AXES.get(hm["ky"], hm["ky"]))
        ax.set_title(f"True success (%) — {name} ({tag})")
        fig.colorbar(im, ax=ax, label="true success %")
        fig.tight_layout(); fig.savefig(os.path.join(outdir, f"plot_heatmap_{name}_{tag}.png"), dpi=130); plt.close(fig)

    print(f"Risultati in {outdir}")
    return out


def main():
    ap = argparse.ArgumentParser(description="Analisi di robustezza / inviluppo operativo v2")
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--curriculum", type=float, default=1.0)
    ap.add_argument("--run-dir", type=str, default=None)
    ap.add_argument("--stochastic", action="store_true")
    args = ap.parse_args()
    run_dir = args.run_dir or (MODEL_SPECS[1][1] if args.curriculum >= 0.5 else MODEL_SPECS[0][1])
    run(args.episodes, args.curriculum, run_dir, deterministic=not args.stochastic)


if __name__ == "__main__":
    main()