#!/usr/bin/env python3
# scratch/test_open_task_v2/robustness_analysis.py

"""
robustness_analysis — Inviluppo operativo della policy di APERTURA v2.

Risponde a «DOVE» generalizza la policy, non con un numero solo ma con curve
esito-vs-parametro (Tobin et al. 2017; Mehta et al. 2020 — la randomization va
valutata per regioni, non in media; Zhao et al. 2020 §3.4; stile delle curve di
ten Pas et al. 2017, Fig. 5).

Metodo: STRATIFICAZIONE.
Si raccolgono molti episodi con la randomization naturale dell'ambiente, si registra il parametro REALIZZATO e l'esito (true success), si
raggruppa in bin per quantili e si calcola il tasso con intervallo di Wilson per bin.
Rispetto a forzare un parametro per volta, questo evita di uscire dalla distribuzione su cui la policy
è stata addestrata (che falserebbe l'esito) e costa un solo giro.

Assi (7):
i cinque fisici comuni alla chiusura,
la distanza della porta e SPECIFICO DELL'APERTURA il `goal_angle` campionato a ogni episodio:
è la difficoltà del task stesso (quanto in là bisogna aprire), e senza di esso
la "generalizzazione al goal" resterebbe non misurata.

Heatmap 2D (3):
    frizione×raggio            (presa),
    rigidità latch×massa porta (dinamica del ritiro),
    goal×rigidità latch        (quanto in là si apre contro quanto forte richiude).
"""

from __future__ import annotations

import os
import json
import argparse
import numpy as np

from _common import (make_raw_env, load_obs_rms, load_model, rollout_episode,
                    results_dir, setup_matplotlib, json_default, resolve_run_dir, CURRICULUM)
import stats_utils as S

AXES = {
    "handle_friction":          "Frizione maniglia",
    "handle_radius":            "Raggio maniglia (m)",
    "latch_stiffness_ratio":    "Rigidità latch (×base)",
    "hinge_damping_ratio":      "Smorzamento cerniera (×base)",
    "door_mass_ratio":          "Massa porta (×base)",
    "goal_angle":               "Goal di apertura (rad)",
    "door_x":                   "Distanza porta (x, m)",
}

MIN_CELL_N = 10
HEATMAPS   = [
        ("friction_radius", "handle_friction",       "handle_radius"),
        ("latch_mass",      "latch_stiffness_ratio", "door_mass_ratio"),
        ("goal_latch",      "goal_angle",            "latch_stiffness_ratio"),
]

def collect(n_episodes, curriculum, run_dir, deterministic, base_seed=40_000):
    env, _cfg = make_raw_env(curriculum_level=curriculum, run_dir=run_dir)
    obs_rms   = load_obs_rms(run_dir=run_dir)
    model     = load_model(run_dir=run_dir)
    recs      = []
    for i in range(n_episodes):
        recs.append(rollout_episode(env, model, obs_rms, deterministic = deterministic, seed = base_seed + i))
        if (i + 1) % 25 == 0:
            print(f"  raccolti {i+1}/{n_episodes} episodi")
    try:
        env.close()
    except Exception:
        pass
    return recs


def stratify_1d(recs, key, n_bins=6):
    vals   = np.array([getattr(r, key) for r in recs], float)
    out    = np.array([1 if r.true_success else 0 for r in recs], int)
    finite = np.isfinite(vals); vals, out = vals[finite], out[finite]
    if len(vals) == 0:
        return []

    edges = np.unique(np.quantile(vals, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 2:
        return []

    bins = []
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        last = (i == len(edges) - 2)
        mask = (vals >= lo) & ((vals <= hi) if last else (vals < hi))
        n    = int(mask.sum())
        if n == 0:
            continue
        s = int(out[mask].sum()); ci = S.wilson_ci(s, n)
        bins.append(dict(lo=float(lo), hi=float(hi), center=float((lo + hi) / 2),
                        n=n, successes=s, rate=ci.point, ci_lo=ci.lo, ci_hi=ci.hi))
    return bins

def heatmap_2d(recs, kx, ky, nb=3):
    vx = np.array([getattr(r, kx) for r in recs], float)
    vy = np.array([getattr(r, ky) for r in recs], float)
    oc = np.array([1 if r.true_success else 0 for r in recs], int)
    ok = np.isfinite(vx) & np.isfinite(vy); vx, vy, oc = vx[ok], vy[ok], oc[ok]
    if len(vx) == 0:
        return None

    ex = np.unique(np.quantile(vx, np.linspace(0, 1, nb + 1)))
    ey = np.unique(np.quantile(vy, np.linspace(0, 1, nb + 1)))
    if len(ex) < 2 or len(ey) < 2:
        return None
    rate = np.full((len(ey) - 1, len(ex) - 1), np.nan); count = np.zeros_like(rate)
    for j in range(len(ey) - 1):
        for i in range(len(ex) - 1):
            mx = (vx >= ex[i]) & ((vx <= ex[i + 1]) if i == len(ex) - 2 else (vx < ex[i + 1]))
            my = (vy >= ey[j]) & ((vy <= ey[j + 1]) if j == len(ey) - 2 else (vy < ey[j + 1]))
            m = mx & my
            if m.sum() > 0:
                rate[j, i] = oc[m].mean(); count[j, i] = m.sum()
    return dict(ex = ex.tolist(), ey = ey.tolist(), rate = rate.tolist(), count = count.tolist(),
                kx = kx, ky = ky)

def run(n_episodes, curriculum, run_dir, deterministic = True, tag = None):
    tag     = tag or "curr1_posa_variabile"
    run_dir = resolve_run_dir(run_dir)
    print("=" * 76)
    print(f"ROBUSTEZZA / INVILUPPO OPERATIVO — APERTURA v2 ({tag}, {n_episodes} episodi)")
    print("=" * 76)
    recs = collect(n_episodes, curriculum, run_dir, deterministic)

    n = len(recs)
    out = dict(curriculum = curriculum, n_episodes = n,
               mode = ("det" if deterministic else "sto"),
               overall_true_success  = S.wilson_ci(sum(r.true_success for r in recs), n).as_dict(),
               overall_clean_success = S.wilson_ci(sum(r.clean_success for r in recs), n).as_dict(),
               envelopes = {})

    for key, label in AXES.items():
        bins = stratify_1d(recs, key); out["envelopes"][key] = bins
        if bins:
            print(f"\n  {label}:")
            for b in bins:
                print(f"    [{b['lo']:.3f},{b['hi']:.3f}]  n={b['n']:>3}  "
                      f"true={b['rate']*100:5.1f}% [{b['ci_lo']*100:4.1f},{b['ci_hi']*100:4.1f}]")

    out["heatmaps"] = {name: heatmap_2d(recs, kx, ky) for name, kx, ky in HEATMAPS}
    print(f"\n  True success complessivo : {S.wilson_ci(sum(r.true_success for r in recs), n)}")
    print(f"  Clean success complessivo: {S.wilson_ci(sum(r.clean_success for r in recs), n)}")
    print("=" * 76)

    outdir = results_dir("robustness")
    with open(os.path.join(outdir, f"robustness_{tag}.json"), "w") as f:
        json.dump(out, f, indent=2, default=json_default)

    plt      = setup_matplotlib()
    fig, axs = plt.subplots(2, 4, figsize = (20, 8)); axs = axs.ravel()
    for ax, (key, label) in zip(axs, AXES.items()):
        bins = out["envelopes"][key]
        if not bins:
            ax.set_title(f"{label}\n(dati insufficienti)"); continue
        c  = [b["center"] for b in bins]; r = [b["rate"] * 100 for b in bins]
        lo = [max(0.0, (b["rate"] - b["ci_lo"]) * 100) for b in bins]
        hi = [max(0.0, (b["ci_hi"] - b["rate"]) * 100) for b in bins]

        ax.errorbar(c, r, yerr = [lo, hi], marker = "o", capsize = 4, color = "#1f77b4")
        ax.set_ylim(0, 105); ax.set_xlabel(label); ax.set_ylabel("True success (%)")
        ax.set_title(label, fontsize=10)
    for ax in axs[len(AXES):]:
        ax.axis("off")

    fig.suptitle(f"Inviluppo operativo apertura v2 — true success vs parametri ({tag})")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, f"plot_envelope_1d_{tag}.png"), dpi=130); plt.close(fig)

    # ── Heatmap 2D ──────────────────────────────────────────────────────────
    for name, hm in out["heatmaps"].items():
        if hm is None:
            continue
        fig, ax = plt.subplots(figsize = (7.5, 6))
        rate    = np.array(hm["rate"], float) * 100
        cnt     = np.array(hm["count"], float)
        shown   = np.where(cnt >= MIN_CELL_N, rate, np.nan)
        cmap    = plt.get_cmap("RdYlGn").copy(); cmap.set_bad("#f0f0f0")
        im      = ax.imshow(shown, origin="lower", aspect="auto", cmap=cmap, vmin=0, vmax=100,
                    extent=[hm["ex"][0], hm["ex"][-1], hm["ey"][0], hm["ey"][-1]])

        ex, ey = np.array(hm["ex"], float), np.array(hm["ey"], float)
        xc     = (ex[:-1] + ex[1:]) / 2.0
        yc     = (ey[:-1] + ey[1:]) / 2.0
        n_low  = 0
        for j, y in enumerate(yc):
            for i, x in enumerate(xc):
                n_ij = int(cnt[j, i])
                if n_ij == 0:
                    continue
                if n_ij < MIN_CELL_N:
                    n_low += 1
                    ax.text(x, y, f"n={n_ij}", ha="center", va="center",
                            fontsize=8, color="#888888", style="italic")
                else:
                    ax.text(x, y, f"{rate[j, i]:.0f}%\nn={n_ij}", ha="center", va="center",
                            fontsize=8, color="black")

        ax.set_xlabel(AXES.get(hm["kx"], hm["kx"])); ax.set_ylabel(AXES.get(hm["ky"], hm["ky"]))
        ax.set_title(f"True success (%) — {name} ({tag})\n"
                     f"celle con n < {MIN_CELL_N} in grigio: non interpretabili "
                     f"({n_low} su {int((cnt > 0).sum())})", fontsize=10)
        fig.colorbar(im, ax=ax, label="true success %")
        fig.tight_layout(); fig.savefig(os.path.join(outdir, f"plot_heatmap_{name}_{tag}.png"), dpi = 130); plt.close(fig)

    print(f"Risultati in {outdir}")
    return out

def main():
    ap = argparse.ArgumentParser(description = "Analisi di robustezza / inviluppo operativo — apertura v2")
    ap.add_argument("--episodes",   type = int,   default = 300)
    ap.add_argument("--curriculum", type = float, default = CURRICULUM)
    ap.add_argument("--run-dir",    type = str,   default = None)
    ap.add_argument("--stochastic", action = "store_true")
    args = ap.parse_args()
    run(args.episodes, args.curriculum, args.run_dir, deterministic = not args.stochastic)

if __name__ == "__main__":
    main()