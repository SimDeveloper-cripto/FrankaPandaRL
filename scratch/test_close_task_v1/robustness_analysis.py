#!/usr/bin/env python3
# scratch/test_close_task_v1/robustness_analysis.py

"""
robustness_analysis — Inviluppo operativo della policy (task di chiusura v1).

NOVITÀ rispetto alla suite originale. Risponde direttamente a 08_risultati_v2 §2/§4:
"è dimostrata la generalizzazione FISICA … non ancora quella alla POSA". Un singolo
success rate aggregato non basta a *caratterizzare* la generalizzazione: bisogna
misurare come il successo varia lungo gli assi di domain randomization e individuare
il "bordo di competenza" (Tobin et al. 2017; Mehta et al. 2020 — Active DR; Zhao et al.
2017 — sim-to-real). È lo stesso stile della Fig. 5 di ten Pas et al. 2017 (errore vs
posizione).

Metodo: STRATIFICAZIONE (robusto, nessun hack degli interni).
  1. esegue molti episodi con la randomizzazione naturale dell'env;
  2. registra per ciascuno il parametro REALIZZATO (frizione, raggio, x della porta)
     e l'esito (true_success);
  3. raggruppa in bin e calcola success ± intervallo di Wilson per bin → curva di
     inviluppo; + heatmap 2D frizione×raggio.

Questo evita di riscrivere a mano la fisica nel reset (fragile) e dà una stima onesta
per regione dello spazio dei parametri. Servono abbastanza episodi: con bin da ~25–40
episodi i CI restano informativi (power analysis — Colas et al. 2018).

Output (in results/robustness/): robustness_<curr>.json + curve e heatmap.
"""

from __future__ import annotations

import os
import json
import argparse
import numpy as np

from _common import make_cfg, make_vec_env, load_model, rollout_episode, results_dir, setup_matplotlib, json_default
import stats_utils as S


def collect(n_episodes: int, curriculum: float, run_dir: str, deterministic: bool,
            base_seed: int = 40_000):
    cfg = make_cfg(run_dir=run_dir)
    venv, raw_env = make_vec_env(cfg, curriculum_level=curriculum)
    model = load_model(cfg, venv)
    recs = []
    for i in range(n_episodes):
        rec = rollout_episode(venv, model, raw_env, deterministic=deterministic,
                              seed=base_seed + i)
        recs.append(rec)
        if (i + 1) % 25 == 0:
            print(f"  raccolti {i+1}/{n_episodes} episodi")
    venv.close()
    return recs


def stratify_1d(recs, key: str, n_bins: int = 6):
    """Success rate (true) ± Wilson per bin equi-frequenza del parametro `key`."""
    vals = np.array([getattr(r, key) for r in recs], float)
    outcomes = np.array([1 if r.true_success else 0 for r in recs], int)
    finite = np.isfinite(vals)
    vals, outcomes = vals[finite], outcomes[finite]
    if len(vals) == 0:
        return []
    edges = np.quantile(vals, np.linspace(0, 1, n_bins + 1))
    edges = np.unique(edges)
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (vals >= lo) & (vals <= hi if hi == edges[-1] else vals < hi)
        n = int(mask.sum())
        if n == 0:
            continue
        s = int(outcomes[mask].sum())
        ci = S.wilson_ci(s, n)
        bins.append(dict(lo=float(lo), hi=float(hi), center=float((lo + hi) / 2),
                         n=n, successes=s, rate=ci.point, ci_lo=ci.lo, ci_hi=ci.hi))
    return bins


def heatmap_2d(recs, kx="handle_friction", ky="handle_radius", nb=4):
    vx = np.array([getattr(r, kx) for r in recs], float)
    vy = np.array([getattr(r, ky) for r in recs], float)
    oc = np.array([1 if r.true_success else 0 for r in recs], int)
    ok = np.isfinite(vx) & np.isfinite(vy)
    vx, vy, oc = vx[ok], vy[ok], oc[ok]
    if len(vx) == 0:
        return None
    ex = np.unique(np.quantile(vx, np.linspace(0, 1, nb + 1)))
    ey = np.unique(np.quantile(vy, np.linspace(0, 1, nb + 1)))
    rate = np.full((len(ey) - 1, len(ex) - 1), np.nan)
    count = np.zeros_like(rate)
    for j in range(len(ey) - 1):
        for i in range(len(ex) - 1):
            mx = (vx >= ex[i]) & (vx <= ex[i + 1] if i == len(ex) - 2 else vx < ex[i + 1])
            my = (vy >= ey[j]) & (vy <= ey[j + 1] if j == len(ey) - 2 else vy < ey[j + 1])
            m = mx & my
            if m.sum() > 0:
                rate[j, i] = oc[m].mean()
                count[j, i] = m.sum()
    return dict(ex=ex.tolist(), ey=ey.tolist(), rate=rate.tolist(), count=count.tolist())


def run(n_episodes: int, curriculum: float, run_dir: str, deterministic: bool = True):
    print("=" * 72)
    print(f"ROBUSTEZZA / INVILUPPO OPERATIVO (curriculum={curriculum}, {n_episodes} ep)")
    print("=" * 72)
    recs = collect(n_episodes, curriculum, run_dir, deterministic)

    out = dict(curriculum=curriculum, n_episodes=n_episodes,
               mode=("det" if deterministic else "sto"),
               overall_true_success=S.wilson_ci(
                   sum(r.true_success for r in recs), len(recs)).as_dict())
    axes = {"handle_friction": "Frizione maniglia",
            "handle_radius": "Raggio maniglia (m)",
            "door_x": "Distanza porta (x, m)"}
    out["envelopes"] = {}
    for key, label in axes.items():
        bins = stratify_1d(recs, key)
        out["envelopes"][key] = bins
        if bins:
            print(f"\n  {label}:")
            for b in bins:
                print(f"    [{b['lo']:.3f},{b['hi']:.3f}]  n={b['n']:>3}  "
                      f"true_succ={b['rate']*100:5.1f}% "
                      f"[{b['ci_lo']*100:4.1f},{b['ci_hi']*100:4.1f}]")
    out["heatmap_friction_radius"] = heatmap_2d(recs)
    print(f"\n  True success complessivo: {S.wilson_ci(sum(r.true_success for r in recs), len(recs))}")
    print("=" * 72)

    outdir = results_dir("robustness")
    cs = f"c{int(curriculum)}"
    with open(os.path.join(outdir, f"robustness_{cs}.json"), "w") as f:
        json.dump(out, f, indent=2, default=json_default)

    plt = setup_matplotlib()
    # curve 1D
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (key, label) in zip(axs, axes.items()):
        bins = out["envelopes"][key]
        if not bins:
            ax.set_title(f"{label}\n(no data)"); continue
        c = [b["center"] for b in bins]
        r = [b["rate"] * 100 for b in bins]
        lo = [max(0.0, (b["rate"] - b["ci_lo"]) * 100) for b in bins]
        hi = [max(0.0, (b["ci_hi"] - b["rate"]) * 100) for b in bins]
        ax.errorbar(c, r, yerr=[lo, hi], marker="o", capsize=4, color="#1f77b4")
        ax.set_ylim(0, 105); ax.set_xlabel(label); ax.set_ylabel("True success (%)")
        ax.set_title(label)
    fig.suptitle(f"Inviluppo operativo — true success vs parametri (curriculum {curriculum})")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, f"plot_envelope_1d_{cs}.png"), dpi=130); plt.close(fig)

    # heatmap 2D
    hm = out["heatmap_friction_radius"]
    if hm is not None:
        fig, ax = plt.subplots(figsize=(7, 6))
        rate = np.array(hm["rate"], float) * 100
        im = ax.imshow(rate, origin="lower", aspect="auto", cmap="RdYlGn",
                       vmin=0, vmax=100,
                       extent=[hm["ex"][0], hm["ex"][-1], hm["ey"][0], hm["ey"][-1]])
        ax.set_xlabel("Frizione maniglia"); ax.set_ylabel("Raggio maniglia (m)")
        ax.set_title(f"True success (%) — frizione × raggio (curriculum {curriculum})")
        fig.colorbar(im, ax=ax, label="true success %")
        fig.tight_layout(); fig.savefig(os.path.join(outdir, f"plot_heatmap_{cs}.png"), dpi=130); plt.close(fig)

    print(f"Risultati in {outdir}")
    return out


def main():
    ap = argparse.ArgumentParser(description="Analisi di robustezza / inviluppo operativo v1")
    ap.add_argument("--episodes", type=int, default=300,
                    help="molti episodi: vanno divisi in bin (default 300)")
    ap.add_argument("--curriculum", type=float, default=1.0)
    ap.add_argument("--run-dir", type=str, default="runs/close_gen")
    ap.add_argument("--stochastic", action="store_true")
    args = ap.parse_args()
    run(args.episodes, args.curriculum, args.run_dir, deterministic=not args.stochastic)


if __name__ == "__main__":
    main()