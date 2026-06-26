#!/usr/bin/env python3
# scratch/test_close_task_v2/ablation_study.py

"""
ablation_study — Confronto controllato baseline vs ablazioni di §1.17/§1.18/§1.21 (v2).

Disegno appaiato: ogni variante e il baseline sono valutati sugli STESSI seed (stesse
condizioni iniziali) → minore varianza (Colas et al. 2018; Patterson et al. 2024).
Confronto del true success con test esatto di Fisher + CI di Newcombe; lunghezza con
Welch + bootstrap CI + probability of improvement (Agarwal et al. 2021); p-value
corretti con Holm-Bonferroni (Colas et al. 2019).

Output in results/ablation/.
"""

from __future__ import annotations

import os
import json
import argparse
import numpy as np

from _common import (make_raw_env, load_obs_rms, load_model, rollout_episode,
                     results_dir, setup_matplotlib, json_default, MODEL_SPECS)
from ablation_variants import VARIANTS
import stats_utils as S


def eval_variant(name, overrides, n_episodes, curriculum, run_dir, deterministic, base_seed):
    env, _cfg = make_raw_env(curriculum_level=curriculum, run_dir=run_dir, cfg_overrides=overrides)
    obs_rms = load_obs_rms(run_dir=run_dir)
    model = load_model(run_dir=run_dir)
    recs = [rollout_episode(env, model, obs_rms, deterministic=deterministic, seed=base_seed + i)
            for i in range(n_episodes)]
    succ = sum(r.success for r in recs); tsucc = sum(r.true_success for r in recs)
    print(f"  {name:<18} succ={succ}/{n_episodes} true={tsucc}/{n_episodes} "
          f"len_iqm={S.iqm([r.length for r in recs]):.1f}")
    return recs


def run(n_episodes, curriculum, run_dir, deterministic=True, base_seed=50_000, variants=None, tag=None):
    tag = tag or f"c{int(curriculum)}"
    variants = variants or list(VARIANTS.keys())
    if "baseline" not in variants:
        variants = ["baseline"] + variants
    print("=" * 72)
    print(f"STUDIO DI ABLAZIONE v2 ({tag}, {n_episodes} ep, seed appaiati)")
    print("=" * 72)

    data = {name: eval_variant(name, VARIANTS[name], n_episodes, curriculum, run_dir,
                               deterministic, base_seed) for name in variants}
    base = data["baseline"]
    b_true = sum(r.true_success for r in base); b_len = [r.length for r in base]

    comparisons = {}
    names_cmp = [n for n in variants if n != "baseline"]
    for name in names_cmp:
        v = data[name]; v_true = sum(r.true_success for r in v); v_len = [r.length for r in v]
        cp = S.compare_proportions(v_true, n_episodes, b_true, n_episodes, name, "baseline")
        cl = S.compare_continuous(v_len, b_len, name, "baseline")
        comparisons[name] = dict(
            true_success=dict(variant=v_true, baseline=b_true, n=n_episodes,
                              diff=cp.diff.as_dict(), fisher_p=cp.p_value, cohens_h=cp.effect_size),
            length=dict(diff=cl.diff.as_dict(), welch_p=cl.p_value,
                        cohens_d=cl.effect_size, prob_improvement=cl.prob_improvement))
    adj = S.holm_bonferroni([comparisons[n]["true_success"]["fisher_p"] for n in names_cmp])
    for n, a in zip(names_cmp, adj):
        comparisons[n]["true_success"]["fisher_p_holm"] = a

    out = dict(curriculum=curriculum, n_episodes=n_episodes, mode=("det" if deterministic else "sto"),
               baseline_true_success=S.wilson_ci(b_true, n_episodes).as_dict(), comparisons=comparisons)

    print("\n  Confronti (true success) vs baseline:")
    print(f"    {'variante':<18} {'Δtrue%':>8} {'95% CI':>20} {'Fisher p':>10} {'p(Holm)':>9}")
    for n in names_cmp:
        d = comparisons[n]["true_success"]; diff = d["diff"]
        print(f"    {n:<18} {diff['point']*100:>+7.1f}% "
              f"[{diff['lo']*100:>+6.1f},{diff['hi']*100:>+6.1f}] "
              f"{d['fisher_p']:>10.3g} {d['fisher_p_holm']:>9.3g}")
    print("=" * 72)

    outdir = results_dir("ablation")
    with open(os.path.join(outdir, f"ablation_{tag}.json"), "w") as f:
        json.dump(out, f, indent=2, default=json_default)

    plt = setup_matplotlib()
    fig, ax = plt.subplots(figsize=(9, 0.7 * len(names_cmp) + 2))
    ys = np.arange(len(names_cmp))
    for y, n in zip(ys, names_cmp):
        diff = comparisons[n]["true_success"]["diff"]
        ax.errorbar(diff["point"] * 100, y,
                    xerr=[[max(0.0, (diff["point"] - diff["lo"]) * 100)],
                          [max(0.0, (diff["hi"] - diff["point"]) * 100)]],
                    fmt="o", capsize=5, color="#1f77b4")
    ax.axvline(0, color="k", ls="--"); ax.set_yticks(ys); ax.set_yticklabels(names_cmp); ax.invert_yaxis()
    ax.set_xlabel("Δ true success vs baseline (punti %)")
    ax.set_title(f"Ablazione v2 §1.17/§1.18/§1.21 — effetto sul true success ({tag})\n"
                 "barre = CI 95% Newcombe; 0 = nessun effetto")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, f"plot_forest_{tag}.png"), dpi=130); plt.close(fig)
    print(f"Risultati in {outdir}")
    return out


def main():
    ap = argparse.ArgumentParser(description="Studio di ablazione controllato v2")
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--curriculum", type=float, default=1.0)
    ap.add_argument("--run-dir", type=str, default=None)
    ap.add_argument("--stochastic", action="store_true")
    ap.add_argument("--variants", nargs="*", default=None, help=f"sottoinsieme di: {list(VARIANTS.keys())}")
    args = ap.parse_args()
    run_dir = args.run_dir or (MODEL_SPECS[1][1] if args.curriculum >= 0.5 else MODEL_SPECS[0][1])
    run(args.episodes, args.curriculum, run_dir, deterministic=not args.stochastic, variants=args.variants)


if __name__ == "__main__":
    main()