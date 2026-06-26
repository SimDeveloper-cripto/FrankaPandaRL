#!/usr/bin/env python3
# scratch/test_close_task_v1/ablation_study.py

from __future__ import annotations

import os
import json
import argparse
import numpy as np

import stats_utils as S
from ablation_envs import VARIANTS
from _common import make_cfg, make_vec_env, load_model, rollout_episode, results_dir, setup_matplotlib, json_default


def eval_variant(name, env_cls, n_episodes, curriculum, run_dir, deterministic, base_seed):
    cfg = make_cfg(run_dir=run_dir)
    venv, raw_env = make_vec_env(cfg, curriculum_level=curriculum, env_cls=env_cls)
    model = load_model(cfg, venv)
    recs = []
    for i in range(n_episodes):
        recs.append(rollout_episode(venv, model, raw_env,
                                    deterministic=deterministic, seed=base_seed + i))
    venv.close()
    succ = sum(r.success for r in recs)
    tsucc = sum(r.true_success for r in recs)
    print(f"  {name:<20} succ={succ}/{n_episodes} "
          f"true={tsucc}/{n_episodes} len_iqm={S.iqm([r.length for r in recs]):.1f}")
    return recs


def run(n_episodes, curriculum, run_dir, deterministic=True, base_seed=50_000,
        variants=None):
    variants = variants or list(VARIANTS.keys())
    if "baseline" not in variants:
        variants = ["baseline"] + variants
    print("=" * 72)
    print(f"STUDIO DI ABLAZIONE (curriculum={curriculum}, {n_episodes} ep, seed appaiati)")
    print("=" * 72)

    data = {}
    for name in variants:
        data[name] = eval_variant(name, VARIANTS[name], n_episodes, curriculum,
                                  run_dir, deterministic, base_seed)

    base   = data["baseline"]
    b_true = sum(r.true_success for r in base)
    b_len  = [r.length for r in base]

    comparisons = {}
    raw_pvals   = []
    for name in variants:
        if name == "baseline":
            continue
        v = data[name]
        v_true = sum(r.true_success for r in v)
        v_len = [r.length for r in v]
        cp = S.compare_proportions(v_true, n_episodes, b_true, n_episodes, name, "baseline")
        cl = S.compare_continuous(v_len, b_len, name, "baseline")
        comparisons[name] = dict(
            true_success=dict(variant=v_true, baseline=b_true, n=n_episodes,
                              diff=cp.diff.as_dict(), fisher_p=cp.p_value,
                              cohens_h=cp.effect_size),
            length=dict(diff=cl.diff.as_dict(), welch_p=cl.p_value,
                        cohens_d=cl.effect_size, prob_improvement=cl.prob_improvement),
        )
        raw_pvals.append(cp.p_value)

    # Holm-Bonferroni sui p-value del success (confronti multipli)
    names_cmp = [n for n in variants if n != "baseline"]
    adj = S.holm_bonferroni([comparisons[n]["true_success"]["fisher_p"] for n in names_cmp])
    for n, a in zip(names_cmp, adj):
        comparisons[n]["true_success"]["fisher_p_holm"] = a

    out = dict(curriculum=curriculum, n_episodes=n_episodes,
               mode=("det" if deterministic else "sto"),
               baseline_true_success=S.wilson_ci(b_true, n_episodes).as_dict(),
               comparisons=comparisons)

    print("\n  Confronti (true success) vs baseline:")
    print(f"    {'variante':<20} {'Δtrue%':>8} {'95% CI':>20} {'Fisher p':>10} {'p(Holm)':>9}")
    for n in names_cmp:
        d = comparisons[n]["true_success"]
        diff = d["diff"]
        print(f"    {n:<20} {diff['point']*100:>+7.1f}% "
              f"[{diff['lo']*100:>+6.1f},{diff['hi']*100:>+6.1f}] "
              f"{d['fisher_p']:>10.3g} {d['fisher_p_holm']:>9.3g}")
    print("=" * 72)

    outdir = results_dir("ablation")
    cs = f"c{int(curriculum)}"
    with open(os.path.join(outdir, f"ablation_{cs}.json"), "w") as f:
        json.dump(out, f, indent=2, default=json_default)

    # forest plot della differenza di true success
    plt = setup_matplotlib()
    fig, ax = plt.subplots(figsize=(9, 0.7 * len(names_cmp) + 2))
    ys = np.arange(len(names_cmp))
    for y, n in zip(ys, names_cmp):
        diff = comparisons[n]["true_success"]["diff"]
        ax.errorbar(diff["point"] * 100, y,
                    xerr=[[max(0.0, (diff["point"] - diff["lo"]) * 100)],
                          [max(0.0, (diff["hi"] - diff["point"]) * 100)]],
                    fmt="o", capsize=5, color="#1f77b4")
    ax.axvline(0, color="k", ls="--")
    ax.set_yticks(ys); ax.set_yticklabels(names_cmp)
    ax.set_xlabel("Δ true success vs baseline (punti %)")
    ax.set_title(f"Ablazione — effetto sul true success (curriculum {curriculum})\n"
                 "barre = CI 95% Newcombe; 0 = nessun effetto")
    ax.invert_yaxis()
    fig.tight_layout(); fig.savefig(os.path.join(outdir, f"plot_forest_{cs}.png"), dpi=130); plt.close(fig)
    print(f"Risultati in {outdir}")
    return out


def main():
    ap = argparse.ArgumentParser(description="Studio di ablazione controllato v1")
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--curriculum", type=float, default=1.0)
    ap.add_argument("--run-dir", type=str, default="runs/close_gen")
    ap.add_argument("--stochastic", action="store_true")
    ap.add_argument("--variants", nargs="*", default=None,
                    help=f"sottoinsieme di: {list(VARIANTS.keys())}")
    args = ap.parse_args()
    run(args.episodes, args.curriculum, args.run_dir,
        deterministic=not args.stochastic, variants=args.variants)


if __name__ == "__main__":
    main()