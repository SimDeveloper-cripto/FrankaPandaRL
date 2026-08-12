#!/usr/bin/env python3
# scratch/test_open_task_v2/ablation_study.py

"""
ablation_study — Confronto controllato baseline vs ablazioni degli override deterministici del RETREAT (task APERTURA v2).

Disegno APPAIATO: ogni variante e il baseline sono valutati sugli STESSI seed, quindi sulle stesse condizioni iniziali
(goal, posa, fisica). A parità di episodi questo riduce la varianza del confronto rispetto a due campioni indipendenti (Colas et al. 2018;
Patterson et al. 2024, cap. sul blocking).

Statistica:
  • true success: test esatto di Fisher + intervallo di Newcombe (1998) sulla differenza
    di proporzioni, che resta valido anche con tassi vicini a 0/1;
  • lunghezza episodio: Welch + bootstrap CI + probability of improvement (Agarwal 2021);
  • molteplicità: correzione di Holm-Bonferroni sui p-value (Colas et al. 2019) — con 7
    confronti, senza correzione ci si aspetterebbe ~1 falso positivo a caso.
  • si riportano anche clean success e allontanamento medio, perché nell'apertura un
    intervento può lasciare invariato il true success e peggiorare il ritiro (o viceversa).

Output in results/ablation/.
"""

from __future__ import annotations

import os
import json
import argparse
import numpy as np

from _common import (make_raw_env, load_obs_rms, load_model, rollout_episode,
                     results_dir, setup_matplotlib, json_default, resolve_run_dir, CURRICULUM)
from ablation_variants import VARIANTS
import stats_utils as S


def eval_variant(name, overrides, n_episodes, curriculum, run_dir, deterministic, base_seed):
    env, _cfg = make_raw_env(curriculum_level = curriculum, run_dir = run_dir, cfg_overrides = overrides)
    obs_rms   = load_obs_rms(run_dir = run_dir)
    model     = load_model(run_dir = run_dir)
    recs      = [rollout_episode(env, model, obs_rms, deterministic = deterministic, seed = base_seed + i) for i in range(n_episodes)]

    try:
        env.close()
    except Exception:
        pass
    succ  = sum(r.success for r in recs)
    tsucc = sum(r.true_success for r in recs)
    csucc = sum(r.clean_success for r in recs)
    print(f"  {name:<20} succ={succ}/{n_episodes} true={tsucc}/{n_episodes} "
          f"clean={csucc}/{n_episodes} len_iqm={S.iqm([r.length for r in recs]):.1f} "
          f"mosso_iqm={S.iqm([r.retreat_moved_max for r in recs]):.3f}m")
    return recs


def run(n_episodes, curriculum, run_dir, deterministic = True, base_seed = 50_000,
        variants = None, tag = None):
    tag      = tag or "curr1_posa_variabile"
    run_dir  = resolve_run_dir(run_dir)
    variants = list(variants) if variants else list(VARIANTS.keys())
    if "baseline" not in variants:
        variants = ["baseline"] + variants
    print("=" * 76)
    print(f"STUDIO DI ABLAZIONE — APERTURA v2 ({tag}, {n_episodes} episodi, seed appaiati)")
    print("=" * 76)

    data = {name: eval_variant(name, VARIANTS[name], n_episodes, curriculum, run_dir,
                               deterministic, base_seed) for name in variants}
    base    = data["baseline"]
    b_true  = sum(r.true_success for r in base)
    b_clean = sum(r.clean_success for r in base)
    b_len   = [r.length for r in base]
    b_moved = [r.retreat_moved_max for r in base]

    comparisons = {}
    names_cmp   = [n for n in variants if n != "baseline"]
    for name in names_cmp:
        v       = data[name]
        v_true  = sum(r.true_success for r in v)
        v_clean = sum(r.clean_success for r in v)
        v_len   = [r.length for r in v]
        v_moved = [r.retreat_moved_max for r in v]

        cp = S.compare_proportions(v_true, n_episodes, b_true, n_episodes, name, "baseline")
        cc = S.compare_proportions(v_clean, n_episodes, b_clean, n_episodes, name, "baseline")
        cl = S.compare_continuous(v_len, b_len, name, "baseline")
        cm = S.compare_continuous(v_moved, b_moved, name, "baseline")
        comparisons[name] = dict(
            true_success  = dict(variant=v_true, baseline=b_true, n=n_episodes, diff=cp.diff.as_dict(), fisher_p=cp.p_value, cohens_h=cp.effect_size),
            clean_success = dict(variant=v_clean, baseline=b_clean, n=n_episodes, diff=cc.diff.as_dict(), fisher_p=cc.p_value),
            length        = dict(diff=cl.diff.as_dict(), welch_p=cl.p_value, cohens_d=cl.effect_size, prob_improvement=cl.prob_improvement),
            retreat_moved = dict(diff=cm.diff.as_dict(), welch_p=cm.p_value, cohens_d=cm.effect_size))

    adj = S.holm_bonferroni([comparisons[n]["true_success"]["fisher_p"] for n in names_cmp])
    for n, a in zip(names_cmp, adj):
        comparisons[n]["true_success"]["fisher_p_holm"] = a
    adj_c = S.holm_bonferroni([comparisons[n]["clean_success"]["fisher_p"] for n in names_cmp])
    for n, a in zip(names_cmp, adj_c):
        comparisons[n]["clean_success"]["fisher_p_holm"] = a

    out = dict(curriculum=curriculum, n_episodes=n_episodes,
               mode = ("det" if deterministic else "sto"),
               baseline_true_success  = S.wilson_ci(b_true, n_episodes).as_dict(),
               baseline_clean_success = S.wilson_ci(b_clean, n_episodes).as_dict(),
               comparisons=comparisons
            )

    print("\n  Confronti vs baseline (true success):")
    print(f"    {'variante':<20} {'Δtrue%':>8} {'95% CI':>20} {'Fisher p':>10} {'p(Holm)':>9}")
    for n in names_cmp:
        d = comparisons[n]["true_success"]; diff = d["diff"]
        print(f"    {n:<20} {diff['point']*100:>+7.1f}% "
              f"[{diff['lo']*100:>+6.1f},{diff['hi']*100:>+6.1f}] "
              f"{d['fisher_p']:>10.3g} {d['fisher_p_holm']:>9.3g}")
    print("\n  Confronti vs baseline (clean success = ritiro davvero completato):")
    for n in names_cmp:
        d = comparisons[n]["clean_success"]; diff = d["diff"]
        print(f"    {n:<20} {diff['point']*100:>+7.1f}% "
              f"[{diff['lo']*100:>+6.1f},{diff['hi']*100:>+6.1f}] "
              f"{d['fisher_p']:>10.3g} {d['fisher_p_holm']:>9.3g}")
    print("=" * 76)

    outdir = results_dir("ablation")
    with open(os.path.join(outdir, f"ablation_{tag}.json"), "w") as f:
        json.dump(out, f, indent=2, default=json_default)

    plt = setup_matplotlib()
    fig, ax = plt.subplots(figsize = (10, 0.75 * len(names_cmp) + 2.5))
    ys = np.arange(len(names_cmp)); off = 0.16
    for y, n in zip(ys, names_cmp):
        for key, col, dy, lab in [("true_success", "#1f77b4", -off, "true success"),
                                  ("clean_success", "#9467bd", +off, "clean success")]:
            diff = comparisons[n][key]["diff"]
            ax.errorbar(diff["point"] * 100, y + dy,
                        xerr  = [[max(0.0, (diff["point"] - diff["lo"]) * 100)],
                                [max(0.0, (diff["hi"] - diff["point"]) * 100)]],
                        fmt   = "o", capsize=5, color=col,
                        label = lab if y == 0 else "")
    ax.axvline(0, color="k", ls="--"); ax.set_yticks(ys); ax.set_yticklabels(names_cmp)
    ax.invert_yaxis(); ax.legend()
    ax.set_xlabel("Δ vs baseline (punti percentuali)")
    ax.set_title(f"Ablazione degli override del RETREAT — apertura v2 ({tag})\n"
                 "barre = CI 95% Newcombe; 0 = nessun effetto")

    fig.tight_layout(); fig.savefig(os.path.join(outdir, f"plot_forest_{tag}.png"), dpi=130); plt.close(fig)
    print(f"Risultati in {outdir}")
    return out

def main():
    ap = argparse.ArgumentParser(description = "Studio di ablazione controllato — apertura v2")
    ap.add_argument("--episodes",   type = int,   default = 50)
    ap.add_argument("--curriculum", type = float, default = CURRICULUM)
    ap.add_argument("--run-dir",    type = str,   default = None)
    ap.add_argument("--stochastic", action = "store_true")
    ap.add_argument("--variants",   nargs = "*", default = None, help = f"sottoinsieme di: {list(VARIANTS.keys())}")
    args = ap.parse_args()
    run(args.episodes, args.curriculum, args.run_dir, deterministic = not args.stochastic, variants = args.variants)

if __name__ == "__main__":
    main()