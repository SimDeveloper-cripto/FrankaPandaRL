#!/usr/bin/env python3
# tests_v2/eval_stats_v2.py

import os
import sys
import argparse
import numpy as np
from collections import Counter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tests_v2._common_v2 import make_raw_env, load_model, load_obs_rms, predict, phase_idx_from_info, PHASE_NAMES


def classify(max_phase_idx, is_success, term_door_qpos, term_latch_qpos):
    if is_success:
        return "SUCCESS"
    if max_phase_idx == 0:
        return "REACH timeout"
    if max_phase_idx == 1:
        return "PUSH timeout / grasp lost"
    if max_phase_idx == 2:
        return "HOLD bounce / timeout"
    if max_phase_idx == 3:
        if abs(term_door_qpos) >= 0.03:
            return "RETREAT door bounce"
        if abs(term_latch_qpos) >= 0.08:
            return "RETREAT latch not neutral"
        return "RETREAT timeout"
    return "Unknown"


def run_eval(n_episodes, curriculum, deterministic, run_dir):
    env, _cfg = make_raw_env(curriculum_level=curriculum, run_dir=run_dir)
    obs_rms   = load_obs_rms(run_dir=run_dir)
    model     = load_model(run_dir=run_dir)

    stats = {"success": [], "true_success": [], "lengths": [], "max_phases": [], "failures": [], "min_door": []}

    obs, info = env.reset()

    ep        = 0
    steps     = 0
    max_phase = 0
    min_door  = np.inf

    while ep < n_episodes:
        action, _ = predict(model, obs, obs_rms, deterministic=deterministic)
        obs, _r, terminated, truncated, info = env.step(action)

        done        = bool(terminated or truncated)
        steps       += 1
        max_phase   = max(max_phase, phase_idx_from_info(info))
        min_door    = min(min_door, abs(float(info.get("door_qpos", info.get("door_angle", 0.0)))))

        if done:
            is_succ = bool(info.get("is_success", False))
            dq      = float(info.get("door_qpos", 0.0))
            lq      = float(info.get("latch_qpos", 0.0))

            true_succ = int(is_succ and abs(dq) < 0.03 and abs(lq) < 0.08)

            stats["success"].append(int(is_succ))
            stats["true_success"].append(true_succ)
            stats["lengths"].append(steps)
            stats["max_phases"].append(PHASE_NAMES[max_phase])
            stats["min_door"].append(min_door)
            stats["failures"].append(classify(max_phase, is_succ, dq, lq))

            print(f"  ep {ep+1:>3}/{n_episodes} | succ={is_succ} | true={bool(true_succ)} "
                  f"| maxphase={PHASE_NAMES[max_phase]} | len={steps}")
            ep += 1

            steps     = 0
            max_phase = 0
            min_door  = np.inf
            obs, info = env.reset()
    return stats


def summary(stats, title):
    n = len(stats["success"])
    print("\n" + "=" * 64)
    print(f"RIEPILOGO — {title} ({n} episodi)")
    print("=" * 64)
    print(f"Success rate (permissivo, phase∈HOLD/RETREAT): {np.mean(stats['success']) * 100:.1f}%")
    print(f"True success (porta chiusa + latch neutro):    {np.mean(stats['true_success']) * 100:.1f}%")
    print(f"Lunghezza media episodio:                      {np.mean(stats['lengths']):.1f} step")
    print(f"Min door-angle medio:                          {np.mean(stats['min_door']):.4f} rad")
    print("-" * 64)
    print("Breakdown fallimenti:")
    for ft, c in Counter(stats["failures"]).most_common():
        print(f"  - {ft:<28}: {c:>3} ({c/n*100:.1f}%)")
    print("=" * 64 + "\n")


def main():
    ap = argparse.ArgumentParser(description = "Eval stats v2")
    ap.add_argument("--episodes",   type = int,   default = 50)
    ap.add_argument("--curriculum", type = float, default = 1.0)
    ap.add_argument("--run-dir",    type = str,   default = "runs/close_gen_v2")
    args = ap.parse_args()

    print("Valutazione DETERMINISTICA...")
    det = run_eval(args.episodes, args.curriculum, True, args.run_dir)
    summary(det, "Eval (deterministico)")

    print("Valutazione STOCASTICA...")
    sto = run_eval(args.episodes, args.curriculum, False, args.run_dir)
    summary(sto, "Train (stocastico)")


if __name__ == "__main__":
    main()