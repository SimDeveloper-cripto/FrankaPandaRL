#!/usr/bin/env python3
# tests_v2/diag_phase34_v2.py

import os
import sys
import argparse
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tests_v2._common_v2 import make_raw_env, load_model, load_obs_rms, predict, phase_idx_from_info, PHASE_NAMES

RETREAT_IDX = 3
HOLD_IDX    = 2


def run(n_episodes, curriculum, run_dir):
    env, _cfg = make_raw_env(curriculum_level=curriculum, run_dir=run_dir)
    obs_rms   = load_obs_rms(run_dir=run_dir)
    model     = load_model(run_dir=run_dir)

    succ                        = 0
    true_succ                   = 0
    latch_neutral_at_transition = 0
    lengths                     = []

    obs, info = env.reset()
    for ep in range(n_episodes):
        done               = False
        steps              = 0
        phase_time         = {n: 0 for n in PHASE_NAMES}
        prev_phase         = 0
        transition_latch   = None

        while not done:
            action, _ = predict(model, obs, obs_rms, deterministic=True)
            obs, _r, terminated, truncated, info = env.step(action)

            done  = bool(terminated or truncated)
            steps += 1

            pidx = phase_idx_from_info(info)
            phase_time[PHASE_NAMES[pidx]] += 1

            # Transizione HOLD → RETREAT: primo step in cui entriamo in RETREAT
            if pidx == RETREAT_IDX and prev_phase != RETREAT_IDX and transition_latch is None:
                transition_latch = float(info.get("latch_qpos", 0.0))
            prev_phase = pidx

        is_succ = bool(info.get("is_success", False))

        dq = float(info.get("door_qpos", 0.0))
        lq = float(info.get("latch_qpos", 0.0))
        ts = is_succ and abs(dq) < 0.03 and abs(lq) < 0.08

        succ += int(is_succ)
        true_succ += int(ts)
        lengths.append(steps)
        if transition_latch is not None and abs(transition_latch) < 0.15:
            latch_neutral_at_transition += 1

        tl = "n/a" if transition_latch is None else f"{transition_latch:+.3f}"
        print(f"  ep {ep+1:>3} | len={steps:>3} | "
              f"phase_t={{R:{phase_time['REACH']},P:{phase_time['PUSH']},"
              f"H:{phase_time['HOLD']},Ret:{phase_time['RETREAT']}}} | "
              f"latch@transiz={tl} | door_end={dq:+.3f} latch_end={lq:+.3f} | "
              f"succ={is_succ} true={ts}")

        obs, info = env.reset()

    n = n_episodes
    print("\n" + "=" * 60)
    print(f"DIAGNOSTICA FASI v2 ({n} episodi, curriculum={curriculum})")
    print("=" * 60)
    print(f"Success rate (permissivo):        {succ/n*100:.1f}%")
    print(f"True success (chiusa+agganciata): {true_succ/n*100:.1f}%")
    print(f"Latch neutro alla transizione:    {latch_neutral_at_transition/n*100:.1f}%")
    print(f"Lunghezza media episodio:         {np.mean(lengths):.1f} step")
    print("=" * 60)


def main():
    ap = argparse.ArgumentParser(description="Diagnostica fasi HOLD/RETREAT v2")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--curriculum", type=float, default=1.0)
    ap.add_argument("--run-dir", type=str, default="runs/close_gen_v2")
    args = ap.parse_args()
    run(args.episodes, args.curriculum, args.run_dir)


if __name__ == "__main__":
    main()