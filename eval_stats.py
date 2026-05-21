#!/usr/bin/env python3
# eval_stats.py

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.append(os.getcwd())
from config.train_close_config import TrainConfig
from close_generalized.env_gen import GeneralizedDoorEnv

os.environ['MPLCONFIGDIR'] = os.path.join(os.getcwd(), 'scratch')

def make_env(cfg):
    def _init():
        env = GeneralizedDoorEnv(cfg, render_mode=None)
        env.set_curriculum_level(1.0)
        return env
    return _init

def classify_failure(max_phase_idx, dist_handle, door_angle, latch_qpos, step_count, is_success):
    if is_success:
        return "SUCCESS"

    if max_phase_idx == 1:
        return "REACH timeout"
    elif max_phase_idx == 2:
        if dist_handle > 0.08:
            return "GRASP lost"
        else:
            return "PUSH timeout"
    elif max_phase_idx == 3:
        return "HOLD bounce / timeout"
    elif max_phase_idx == 4:
        if door_angle >= 0.03:
            return "RETREAT door bounce"
        elif abs(latch_qpos) >= 0.08:
            return "RETREAT latch not neutral"
        else:
            return "RETREAT timeout"
    return "Unknown"

def run_eval(n_episodes=50, deterministic=True):
    cfg  = TrainConfig(run_dir="runs/close_gen", num_envs=1, horizon=500)
    venv = DummyVecEnv([make_env(cfg)])

    vn_path = os.path.join(cfg.run_dir, "vecnormalize.pkl")
    if os.path.exists(vn_path):
        venv             = VecNormalize.load(vn_path, venv)
        venv.training    = False
        venv.norm_reward = False

    model_path = os.path.join(cfg.run_dir, "best_model.zip")
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return None

    model = SAC.load(model_path, env=venv)
    model.policy.set_training_mode(False)

    stats = {
        'success'         : [],
        'lengths'         : [],
        'max_phases'      : [],
        'phase_times'     : [],
        'min_door_angles' : [],
        'latch_neutral'   : [],
        'door_closed'     : [],
        'failures'        : []
    }

    obs      = venv.reset()
    ep_count = 0
    raw_env  = venv.envs[0]

    phase_time    = {'1:REACH': 0, '2:PUSH': 0, '3:HOLD': 0, '4:BACK': 0}
    max_phase_idx = 1
    step_count    = 0
    min_door_angle = np.inf

    while ep_count < n_episodes:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, rewards, dones, infos = venv.step(action)

        latch_qpos     = float(raw_env._rs_env.sim.data.qpos[raw_env._rs_env.handle_qpos_addr])
        door_qpos      = float(raw_env._rs_env.sim.data.qpos[raw_env._rs_env.hinge_qpos_addr])
        door_angle     = abs(door_qpos)
        min_door_angle = min(min_door_angle, door_angle)

        eef_site_id = raw_env._rs_env.robots[0].eef_site_id
        if isinstance(eef_site_id, dict):
            site_id = eef_site_id.get('right', list(eef_site_id.values())[0])
        else:
            site_id = eef_site_id
        eef_pos     = raw_env._rs_env.sim.data.site_xpos[site_id]
        handle_pos  = raw_env._rs_env.sim.data.geom_xpos[raw_env.handle_geom_id] if getattr(raw_env, "handle_geom_id", None) is not None else eef_pos
        dist_handle = float(np.linalg.norm(eef_pos - handle_pos))

        if raw_env._success_latched:
            phase = "4:BACK" if getattr(raw_env, "_ready_to_retreat", False) else "3:HOLD"
            pidx  = 4 if getattr(raw_env, "_ready_to_retreat", False) else 3
        elif raw_env._grasp_phase:
            phase = "2:PUSH"
            pidx  = 2
        else:
            phase = "1:REACH"
            pidx  = 1

        phase_time[phase] += 1
        max_phase_idx     = max(max_phase_idx, pidx)
        step_count        += 1

        if dones[0]:
            is_success      = infos[0].get("is_success", False)
            term_latch_qpos = infos[0].get("latch_qpos", latch_qpos)
            term_door_qpos  = infos[0].get("door_qpos", door_qpos)
            
            latch_neutral_end = abs(term_latch_qpos) < 0.08
            door_closed_end   = abs(term_door_qpos) < 0.03

            failure_type = classify_failure(
                max_phase_idx = max_phase_idx,
                dist_handle   = dist_handle,
                door_angle    = abs(term_door_qpos),
                latch_qpos    = term_latch_qpos,
                step_count    = step_count,
                is_success    = is_success
            )

            stats['success'].append(int(is_success))
            stats['lengths'].append(step_count)
            stats['min_door_angles'].append(min_door_angle)
            stats['latch_neutral'].append(int(latch_neutral_end))
            stats['door_closed'].append(int(door_closed_end))
            stats['failures'].append(failure_type)

            phases = ["1:REACH", "2:PUSH", "3:HOLD", "4:BACK"]
            stats['max_phases'].append(phases[max_phase_idx - 1])
            stats['phase_times'].append(dict(phase_time))

            print(f"Episode {ep_count+1}/{n_episodes} - Success: {is_success} - Fail Mode: {failure_type} - Length: {step_count}")

            ep_count       += 1
            step_count     = 0
            min_door_angle = np.inf
            phase_time     = {'1:REACH': 0, '2:PUSH': 0, '3:HOLD': 0, '4:BACK': 0}
            max_phase_idx  = 1
            obs            = venv.reset()

    return stats

def print_summary(stats, mode_name):
    if stats is None:
        return

    n_episodes         = len(stats['success'])
    success_rate       = np.mean(stats['success']) * 100
    avg_length         = np.mean(stats['lengths'])
    mean_min_door      = np.mean(stats['min_door_angles'])
    std_min_door       = np.std(stats['min_door_angles'])
    latch_neutral_rate = np.mean(stats['latch_neutral']) * 100
    door_closed_rate   = np.mean(stats['door_closed']) * 100

    fail_counts = Counter(stats['failures'])

    print("\n" + "="*70)
    print(f"EVALUATION STATISTICS SUMMARY ({n_episodes} episodes)")
    print("="*70)
    print(f"Mode:                         {mode_name}")
    print(f"Success Rate:                 {success_rate:.1f}%")
    print(f"Average Episode Length:       {avg_length:.1f} steps")
    print(f"Min Door Angle Achieved:      {mean_min_door:.4f} ± {std_min_door:.4f} rad")
    print(f"Latch Neutral at End:         {latch_neutral_rate:.1f}%")
    print(f"Door Closed at End:           {door_closed_rate:.1f}%")
    print("-"*70)
    print("Failure Mode Breakdown:")

    failure_types = [
        "SUCCESS",
        "REACH timeout",
        "GRASP lost",
        "PUSH timeout",
        "HOLD bounce / timeout",
        "RETREAT door bounce",
        "RETREAT latch not neutral",
        "RETREAT timeout"
    ]
    for ft in failure_types:
        count = fail_counts[ft]
        pct = (count / n_episodes) * 100
        print(f"- {ft:<27}: {count:>3} ({pct:.1f}%)")
    print("="*70 + "\n")

def create_plots(train_stats, eval_stats):
    os.makedirs('scratch/plots', exist_ok=True)

    # 1. Success Rate Bar Chart
    train_sr = np.mean(train_stats['success']) * 100
    eval_sr  = np.mean(eval_stats['success']) * 100

    plt.figure(figsize=(8, 6))
    bars = plt.bar(['Train (Stochastic)', 'Eval (Deterministic)'], [train_sr, eval_sr], color=['#1f77b4', '#2ca02c'])
    plt.title('Success Rate (%)')
    plt.ylim(0, 105)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.1f}%', ha='center', va='bottom', fontweight='bold')
    plt.savefig('scratch/plots/success_rate.png')
    plt.close()

    # 2. Phase Breakdown (Stacked Bar)
    phases = ["1:REACH", "2:PUSH", "3:HOLD", "4:BACK"]
    def get_phase_counts(stats):
        counts = {p: 0 for p in phases}
        for mp in stats['max_phases']:
            counts[mp] += 1
        return [counts[p] for p in phases]

    train_pc = get_phase_counts(train_stats)
    eval_pc  = get_phase_counts(eval_stats)

    fig, ax      = plt.subplots(figsize=(10, 6))
    bottom_train = 0
    bottom_eval  = 0
    colors       = ['#d62728', '#ff7f0e', '#1f77b4', '#2ca02c']

    for i, phase in enumerate(phases):
        ax.bar('Train', train_pc[i], bottom=bottom_train, label=phase if bottom_train==0 else "", color=colors[i])
        ax.bar('Eval', eval_pc[i], bottom=bottom_eval, color=colors[i])

        if train_pc[i] > 0:
            ax.text(0, bottom_train + train_pc[i]/2, f'{train_pc[i]}', ha='center', va='center', color='white', fontweight='bold')
        if eval_pc[i] > 0:
            ax.text(1, bottom_eval + eval_pc[i]/2, f'{eval_pc[i]}', ha='center', va='center', color='white', fontweight='bold')

        bottom_train += train_pc[i]
        bottom_eval  += eval_pc[i]

    ax.set_title('Max Phase Reached Distribution')
    ax.legend(phases)
    plt.savefig('scratch/plots/max_phase_dist.png')
    plt.close()

    # 3. Average Time in Each Phase
    def get_avg_phase_times(stats):
        totals = {p: [] for p in phases}
        for pt in stats['phase_times']:
            for p in phases:
                totals[p].append(pt[p])
        return [np.mean(totals[p]) for p in phases]

    train_times = get_avg_phase_times(train_stats)
    eval_times  = get_avg_phase_times(eval_stats)

    x     = np.arange(len(phases))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1  = ax.bar(x - width/2, train_times, width, label='Train', color='#1f77b4')
    rects2  = ax.bar(x + width/2, eval_times, width, label='Eval', color='#2ca02c')

    ax.set_ylabel('Average Steps')
    ax.set_title('Average Time Spent in Each Phase')
    ax.set_xticks(x)
    ax.set_xticklabels(phases)
    ax.legend()

    for rects in [rects1, rects2]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}',
                        xy             = (rect.get_x() + rect.get_width() / 2, height),
                        xytext         = (0, 3),
                        textcoords     = "offset points",
                        ha             = 'center',
                        va             = 'bottom')

    plt.savefig('scratch/plots/avg_phase_time.png')
    plt.close()

if __name__ == "__main__":
    os.makedirs('scratch', exist_ok=True)

    print("Evaluating EVAL (Deterministic)...")
    eval_stats = run_eval(n_episodes=50, deterministic=True)

    print("\nEvaluating TRAIN (Stochastic)...")
    train_stats = run_eval(n_episodes=50, deterministic=False)

    import pickle
    with open('scratch/eval_stats.pkl', 'wb') as f:
        pickle.dump({'eval': eval_stats, 'train': train_stats}, f)

    print("Stats computed and saved. Generating summaries...")
    print_summary(eval_stats, "Eval (Deterministic)")
    print_summary(train_stats, "Train (Stochastic)")
    
    print("Generating plots...")
    create_plots(train_stats, eval_stats)
    print("Done!")