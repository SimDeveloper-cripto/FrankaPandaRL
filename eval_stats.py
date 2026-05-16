# TESTS
# 50 DETERMINISTIC EVAL  RUNS
# 50 STOCHASTIC    TRAIN RUNS

# Note: 14_05_2026 model almost complete (tests after call)

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

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
        'success'    : [],
        'lengths'    : [],
        'max_phases' : [],
        'phase_times': []
    }

    obs      = venv.reset()
    ep_count = 0
    raw_env  = venv.envs[0]

    phase_time    = {'1:REACH': 0, '2:PUSH': 0, '3:HOLD': 0, '4:BACK': 0}
    max_phase_idx = 1
    step_count    = 0

    while ep_count < n_episodes:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, rewards, dones, infos = venv.step(action)

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
            is_success = infos[0].get("is_success", False)
            stats['success'].append(int(is_success))
            stats['lengths'].append(step_count)

            phases = ["1:REACH", "2:PUSH", "3:HOLD", "4:BACK"]
            stats['max_phases'].append(phases[max_phase_idx - 1])
            stats['phase_times'].append(dict(phase_time))

            print(f"Episode {ep_count+1}/{n_episodes} - Success: {is_success} - Max Phase: {phases[max_phase_idx-1]} - Length: {step_count}")

            ep_count      += 1
            step_count    = 0
            phase_time    = {'1:REACH': 0, '2:PUSH': 0, '3:HOLD': 0, '4:BACK': 0}
            max_phase_idx = 1
            obs           = venv.reset()

    return stats

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

    print("Stats computed and saved. Generating plots...")
    create_plots(train_stats, eval_stats)
    print("Done!")