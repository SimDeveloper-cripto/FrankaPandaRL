#!/usr/bin/env python3
# open_generalized/train_curriculum.py

from __future__ import annotations

import os
import sys
import json
import time
import argparse
import numpy as np
from dataclasses import asdict
from typing import Any, Dict, Optional, List

from dotenv import load_dotenv
load_dotenv()

if os.name == "nt":
    mujoco_path = os.getenv("MUJOCO_PATH")
    if mujoco_path and os.path.exists(mujoco_path):
        os.add_dll_directory(mujoco_path)

# Setup paths so imports work whether run from workspace root or inside the folder
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

from env_goal_door import GoalDoorEnv
from teacher import StageTeacher, StageSpec
from config.train_open_config import TrainConfig


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def save_config(cfg: TrainConfig, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2, sort_keys=True)

def make_env_fn(cfg: TrainConfig, teacher: StageTeacher, render_mode: Optional[str] = None):
    def _thunk():
        return GoalDoorEnv(cfg=cfg, teacher=teacher, render_mode=render_mode, door_open_cap_rad=0.400)
    return _thunk

class TeacherUpdateCallback(BaseCallback):
    def __init__(self, teacher: StageTeacher, verbose: int = 0):
        super().__init__(verbose)
        self.teacher   = teacher
        self._ep_count = 0

    def _on_step(self) -> bool:
        infos: List[Dict[str, Any]] = self.locals.get("infos", [])
        dones                       = self.locals.get("dones", None)

        if dones is None:
            return True

        # For each environment instance, when (done == True), we treat it as episode end
        for done, info in zip(dones, infos):
            if not done:
                continue

            # Success for curriculum: use info["is_success"] if present, else infer from episode data
            success = bool(info.get("is_success", False))
            self.teacher.update(success=success)
            self._ep_count += 1

        if self._ep_count > 0 and (self._ep_count % 100 == 0):
            st = self.teacher.stats()
            self.logger.record("curriculum/stage_idx", st["stage_idx"])
            self.logger.record("curriculum/success_rate_window", st["success_rate_window"])

        return True


class CustomEvalCallback(BaseCallback):
    def __init__(
        self,
        eval_env,
        best_model_save_path: str,
        log_path            : str,
        eval_freq           : int = 10000,
        n_eval_episodes     : int = 20,
        verbose             : int = 1
    ):
        super().__init__(verbose)
        self.eval_env             = eval_env
        self.best_model_save_path = best_model_save_path
        self.log_path             = log_path
        self.eval_freq            = eval_freq
        self.n_eval_episodes      = n_eval_episodes
        self.best_success_rate    = -1.0
        self.best_mean_reward     = -np.inf
        self.degradation_count    = 0

        os.makedirs(self.best_model_save_path, exist_ok=True)
        os.makedirs(self.log_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            import pickle
            # Sync observation normalization stats between training env and eval env
            if self.model.get_vec_normalize_env() is not None:
                self.eval_env.obs_rms = self.model.get_vec_normalize_env().obs_rms
                if hasattr(self.model.get_vec_normalize_env(), "ret_rms"):
                    self.eval_env.ret_rms = self.model.get_vec_normalize_env().ret_rms

            # Run evaluation
            mean_reward, mean_length, mean_success = self.evaluate()

            # Print evaluation summary
            print(f"\n--- [EVALUATION] Step {self.num_timesteps} ---")
            print(f"Mean Reward: {mean_reward:.2f} (best: {self.best_mean_reward:.2f})")
            print(f"Mean Success Rate: {mean_success*100:.1f}% (best: {self.best_success_rate*100:.1f}%)")
            print(f"Mean Episode Length: {mean_length:.1f}")
            print(f"-----------------------------------------\n")

            # Check if this is the best model
            is_new_best = False
            if mean_success > self.best_success_rate:
                is_new_best = True
            elif abs(mean_success - self.best_success_rate) < 1e-4 and mean_reward > self.best_mean_reward:
                is_new_best = True

            if is_new_best:
                print(f"[BEST MODEL] Saving new best model with success rate {mean_success*100:.1f}% and mean reward {mean_reward:.2f}")
                self.best_success_rate = mean_success
                self.best_mean_reward = mean_reward

                # Save model weights
                model_path = os.path.join(self.best_model_save_path, "best_model.zip")
                self.model.save(model_path)

                # Save VecNormalize stats
                if self.model.get_vec_normalize_env() is not None:
                    vn_path = os.path.join(self.best_model_save_path, "vecnormalize.pkl")
                    self.model.get_vec_normalize_env().save(vn_path)

                # Reset degradation count since we found a new best
                self.degradation_count = 0
            else:
                # degradation/overfitting check:
                if self.best_success_rate > 0.40 and mean_success < self.best_success_rate - 0.25:
                    self.degradation_count += 1
                    print(f"[WARNING] Performance degradation detected ({self.degradation_count}/2): current success rate {mean_success*100:.1f}% vs best {self.best_success_rate*100:.1f}%")

                    if self.degradation_count >= 2:
                        print("\n======================================================================")
                        print("[RECOVERY] Significant policy degradation / overfitting detected.")
                        print(f"Reloading best model weights and VecNormalize stats from step of best success rate ({self.best_success_rate*100:.1f}%)...")
                        print("======================================================================\n")

                        best_model_path = os.path.join(self.best_model_save_path, "best_model.zip")
                        best_vn_path    = os.path.join(self.best_model_save_path, "vecnormalize.pkl")

                        if os.path.exists(best_model_path):
                            self.model.set_parameters(best_model_path)

                        if os.path.exists(best_vn_path) and self.model.get_vec_normalize_env() is not None:
                            with open(best_vn_path, "rb") as f:
                                best_vn = pickle.load(f)
                            self.model.get_vec_normalize_env().obs_rms = best_vn.obs_rms
                            if hasattr(best_vn, "ret_rms"):
                                self.model.get_vec_normalize_env().ret_rms = best_vn.ret_rms

                        self.degradation_count = 0
                else:
                    self.degradation_count = 0

            # Log to tensorboard
            self.logger.record("eval/mean_reward", mean_reward)
            self.logger.record("eval/mean_ep_length", mean_length)
            self.logger.record("eval/success_rate", mean_success)
            self.logger.dump(step=self.num_timesteps)

        return True

    def evaluate(self):
        episode_rewards   = []
        episode_lengths   = []
        episode_successes = []

        obs             = self.eval_env.reset()
        current_rewards = np.zeros(self.eval_env.num_envs)
        current_lengths = np.zeros(self.eval_env.num_envs)

        episodes_completed = 0
        while episodes_completed < self.n_eval_episodes:
            actions, _ = self.model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = self.eval_env.step(actions)

            current_rewards += rewards
            current_lengths += 1

            for i in range(self.eval_env.num_envs):
                if dones[i]:
                    episode_rewards.append(current_rewards[i])
                    episode_lengths.append(current_lengths[i])

                    is_success = infos[i].get("is_success", False)
                    episode_successes.append(int(is_success))

                    current_rewards[i] = 0.0
                    current_lengths[i] = 0
                    episodes_completed += 1
                    if episodes_completed >= self.n_eval_episodes:
                        break

        return float(np.mean(episode_rewards)), float(np.mean(episode_lengths)), float(np.mean(episode_successes))


def build_teacher(seed: int) -> StageTeacher:
    # Stage curriculum tuned for limit (0.400 rad cap).
    stages = (
        StageSpec(
            name="S0_easy",
            goal_frac_min=0.05, goal_frac_max=0.20,
            friction_scale_min=1.0, friction_scale_max=1.0,
            damping_scale_min=1.0,  damping_scale_max=1.0,
        ),
        StageSpec(
            name="S1_mid",
            goal_frac_min=0.20, goal_frac_max=0.50,
            friction_scale_min=0.9, friction_scale_max=1.1,
            damping_scale_min=0.9,  damping_scale_max=1.2,
        ),
        StageSpec(
            name="S2_harder",
            goal_frac_min=0.50, goal_frac_max=0.80,
            friction_scale_min=0.7, friction_scale_max=1.3,
            damping_scale_min=0.8,  damping_scale_max=1.6,
        ),
        StageSpec(
            name="S3_full",
            goal_frac_min=0.80, goal_frac_max=1.00,
            friction_scale_min=0.6, friction_scale_max=1.5,
            damping_scale_min=0.7,  damping_scale_max=2.0,
        ),
        StageSpec(
            name="S4_mix",
            goal_frac_min=0.05, goal_frac_max=1.00,
            friction_scale_min=0.5, friction_scale_max=1.5,
            damping_scale_min=0.5,  damping_scale_max=2.0,
        ),
    )

    return StageTeacher(
        stages            = stages,
        window_episodes   = 200,
        promote_threshold = 0.85,
        seed              = seed,
    )

def train(cfg: TrainConfig, resume: bool = False, resume_model: Optional[str] = None, resume_vecnorm: Optional[str] = None):
    ensure_dir(cfg.run_dir)
    ensure_dir(cfg.tb_dir)
    save_config(cfg, os.path.join(cfg.run_dir, "open_config_curriculum.json"))

    eval_episodes = 20
    teacher       = build_teacher(seed=cfg.seed)
    
    # We want to support environment creation
    vec           = DummyVecEnv([make_env_fn(cfg, teacher, render_mode=None) for _ in range(cfg.num_envs)])
    vec           = VecMonitor(vec)

    # Determine paths to load
    load_model_path = None
    load_vn_path    = None

    if resume_model:
        load_model_path = resume_model
    elif resume:
        best_model_path  = os.path.join(cfg.run_dir, "best_model.zip")
        final_model_path = os.path.join(cfg.run_dir, "final_model_open_curriculum.zip")
        if os.path.exists(best_model_path):
            load_model_path = best_model_path
        elif os.path.exists(final_model_path):
            load_model_path = final_model_path
        else:
            load_model_path = best_model_path # Fallback

    if resume_vecnorm:
        load_vn_path = resume_vecnorm
    elif resume or resume_model:
        potential_paths = []
        if load_model_path:
            potential_paths.append(os.path.join(os.path.dirname(load_model_path), "vecnormalize.pkl"))
        potential_paths.append(os.path.join(cfg.run_dir, "vecnormalize.pkl"))
        
        # Check checkpoints folder
        checkpoints_dir = os.path.join(cfg.run_dir, "checkpoints")
        if os.path.exists(checkpoints_dir):
            import glob
            pkls = glob.glob(os.path.join(checkpoints_dir, "*vecnormalize*.pkl"))
            if pkls:
                pkls.sort()
                potential_paths.append(pkls[-1])
                
        for p in potential_paths:
            if os.path.exists(p):
                load_vn_path = p
                break

    # Load or create VecNormalize
    if cfg.vecnormalize:
        if load_vn_path and os.path.exists(load_vn_path):
            print(f"[RESUME] Loading VecNormalize statistics from {load_vn_path}")
            vec = VecNormalize.load(load_vn_path, vec)
            vec.training = True
            vec.norm_reward = True
        else:
            print("[TRAIN] Creating new VecNormalize statistics")
            vec = VecNormalize(vec, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # Eval environment
    eval_teacher = build_teacher(seed=cfg.seed + 999)
    eval_env     = DummyVecEnv([make_env_fn(cfg, eval_teacher, render_mode=None)])
    eval_env     = VecMonitor(eval_env)
    if cfg.vecnormalize:
        eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    tb_log = os.path.join(cfg.tb_dir, "tb_open_door_sac_curriculum")

    # Instantiate or Load SAC Model
    if load_model_path and os.path.exists(load_model_path):
        print(f"[RESUME] Loading SAC model weights from {load_model_path}")
        model = SAC.load(
            load_model_path,
            env             = vec,
            tensorboard_log = tb_log,
            custom_objects  = {
                "learning_rate": cfg.learning_rate,
                "lr_schedule": None
            }
        )
        # Adjust learning starts
        model.learning_starts = max(0, model.learning_starts - model.num_timesteps)
    else:
        print("[TRAIN] Initializing fresh SAC model")
        model = SAC(
            policy         = "MlpPolicy",
            env            = vec,
            verbose        = 1,
            tensorboard_log= tb_log,
            seed           = cfg.seed,
            learning_rate  = cfg.learning_rate,
            buffer_size    = cfg.buffer_size,
            batch_size     = cfg.batch_size,
            gamma          = cfg.gamma,
            tau            = cfg.tau,
            train_freq     = cfg.train_freq,
            gradient_steps = cfg.gradient_steps,
            ent_coef       = cfg.ent_coef,
            policy_kwargs  = dict(net_arch=list(cfg.policy_net_arch)),
        )

    eval_cb = CustomEvalCallback(
        eval_env,
        best_model_save_path = cfg.run_dir,
        log_path             = os.path.join(cfg.run_dir, "eval"),
        eval_freq            = max(1, cfg.eval_freq // max(1, cfg.num_envs)),
        n_eval_episodes      = eval_episodes,
        verbose              = 1
    )

    callbacks = [
        TeacherUpdateCallback(teacher=teacher),
        CheckpointCallback(
            save_freq   = max(1, cfg.checkpoint_freq // max(1, cfg.num_envs)),
            save_path   = os.path.join(cfg.run_dir, "checkpoints"),
            name_prefix = "open_door_sac_curriculum",
            save_replay_buffer = True,
            save_vecnormalize  = True,
        ),
        eval_cb,
    ]

    model.learn(
        total_timesteps     = int(cfg.total_steps), 
        callback            = callbacks,
        reset_num_timesteps = False if (load_model_path is not None) else True
    )
    
    # Save final models
    model.save(os.path.join(cfg.run_dir, "final_model_open_curriculum"))
    model.save(os.path.join(cfg.run_dir, "best_model")) # Save as best_model for consistency
    if cfg.vecnormalize:
        vec.save(os.path.join(cfg.run_dir, "vecnormalize.pkl"))

def main():
    parser = argparse.ArgumentParser(description="Train/Play Generalized Door Opening Task")
    parser.add_argument("--play",           action="store_true",    help="Play the model using human rendering")
    parser.add_argument("--model",          type=str, default=None, help="Path to model checkpoint to play")
    parser.add_argument("--resume",         action="store_true",    help="Resume training from best_model.zip / vecnormalize.pkl in run_dir")
    parser.add_argument("--resume-model",   type=str, default=None, help="Specific model checkpoint path to resume training from")
    parser.add_argument("--resume-vecnorm", type=str, default=None, help="Specific VecNormalize statistics path to resume training from")
    parser.add_argument("--total-steps",    type=int, default=None, help="Override total training steps")
    args = parser.parse_args()

    cfg = TrainConfig(run_dir="runs/open_gen")
    
    if args.total_steps is not None:
        cfg.total_steps = args.total_steps

    if args.play:
        # Play mode
        model_path = args.model
        if model_path is None:
            best_model_path  = os.path.join(cfg.run_dir, "best_model.zip")
            final_model_path = os.path.join(cfg.run_dir, "final_model_open_curriculum.zip")
            if os.path.exists(best_model_path):
                model_path = best_model_path
            elif os.path.exists(final_model_path):
                model_path = final_model_path
            else:
                model_path = best_model_path # Fallback

        vn_path = os.path.join(os.path.dirname(model_path), "vecnormalize.pkl")
        if not os.path.exists(vn_path):
            checkpoints_dir = os.path.join(os.path.dirname(model_path), "checkpoints")
            if os.path.exists(checkpoints_dir):
                import glob
                pkls = glob.glob(os.path.join(checkpoints_dir, "*vecnormalize*.pkl"))
                if pkls:
                    pkls.sort()
                    vn_path = pkls[-1]

        teacher = StageTeacher(
            stages=(
                StageSpec(
                    name="test_max",
                    goal_frac_min      = 1.0,
                    goal_frac_max      = 1.0,
                    friction_scale_min = 1.0,
                    friction_scale_max = 1.0,
                    damping_scale_min  = 1.0,
                    damping_scale_max  = 1.0,
                ),
            ),
            promote_threshold=1.0,
        )

        raw_env = GoalDoorEnv(cfg=cfg, teacher=teacher, render_mode="human", door_open_cap_rad=0.400)

        obs_rms = None
        if os.path.exists(vn_path):
            print(f"[PLAY] Loading VecNormalize statistics from {vn_path}")
            import pickle
            with open(vn_path, "rb") as f:
                vn_data = pickle.load(f)
            obs_rms = vn_data.obs_rms
        else:
            print("[PLAY] Warning: No VecNormalize statistics found, using raw observations.")

        model = SAC.load(model_path)

        def setup_interactive_viewer(env):
            try:
                env._rs_env.viewer.set_camera(camera_id = -1)
                env._rs_env.viewer.user_camera_action = True 
                env._rs_env.viewer.vopt.flags[:]      = 1  
                if hasattr(env._rs_env.viewer, "ui"):
                    env._rs_env.viewer.ui.enable  = True
                env._rs_env.viewer.custom_profile = True
            except Exception:
                pass

        obs, _ = raw_env.reset()
        setup_interactive_viewer(raw_env)

        prev_action = np.zeros(raw_env.action_space.shape)
        alpha       = cfg.action_smooth_alpha if hasattr(cfg, "action_smooth_alpha") else 0.2
        target_dt   = 1.0 / cfg.control_freq

        print(f"[PLAY] Playing model: {model_path} (Ctrl+C to stop)")
        while True:
            start_t = time.perf_counter()

            if obs_rms is not None:
                obs_norm = (obs - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8)
                obs_norm = np.clip(obs_norm, -10.0, 10.0)
                action, _ = model.predict(obs_norm, deterministic=True)
            else:
                action, _ = model.predict(obs, deterministic=True)

            action      = alpha * action + (1.0 - alpha) * prev_action
            prev_action = action.copy()

            obs, current_reward, terminated, truncated, info = raw_env.step(action)
            done = terminated or truncated

            raw_env.render()
            elapsed = time.perf_counter() - start_t
            if elapsed < target_dt:
                time.sleep(target_dt - elapsed)

            if done:
                print(f"[PLAY] Episode done - success: {info.get('is_success')}, goal_angle: {info.get('goal_angle')}")
                obs, _ = raw_env.reset()
                setup_interactive_viewer(raw_env)
                prev_action[:] = 0.0
    else:
        train(
            cfg           = cfg,
            resume        = args.resume,
            resume_model  = args.resume_model,
            resume_vecnorm = args.resume_vecnorm
        )

if __name__ == "__main__":
    main()