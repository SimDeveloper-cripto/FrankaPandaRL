#!/usr/bin/env python3
# close_generalized/train_gen.py

import numpy as np
import os, sys, time, argparse

from dotenv import load_dotenv
load_dotenv()

if os.name == "nt":
    mujoco_path = os.getenv("MUJOCO_PATH")
    if mujoco_path and os.path.exists(mujoco_path):
        os.add_dll_directory(mujoco_path)

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecMonitor

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.train_close_config import TrainConfig

from env_gen import GeneralizedDoorEnv
from train_close import SuccessRateCallback


# ─────────────────────────────────────────────────────────────────────────────
class GraspDiagnosticCallback(BaseCallback):
    def __init__(self, log_every: int = 10_000):
        super().__init__()
        self.log_every      = log_every
        self.grasps         = 0
        self.retreats       = 0
        self.episodes       = 0
        self._was_grasp     = {}
        self._was_retreat   = {}

    def _on_step(self) -> bool:
        try:
            grasp_phases      = self.training_env.get_attr("_grasp_phase")
            ready_to_retreats = self.training_env.get_attr("_ready_to_retreat")
            dones             = self.locals.get("dones", [False] * len(grasp_phases))

            for i, (gp, rr, done) in enumerate(zip(grasp_phases, ready_to_retreats, dones)):
                prev_g = self._was_grasp.get(i, False)
                if gp and not prev_g:
                    self.grasps += 1

                prev_r = self._was_retreat.get(i, False)
                if rr and not prev_r:
                    self.retreats += 1

                if done:
                    self.episodes += 1

                self._was_grasp[i]   = gp if not done else False
                self._was_retreat[i] = rr if not done else False
        except Exception:
            pass

        if self.n_calls % self.log_every == 0 and self.episodes > 0:
            grasp_rate   = self.grasps   / max(1, self.episodes)
            retreat_rate = self.retreats / max(1, self.episodes)

            self.logger.record("custom/grasp_rate"   , grasp_rate)
            self.logger.record("custom/grasp_count"  , self.grasps)
            self.logger.record("custom/retreat_rate" , retreat_rate)
            self.logger.record("custom/retreat_count", self.retreats)
            self.logger.record("custom/episodes"     , self.episodes)

            self.grasps   = 0
            self.retreats = 0
            self.episodes = 0
        return True


# ─────────────────────────────────────────────────────────────────────────────
class AdaptiveCurriculumCallback(BaseCallback):
    def __init__(
        self,
        success_callback: SuccessRateCallback,
        grasp_callback  : GraspDiagnosticCallback,
        check_freq      : int = 25_000
    ):
        super().__init__()
        self.success_cb = success_callback
        self.grasp_cb   = grasp_callback
        self.check_freq = check_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            sr = self.success_cb.successes / max(1, self.success_cb.episodes)
            gr = self.grasp_cb.grasps      / max(1, self.grasp_cb.episodes)

            current_level = self.training_env.get_attr("curriculum_level")[0]

            if sr > 0.85 and gr > 0.50 and current_level < 1.0:
                new_level = min(1.0, current_level + 0.05)

                self.training_env.env_method("set_curriculum_level", new_level)

                self.success_cb.successes = 0
                self.success_cb.episodes  = 0
                print(f"\n[CURRICULUM] Level Up → {new_level:.2f}  "
                      f"(success={sr:.2f}, grasp={gr:.2f})")
            elif sr > 0.85 and gr <= 0.50:
                print(f"\n[CURRICULUM] Bloccato: success={sr:.2f} ok ma "
                      f"grasp_rate={gr:.2f} < 0.50. Il robot sta ancora usando pushing.")

        return True

class CustomEvalCallback(BaseCallback):
    def __init__(
        self,
        eval_env,
        best_model_save_path: str,
        log_path: str,
        eval_freq: int = 10000,
        n_eval_episodes: int = 20,
        verbose: int = 1
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
                # If best success rate is reasonable (>40%), and current success rate drops by more than 25% (0.25)
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
                    # Reset degradation count if performance is acceptable
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--play",  action="store_true")
    parser.add_argument("--model", type=str, default="runs/close_gen/best_model.zip")
    parser.add_argument("--resume", action="store_true", help="Resume training from best_model.zip in the configured run directory")
    parser.add_argument("--resume-model", type=str, default=None, help="Path to specific model checkpoint to load/resume")
    parser.add_argument("--resume-vecnorm", type=str, default=None, help="Path to specific VecNormalize statistics to load")
    parser.add_argument("--total-steps", type=int, default=None, help="Override total training steps")
    args = parser.parse_args()

    my_cfg = TrainConfig(run_dir="runs/close_gen", num_envs=8, horizon=500)
    if args.play:
        raw_env = GeneralizedDoorEnv(my_cfg, render_mode="human")
        vn_path = os.path.join(os.path.dirname(args.model), "vecnormalize.pkl")

        if os.path.exists(vn_path):
            import pickle
            with open(vn_path, "rb") as f:
                vn_data = pickle.load(f)
            obs_rms = vn_data.obs_rms
        else:
            obs_rms = None

        raw_env.set_curriculum_level(1.0)
        model = SAC.load(args.model)

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
        alpha       = 0.5
        target_dt   = 1.0 / my_cfg.control_freq

        print("[INFO] Playing...")
        
        current_reward = 0.0
        step_counter   = 0
        while True:
            start_t = time.perf_counter()

            if obs_rms is not None:
                obs_norm  = (obs - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8)
                obs_norm  = np.clip(obs_norm, -10.0, 10.0)
                action, _ = model.predict(obs_norm, deterministic = True)
            else:
                action, _ = model.predict(obs, deterministic = True)

            action      = alpha * action + (1.0 - alpha) * prev_action
            prev_action = action.copy()

            obs, current_reward, terminated, truncated, info = raw_env.step(action)
            done                                             = terminated or truncated
            step_counter += 1

            # ─────────────────────────────────────────────────────────────────
            try:
                viewer = raw_env._rs_env.viewer

                if raw_env._success_latched:
                    if getattr(raw_env, "_ready_to_retreat", False):
                        colore_fase = [0.0, 0.0, 1.0, 1.0]
                    else:
                        colore_fase = [0.0, 1.0, 0.0, 1.0]
                elif raw_env._grasp_phase:
                    colore_fase = [1.0, 1.0, 0.0, 1.0]
                else:
                    colore_fase = [1.0, 0.0, 0.0, 1.0]

                gripper_action = action[-1]
                colore_grip    = [0.0, 1.0, 1.0, 1.0] if gripper_action > 0.65 else [0.4, 0.4, 0.4, 1.0]

                altezza_barra = float(np.clip((current_reward + 10) / 20.0, 0.01, 0.3))
                colore_reward = [1.0 - (altezza_barra*3), altezza_barra*3, 0.0, 1.0]

                if hasattr(viewer, "_markers"):
                    viewer._markers.clear()

                if hasattr(viewer, "_markers"):
                    viewer._markers.clear() 
                
                viewer.add_marker(
                    pos  = [0.2, 0.0, 1.1],
                    size = [0.06, 0.06, 0.06],
                    rgba = colore_fase,
                    type = 2
                )
                viewer.add_marker(
                    pos  = [0.2, -0.15, 1.1],
                    size = [0.04, 0.04, 0.04],
                    rgba = colore_grip,
                    type = 3
                )
                viewer.add_marker(
                    pos  = [0.2, 0.15, 1.0 + (altezza_barra / 2.0)],
                    size = [0.02, 0.02, altezza_barra],
                    rgba = colore_reward,
                    type = 1
                )

                if hasattr(viewer, "update"):
                    viewer.update()

            except Exception as e:
                pass
            # ─────────────────────────────────────────────────────────────────

            raw_env.render()
            elapsed = time.perf_counter() - start_t
            if elapsed < target_dt:
                time.sleep(target_dt - elapsed)

            if done:
                obs, _ = raw_env.reset()
                setup_interactive_viewer(raw_env)

                prev_action[:] = 0
                step_counter   = 0
    else:  # Train
        os.makedirs(my_cfg.run_dir, exist_ok=True)

        if args.total_steps is not None:
            my_cfg.total_steps = args.total_steps

        # Create env
        env = DummyVecEnv([lambda: GeneralizedDoorEnv(my_cfg) for _ in range(my_cfg.num_envs)])
        env = VecMonitor(env)

        # Determine paths to load
        load_model_path = None
        load_vn_path = None

        if args.resume_model:
            load_model_path = args.resume_model
        elif args.resume:
            load_model_path = os.path.join(my_cfg.run_dir, "best_model.zip")

        if args.resume_vecnorm:
            load_vn_path = args.resume_vecnorm
        elif args.resume or args.resume_model:
            potential_paths = []
            if load_model_path:
                potential_paths.append(os.path.join(os.path.dirname(load_model_path), "vecnormalize.pkl"))
            potential_paths.append(os.path.join(my_cfg.run_dir, "vecnormalize.pkl"))
            for p in potential_paths:
                if os.path.exists(p):
                    load_vn_path = p
                    break

        # Load or create VecNormalize
        if load_vn_path and os.path.exists(load_vn_path):
            print(f"[RESUME] Loading VecNormalize statistics from {load_vn_path}")
            env = VecNormalize.load(load_vn_path, env)
            env.training = True
            env.norm_reward = True
        else:
            print("[TRAIN] Creating new VecNormalize statistics")
            env = VecNormalize(env, norm_obs=True, norm_reward=True)

        scb = SuccessRateCallback(log_every=10_000)
        gcb = GraspDiagnosticCallback(log_every=10_000)
        ccb = AdaptiveCurriculumCallback(success_callback=scb, grasp_callback=gcb)

        # Create evaluation environment (synced before eval)
        eval_env = DummyVecEnv([lambda: GeneralizedDoorEnv(my_cfg)])
        eval_env = VecMonitor(eval_env)
        eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False)

        eval_cb = CustomEvalCallback(
            eval_env,
            best_model_save_path = my_cfg.run_dir,
            log_path             = os.path.join(my_cfg.run_dir, "eval"),
            eval_freq            = 10_000,
            n_eval_episodes      = 20,
            verbose              = 1
        )

        # Instantiate or Load SAC Model
        if load_model_path and os.path.exists(load_model_path):
            print(f"[RESUME] Loading SAC model weights from {load_model_path}")
            model = SAC.load(
                load_model_path,
                env             = env,
                tensorboard_log = my_cfg.tb_dir,
                custom_objects  = {
                    "learning_rate": my_cfg.learning_rate,
                    "lr_schedule": None
                }
            )
            # Adjust learning starts
            model.learning_starts = max(0, model.learning_starts - model.num_timesteps)
        else:
            print("[TRAIN] Initializing fresh SAC model")
            model = SAC(
                "MlpPolicy",
                env,
                verbose         = 1,
                tensorboard_log = my_cfg.tb_dir,
                learning_rate   = my_cfg.learning_rate,
                buffer_size     = my_cfg.buffer_size,
                batch_size      = my_cfg.batch_size,
                gamma           = my_cfg.gamma,
                tau             = my_cfg.tau,
                train_freq      = my_cfg.train_freq,
                gradient_steps  = my_cfg.gradient_steps,
                learning_starts = my_cfg.learning_starts,
                ent_coef        = my_cfg.ent_coef,
                policy_kwargs   = dict(net_arch=list(my_cfg.policy_net_arch)),
            )

        model.learn(
            total_timesteps     = my_cfg.total_steps,
            callback            = [scb, gcb, ccb, eval_cb],
            reset_num_timesteps = False if (load_model_path is not None) else True
        )
        model.save(os.path.join(my_cfg.run_dir, "best_model"))
        env.save(os.path.join(my_cfg.run_dir, "vecnormalize.pkl"))

if __name__ == "__main__":
    main()