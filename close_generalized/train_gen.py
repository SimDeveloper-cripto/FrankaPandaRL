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

# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--play",  action="store_true")
    parser.add_argument("--model", type=str, default="runs/close_gen/best_model.zip")
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

        env = DummyVecEnv([lambda: GeneralizedDoorEnv(my_cfg) for _ in range(my_cfg.num_envs)])
        env = VecMonitor(env)
        env = VecNormalize(env, norm_obs=True, norm_reward=True)

        scb = SuccessRateCallback(log_every=10_000)
        gcb = GraspDiagnosticCallback(log_every=10_000)
        ccb = AdaptiveCurriculumCallback(success_callback=scb, grasp_callback=gcb)

        from stable_baselines3.common.callbacks import EvalCallback
        from train_close import SaveVecNormalizeCallback

        eval_env = DummyVecEnv([lambda: GeneralizedDoorEnv(my_cfg)])
        eval_env = VecMonitor(eval_env)
        eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False)

        eval_env.obs_rms = env.obs_rms

        eval_cb = EvalCallback(
            eval_env,
            best_model_save_path = my_cfg.run_dir,
            log_path             = os.path.join(my_cfg.run_dir, "eval"),
            eval_freq            = 10_000,
            n_eval_episodes      = 20,
            deterministic        = True,
            render               = False,
            callback_on_new_best = SaveVecNormalizeCallback(
                save_path = os.path.join(my_cfg.run_dir, "vecnormalize.pkl")
            ),
        )

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
            total_timesteps = my_cfg.total_steps,
            callback        = [scb, gcb, ccb, eval_cb]
        )
        model.save(os.path.join(my_cfg.run_dir, "best_model"))
        env.save(os.path.join(my_cfg.run_dir, "vecnormalize.pkl"))

if __name__ == "__main__":
    main()