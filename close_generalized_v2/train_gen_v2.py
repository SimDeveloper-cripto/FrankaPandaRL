#!/usr/bin/env python3
# close_generalized_v2/train_gen_v2.py
#
# Training entry-point for AdvancedGeneralizedDoorEnv (v2)

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

from train_close import SuccessRateCallback
from close_generalized_v2.config_v2 import TrainConfigV2
from close_generalized_v2.env_v2    import AdvancedGeneralizedDoorEnv


# ─────────────────────────────────────────────────────────────────────────────
class GraspDiagnosticCallbackV2(BaseCallback):
    def __init__(self, log_every: int = 10_000):
        super().__init__()
        self.log_every    = log_every
        self.grasps       = 0
        self.retreats     = 0
        self.episodes     = 0
        self._was_grasp   = {}
        self._was_retreat = {}

        self.reach_steps_total = 0
        self.push_steps_total  = 0
        self.episodes_eff      = 0

    def _on_step(self) -> bool:
        try:
            # v2: delegate to FSM state via env properties
            grasp_phases      = self.training_env.get_attr("_grasp_phase")
            ready_to_retreats = self.training_env.get_attr("_ready_to_retreat")
            dones             = self.locals.get("dones", [False] * len(grasp_phases))

            for i, (gp, rr, done) in enumerate(zip(grasp_phases, ready_to_retreats, dones)):
                if gp and not self._was_grasp.get(i, False):
                    self.grasps += 1
                if rr and not self._was_retreat.get(i, False):
                    self.retreats += 1
                if done:
                    self.episodes += 1
                    self.episodes_eff += 1

                self._was_grasp[i]   = gp if not done else False
                self._was_retreat[i] = rr if not done else False
        except Exception:
            pass

        if self.n_calls % self.log_every == 0 and self.episodes > 0:
            gr = self.grasps  / max(1, self.episodes)
            rr = self.retreats / max(1, self.episodes)

            self.logger.record("custom/grasp_rate",   gr)
            self.logger.record("custom/retreat_rate", rr)
            self.logger.record("custom/episodes",     self.episodes)

            self.grasps       = 0
            self.retreats     = 0
            self.episodes     = 0
            self.episodes_eff = 0

        return True


# ─────────────────────────────────────────────────────────────────────────────
class AdaptiveCurriculumV2(BaseCallback):
    def __init__(
        self,
        success_callback: SuccessRateCallback,
        grasp_callback  : GraspDiagnosticCallbackV2,
        cfg             : TrainConfigV2,
    ):
        super().__init__()
        self.success_cb = success_callback
        self.grasp_cb   = grasp_callback
        self.cfg        = cfg

    def _on_step(self) -> bool:
        if self.n_calls % self.cfg.curriculum_check_freq == 0:
            sr = self.success_cb.successes / max(1, self.success_cb.episodes)
            gr = self.grasp_cb.grasps      / max(1, self.grasp_cb.episodes)

            current_level = self.training_env.get_attr("curriculum_level")[0]
            efficiency_ok = True

            if sr > 0.85 and gr > 0.50 and efficiency_ok and current_level < 1.0:
                new_level = min(1.0, current_level + self.cfg.curriculum_advance_delta)
                self.training_env.env_method("set_curriculum_level", new_level)

                self.success_cb.successes = 0
                self.success_cb.episodes  = 0
                print(f"\n[CURRICULUM v2] Level Up → {new_level:.2f}  "
                      f"(success={sr:.2f}, grasp={gr:.2f})")

            elif sr > 0.85 and gr <= 0.50:
                print(f"\n[CURRICULUM v2] Blocked: success={sr:.2f} ok "
                      f"but grasp_rate={gr:.2f} < 0.50")

        return True


# ─────────────────────────────────────────────────────────────────────────────
class CustomEvalCallbackV2(BaseCallback):
    """
    Evaluation callback for v2.
    Same logic as v1 CustomEvalCallback, adapted for AdvancedGeneralizedDoorEnv
    """

    def __init__(
        self,
        eval_env,
        best_model_save_path: str,
        log_path            : str,
        eval_freq           : int = 10_000,
        n_eval_episodes     : int = 20,
        verbose             : int = 1,
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
        os.makedirs(self.log_path,             exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            import pickle
            if self.model.get_vec_normalize_env() is not None:
                self.eval_env.obs_rms = self.model.get_vec_normalize_env().obs_rms
                if hasattr(self.model.get_vec_normalize_env(), "ret_rms"):
                    self.eval_env.ret_rms = self.model.get_vec_normalize_env().ret_rms

            mean_reward, mean_length, mean_success = self._evaluate()

            print(f"\n--- [EVAL v2] Step {self.num_timesteps} ---")
            print(f"Mean Reward   : {mean_reward:.2f}")
            print(f"Success Rate  : {mean_success*100:.1f}%")
            print(f"Episode Length: {mean_length:.1f}")
            print(f"-----------------------------------\n")

            is_new_best = (
                mean_success > self.best_success_rate
                or (abs(mean_success - self.best_success_rate) < 1e-4
                    and mean_reward > self.best_mean_reward)
            )

            if is_new_best:
                self.best_success_rate = mean_success
                self.best_mean_reward  = mean_reward
                self.degradation_count = 0

                model_path = os.path.join(self.best_model_save_path, "best_model.zip")
                vn_path    = os.path.join(self.best_model_save_path, "vecnormalize.pkl")

                self.model.save(model_path)
                if self.model.get_vec_normalize_env() is not None:
                    self.model.get_vec_normalize_env().save(vn_path)

                print(f"[BEST v2] New best: {mean_success*100:.1f}% success")
            else:
                if self.best_success_rate > 0.40 and mean_success < self.best_success_rate - 0.25:
                    self.degradation_count += 1
                    if self.degradation_count >= 2:
                        # RECOVERY: RELOAD BEST MODEL
                        best_p  = os.path.join(self.best_model_save_path, "best_model.zip")
                        best_vn = os.path.join(self.best_model_save_path, "vecnormalize.pkl")
                        if os.path.exists(best_p):
                            self.model.set_parameters(best_p)
                        if os.path.exists(best_vn) and self.model.get_vec_normalize_env():
                            with open(best_vn, "rb") as f:
                                bvn = pickle.load(f)
                            self.model.get_vec_normalize_env().obs_rms = bvn.obs_rms
                        self.degradation_count = 0
                        print("[RECOVERY v2] Reloaded best model.")
                else:
                    self.degradation_count = 0

            self.logger.record("eval/mean_reward",    mean_reward)
            self.logger.record("eval/mean_ep_length", mean_length)
            self.logger.record("eval/success_rate",   mean_success)
            self.logger.dump(step=self.num_timesteps)

        return True

    def _evaluate(self):
        rewards, lengths, successes = [], [], []
        obs             = self.eval_env.reset()
        curr_r          = np.zeros(self.eval_env.num_envs)
        curr_l          = np.zeros(self.eval_env.num_envs)
        episodes_done   = 0

        while episodes_done < self.n_eval_episodes:
            actions, _              = self.model.predict(obs, deterministic=True)
            obs, rews, dones, infos = self.eval_env.step(actions)
            curr_r += rews
            curr_l += 1

            for i in range(self.eval_env.num_envs):
                if dones[i]:
                    rewards.append(curr_r[i])
                    lengths.append(curr_l[i])
                    successes.append(int(infos[i].get("is_success", False)))
                    curr_r[i] = 0.0
                    curr_l[i] = 0.0
                    episodes_done += 1
                    if episodes_done >= self.n_eval_episodes:
                        break

        return float(np.mean(rewards)), float(np.mean(lengths)), float(np.mean(successes))


# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Train/Play AdvancedGeneralizedDoorEnv v2")
    parser.add_argument("--play",           action = "store_true", help = "Play best model")
    parser.add_argument("--model",          type = str, default = "runs/close_gen_v2/best_model.zip")
    parser.add_argument("--resume",         action = "store_true")

    parser.add_argument("--resume-model",   type = str, default = None)
    parser.add_argument("--resume-vecnorm", type = str, default = None)
    parser.add_argument("--total-steps",    type = int, default = None)

    parser.add_argument("--beta-net", action = "store_true", help = "Enable beta-network (§3.5) — Phase 4")
    args = parser.parse_args()

    cfg = TrainConfigV2(run_dir = "runs/close_gen_v2", num_envs = 8, horizon = 500)

    if args.beta_net:
        cfg.use_beta_net = True
        print("[v2] Beta-network enabled (§3.5 — Phase 4)")

    if args.total_steps is not None:
        cfg.total_steps = args.total_steps

    # ── Play mode ─────────────────────────────────────────────────────────────
    if args.play:
        raw_env = AdvancedGeneralizedDoorEnv(cfg, render_mode="human")
        raw_env.set_curriculum_level(1.0)

        vn_path = os.path.join(os.path.dirname(args.model), "vecnormalize.pkl")
        obs_rms = None
        if os.path.exists(vn_path):
            import pickle
            with open(vn_path, "rb") as f:
                obs_rms = pickle.load(f).obs_rms

        model     = SAC.load(args.model)
        obs, _    = raw_env.reset()
        alpha     = 0.5
        target_dt = 1.0 / cfg.control_freq
        prev_act  = np.zeros(raw_env.action_space.shape)

        print("[v2] Playing... (Ctrl+C to stop)")
        while True:
            t0 = time.perf_counter()
            if obs_rms is not None:
                obs_n  = np.clip((obs - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8), -10, 10)
                act, _ = model.predict(obs_n, deterministic=True)
            else:
                act, _ = model.predict(obs, deterministic=True)

            act       = alpha * act + (1.0 - alpha) * prev_act
            prev_act  = act.copy()
            obs, r, term, trunc, info = raw_env.step(act)
            raw_env.render()

            elapsed = time.perf_counter() - t0
            if elapsed < target_dt:
                time.sleep(target_dt - elapsed)

            if term or trunc:
                print(f"[v2] Episode done — success={info.get('is_success')}, "
                      f"phase={info.get('fsm_phase_name')}")
                obs, _ = raw_env.reset()
                prev_act[:] = 0

    # ── Train mode ────────────────────────────────────────────────────────────
    else:
        os.makedirs(cfg.run_dir, exist_ok = True)

        env = DummyVecEnv([lambda: AdvancedGeneralizedDoorEnv(cfg) for _ in range(cfg.num_envs)])
        env = VecMonitor(env)

        # Determine Resume Paths
        load_model_path = args.resume_model or (
            os.path.join(cfg.run_dir, "best_model.zip") if args.resume else None
        )
        load_vn_path = args.resume_vecnorm
        if load_vn_path is None and (args.resume or args.resume_model):
            for p in [
                os.path.join(os.path.dirname(load_model_path or ""), "vecnormalize.pkl"),
                os.path.join(cfg.run_dir, "vecnormalize.pkl"),
            ]:
                if os.path.exists(p):
                    load_vn_path = p
                    break

        if load_vn_path and os.path.exists(load_vn_path):
            print(f"[RESUME v2] Loading VecNormalize from {load_vn_path}")

            env             = VecNormalize.load(load_vn_path, env)
            env.training    = True
            env.norm_reward = True
        else:
            env = VecNormalize(env, norm_obs = True, norm_reward = True)

        # Callbacks
        scb = SuccessRateCallback(log_every=10_000)
        gcb = GraspDiagnosticCallbackV2(log_every=10_000)
        ccb = AdaptiveCurriculumV2(scb, gcb, cfg)

        eval_env = DummyVecEnv([lambda: AdvancedGeneralizedDoorEnv(cfg)])
        eval_env = VecMonitor(eval_env)
        eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False)

        eval_cb = CustomEvalCallbackV2(
            eval_env,
            best_model_save_path = cfg.run_dir,
            log_path             = os.path.join(cfg.run_dir, "eval"),
            eval_freq            = cfg.eval_freq,
            n_eval_episodes      = cfg.n_eval_episodes,
            verbose              = 1,
        )

        # Model
        if load_model_path and os.path.exists(load_model_path):
            print(f"[RESUME v2] Loading SAC from {load_model_path}")
            model = SAC.load(
                load_model_path, env = env,
                tensorboard_log = cfg.tb_dir,
                custom_objects  = {"learning_rate": cfg.learning_rate, "lr_schedule": None}
            )
            model.learning_starts = max(0, model.learning_starts - model.num_timesteps)
        else:
            print("[TRAIN v2] Initializing fresh SAC model")
            model = SAC(
                "MlpPolicy", env,
                verbose         = 1,
                tensorboard_log = cfg.tb_dir,
                learning_rate   = cfg.learning_rate,
                buffer_size     = cfg.buffer_size,
                batch_size      = cfg.batch_size,
                gamma           = cfg.gamma,
                tau             = cfg.tau,
                train_freq      = cfg.train_freq,
                gradient_steps  = cfg.gradient_steps,
                learning_starts = cfg.learning_starts,
                ent_coef        = cfg.ent_coef,
                target_entropy  = cfg.target_entropy,
                policy_kwargs   = dict(net_arch=list(cfg.policy_net_arch)),
            )

        print(f"[v2] Training for {cfg.total_steps:,} steps → {cfg.run_dir}")
        model.learn(
            total_timesteps     = cfg.total_steps,
            callback            = [scb, gcb, ccb, eval_cb],
            reset_num_timesteps = (load_model_path is None),
        )
        model.save(os.path.join(cfg.run_dir, "best_model"))
        env.save(os.path.join(cfg.run_dir, "vecnormalize.pkl"))
        print("[v2] Training complete.")

if __name__ == "__main__":
    main()