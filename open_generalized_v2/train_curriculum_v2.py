#!/usr/bin/env python3
# open_generalized_v2/train_curriculum_v2.py
#
# Entry-point di training per l'APERTURA generalizzata v2 — SOLO curriculum 1
# (posa variabile, soglie adattive, fisica randomizzata). Speculare a
# close_generalized_v2/train_gen_v2.py, con la stessa logica di eval/best-model/
# VecNormalize e lo stesso schema di --play (env raw + obs_rms manuale).
#
# Uso:
#   # training (curriculum 1)
#   python -m open_generalized_v2.train_curriculum_v2 --total-steps 1500000
#
#   # play (visualizza la policy migliore)
#   python -m open_generalized_v2.train_curriculum_v2 --play
#
# Riferimenti: SAC (sb3); potential-based shaping [3]; domain randomization [8][17].

from __future__ import annotations

import os
import sys
import time
import argparse
import numpy as np

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import robusti (package-qualified con fallback piatto): funziona con
# `mjpython open_generalized_v2/train_curriculum_v2.py` e con `python -m ...`.
try:
    from open_generalized_v2.config_v2 import TrainConfigV2Open
    from open_generalized_v2.env_v2 import AdvancedGeneralizedOpenDoorEnv
except ModuleNotFoundError:
    from config_v2 import TrainConfigV2Open
    from env_v2 import AdvancedGeneralizedOpenDoorEnv


def make_env_fn(cfg, render_mode=None):
    def _thunk():
        env = AdvancedGeneralizedOpenDoorEnv(cfg, render_mode=render_mode)
        env.set_curriculum_level(cfg.fixed_curriculum_level)  # curriculum 1 fisso
        return env
    return _thunk


# ─────────────────────────────────────────────────────────────────────────────
# Eval callback: salva best_model + vecnormalize.pkl quando migliora il success.
# ─────────────────────────────────────────────────────────────────────────────
def build_eval_callback():
    from stable_baselines3.common.callbacks import BaseCallback

    class EvalBestCallback(BaseCallback):
        def __init__(self, eval_env, save_path, eval_freq, n_eval_episodes, verbose=1):
            super().__init__(verbose)
            self.eval_env = eval_env
            self.save_path = save_path
            self.eval_freq = eval_freq
            self.n_eval_episodes = n_eval_episodes
            self.best_success = -1.0
            self._next_eval = eval_freq
            os.makedirs(save_path, exist_ok=True)

        def _evaluate(self):
            succ, lengths = [], []
            for _ in range(self.n_eval_episodes):
                obs = self.eval_env.reset()
                done = np.array([False])
                steps = 0
                last_info = {}
                while not done[0]:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, _r, done, infos = self.eval_env.step(action)
                    last_info = infos[0]
                    steps += 1
                succ.append(int(bool(last_info.get("is_success", False))))
                lengths.append(steps)
            return float(np.mean(succ)), float(np.mean(lengths))

        def _on_step(self) -> bool:
            # trigger basato su num_timesteps (robusto a num_envs): n_calls*num_envs.
            # Usiamo una soglia progressiva così l'eval scatta a multipli REALI di eval_freq.
            if self.num_timesteps >= self._next_eval:
                self._next_eval += self.eval_freq
                if self.model.get_vec_normalize_env() is not None:
                    self.eval_env.obs_rms = self.model.get_vec_normalize_env().obs_rms
                sr, ml = self._evaluate()
                print(f"\n--- [EVAL OPEN v2] step {self.num_timesteps} ---")
                print(f"Success: {sr*100:.1f}%  (best {max(sr,self.best_success)*100:.1f}%)  ep_len {ml:.1f}\n")
                # checkpoint SEMPRE aggiornato (così interrompere a metà lascia un modello caricabile dal diagnostico)
                self.model.save(os.path.join(self.save_path, "latest_model.zip"))
                if self.model.get_vec_normalize_env() is not None:
                    self.model.get_vec_normalize_env().save(
                        os.path.join(self.save_path, "vecnormalize.pkl"))
                if sr > self.best_success:
                    self.best_success = sr
                    self.model.save(os.path.join(self.save_path, "best_model.zip"))
                    if self.model.get_vec_normalize_env() is not None:
                        self.model.get_vec_normalize_env().save(
                            os.path.join(self.save_path, "vecnormalize.pkl"))
                    print(f"[BEST OPEN v2] nuovo best: {sr*100:.1f}%")
            return True

    return EvalBestCallback


def train(cfg, total_steps):
    from stable_baselines3 import SAC
    from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize

    os.makedirs(cfg.run_dir, exist_ok=True)

    venv = DummyVecEnv([make_env_fn(cfg) for _ in range(cfg.num_envs)])
    venv = VecMonitor(venv)
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)

    eval_env = DummyVecEnv([make_env_fn(cfg)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    eval_env.training = False

    EvalBestCallback = build_eval_callback()
    cb = EvalBestCallback(eval_env, cfg.run_dir, cfg.eval_freq, cfg.n_eval_episodes)

    model = SAC(
        "MlpPolicy", venv,
        learning_rate=cfg.learning_rate, buffer_size=cfg.buffer_size,
        batch_size=cfg.batch_size, gamma=cfg.gamma, tau=cfg.tau,
        train_freq=cfg.train_freq, gradient_steps=cfg.gradient_steps,
        learning_starts=cfg.learning_starts, ent_coef=cfg.ent_coef,
        target_entropy=cfg.target_entropy,
        policy_kwargs=dict(net_arch=list(cfg.policy_net_arch)),
        tensorboard_log=cfg.tb_dir, seed=cfg.seed, verbose=1,
    )
    model.learn(total_timesteps=int(total_steps), callback=cb)
    model.save(os.path.join(cfg.run_dir, "final_model.zip"))
    venv.save(os.path.join(cfg.run_dir, "vecnormalize.pkl"))
    print("[OPEN v2] Training complete.")


def play(cfg, model_path=None):
    import pickle
    from stable_baselines3 import SAC

    env = AdvancedGeneralizedOpenDoorEnv(cfg, render_mode="human")
    env.set_curriculum_level(cfg.fixed_curriculum_level)

    obs_rms = None
    vn = os.path.join(cfg.run_dir, "vecnormalize.pkl")
    if os.path.exists(vn):
        with open(vn, "rb") as f:
            obs_rms = pickle.load(f).obs_rms

    model = SAC.load(model_path or os.path.join(cfg.run_dir, "best_model.zip"))

    def norm(o):
        if obs_rms is None:
            return o
        return np.clip((o - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8), -10.0, 10.0)

    obs, _ = env.reset()
    while True:
        action, _ = model.predict(norm(obs), deterministic=True)
        obs, _r, term, trunc, info = env.step(action)
        env.render()
        time.sleep(1.0 / 30.0)
        if term or trunc:
            obs, _ = env.reset()


def main():
    ap = argparse.ArgumentParser(description="Apertura generalizzata v2 — curriculum 1")
    ap.add_argument("--total-steps", type=int, default=None)
    ap.add_argument("--play", action="store_true")
    ap.add_argument("--model", type=str, default=None)
    args = ap.parse_args()

    cfg = TrainConfigV2Open()
    cfg.fixed_curriculum_level = 1.0   # questo progetto è SOLO curriculum 1
    if args.play:
        play(cfg, args.model)
    else:
        train(cfg, args.total_steps or cfg.total_steps)


if __name__ == "__main__":
    main()