#!/usr/bin/env python3
# tests_v2/_common_v2.py
#
# Schema IDENTICO al --play di train_gen_v2.py
#   - env RAW: AdvancedGeneralizedDoorEnv(cfg)
#   - normalizzazione osservazione a mano con obs_rms preso da vecnormalize.pkl (se esiste)
#   - lettura stato SOLO da `info`:
#       info["fsm_phase"]     / ["fsm_phase_name"]    / ["is_success"]
#       info["latch_qpos"]    / ["door_qpos"]         / ["door_angle"]
#       info["hold_duration"] / ["target_hold_steps"] / ["curriculum_level"]
#
# In v2 tutto passa da `info`, e l'eval ricalca esattamente il play che gia' funziona

import os
import pickle
import numpy as np

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

DEFAULT_RUN_DIR = "runs/close_gen_v2"
PHASE_NAMES     = ["REACH", "PUSH", "HOLD", "RETREAT"]

def make_raw_env(curriculum_level = 1.0, run_dir = DEFAULT_RUN_DIR, horizon = 500):
    from close_generalized_v2.config_v2 import TrainConfigV2
    from close_generalized_v2.env_v2    import AdvancedGeneralizedDoorEnv

    cfg = TrainConfigV2(run_dir = run_dir, num_envs = 1, horizon = horizon)
    cfg.fixed_curriculum_level = float(curriculum_level)

    env = AdvancedGeneralizedDoorEnv(cfg)
    env.set_curriculum_level(float(curriculum_level))
    return env, cfg


def load_obs_rms(run_dir = DEFAULT_RUN_DIR):
    vn_path = os.path.join(run_dir, "vecnormalize.pkl")
    if not os.path.exists(vn_path):
        return None
    with open(vn_path, "rb") as f:
        return pickle.load(f).obs_rms


def load_model(model_path = None, run_dir = DEFAULT_RUN_DIR):
    from stable_baselines3 import SAC

    if model_path is None:
        model_path = os.path.join(run_dir, "best_model.zip")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Modello non trovato in {model_path}. Esegui prima il training "
            f"(python -m close_generalized_v2.train_gen_v2 ...)."
        )

    model = SAC.load(model_path)
    model.policy.set_training_mode(False)
    return model


def norm_obs(obs, obs_rms):
    if obs_rms is None:
        return obs
    return np.clip((obs - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8), -10.0, 10.0)


def predict(model, obs, obs_rms, deterministic = True):
    return model.predict(norm_obs(obs, obs_rms), deterministic=deterministic)


def phase_idx_from_info(info):
    if info.get("fsm_phase", None) is not None:
        return int(info["fsm_phase"])

    name = str(info.get("fsm_phase_name", "REACH")).upper()
    for i, n in enumerate(PHASE_NAMES):
        if n in name:
            return i
    return 0