import os
import sys
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.train_close_config import TrainConfig
from close_generalized.env_gen import GeneralizedDoorEnv

class WaitDoorEnv(GeneralizedDoorEnv):
    def step(self, action: np.ndarray):
        obs, reward, terminated, truncated, info = super().step(action)

        latch_qpos        = self._rs_env.sim.data.qpos[self._rs_env.handle_qpos_addr]
        latch_is_neutral  = abs(latch_qpos) < 0.15

        if getattr(self, "_ready_to_retreat", False) and not latch_is_neutral:
            self._ready_to_retreat = False
            info["ready_retreat"]  = False

            obs = self._flatten_obs(self._rs_env._get_observations() if hasattr(self._rs_env, "_get_observations") else {})

        return obs, reward, terminated, truncated, info

def run_test():
    cfg                      = TrainConfig(run_dir = "runs/close_gen", num_envs = 1, horizon = 500)
    raw_env                  = WaitDoorEnv(cfg)
    raw_env.curriculum_level = 1.0

    vn_path     = "runs/close_gen/vecnormalize.pkl"
    raw_env_vec = DummyVecEnv([lambda: raw_env])
    vec_env     = VecNormalize.load(vn_path, raw_env_vec)

    vec_env.training    = False
    vec_env.norm_reward = False
    
    model = SAC.load("runs/close_gen/best_model.zip", env = vec_env)

    successes = 0
    for ep in range(10):
        obs   = vec_env.reset()
        done  = False
        steps = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = vec_env.step(action)
            done  = dones[0]
            steps += 1

        is_success = infos[0].get("is_success", False)
        print(f"Episode {ep+1} - Steps: {steps} - Success: {is_success}")
        if is_success:
            successes += 1

    print(f"Total Success Rate with Wait Logic: {successes / 10 * 100:.1f}%")

if __name__ == "__main__":
    run_test()