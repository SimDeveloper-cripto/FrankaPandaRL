import os
import sys
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.train_close_config import TrainConfig
from close_generalized.env_gen import GeneralizedDoorEnv

class LogOverrideGripDoorEnv(GeneralizedDoorEnv):
    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32).copy()
        action = np.clip(action, -1.0, 1.0)

        # Action Smoothing (EMA)
        alpha = getattr(self.cfg, "action_smooth_alpha", 1.0)
        if alpha < 1.0:
            action = alpha * action + (1.0 - alpha) * self._prev_action

        if self._success_latched:
            is_ready_retreat = getattr(self, "_ready_to_retreat", False)
            if not is_ready_retreat:
                # HOLD phase
                duration = getattr(self, "_hold_closed_duration", 0)
                latch_qpos = float(self._rs_env.sim.data.qpos[self._rs_env.handle_qpos_addr])
                
                # Check if we should override
                if duration >= 45:
                    action[-1] = -1.0
                    print(f"Step {self._step_count} (duration={duration}): Enforcing gripper open. Latch before step: {latch_qpos:.4f}")
                else:
                    print(f"Step {self._step_count} (duration={duration}): Gripper action={action[-1]:.4f}. Latch before step: {latch_qpos:.4f}")
                
                # Scale arm actions
                action[:-1] *= 0.1
            else:
                latch_qpos = float(self._rs_env.sim.data.qpos[self._rs_env.handle_qpos_addr])
                print(f"Step {self._step_count}: RETREAT. Latch before step: {latch_qpos:.4f}")
                
                # RETREAT phase
                if self._retreat_pos is not None:
                    eef_site_id = self._rs_env.robots[0].eef_site_id
                    if isinstance(eef_site_id, dict):
                        site_id = eef_site_id.get('right', list(eef_site_id.values())[0])
                    else:
                        site_id = eef_site_id

                    eef_pos      = self._rs_env.sim.data.site_xpos[site_id]
                    dist_retreat = float(np.linalg.norm(eef_pos - self._retreat_pos))
                    returned     = dist_retreat < self.cfg.return_pos_tol

                    if returned or self._return_hold >= self.cfg.return_hold_steps:
                        action = np.zeros_like(action)  # Freeze completely!

        obs, _, rs_done, info = self._rs_env.step(action)
        self._step_count += 1

        door_angle            = self._get_door_angle()
        prev_angle            = float(self._prev_door_angle) if self._prev_door_angle is not None else door_angle
        self._prev_door_angle = door_angle

        just_succeeded = False
        if door_angle <= self._success_angle and not self._success_latched:
            self._success_latched = True
            just_succeeded        = True

        is_success                    = self._success_latched
        reward, terminated, truncated = self._calculate_reward(action, obs, rs_done, door_angle, prev_angle, just_succeeded)
        self._prev_action             = action.copy()

        door_qpos  = float(self._rs_env.sim.data.qpos[self._rs_env.hinge_qpos_addr])
        latch_qpos = float(self._rs_env.sim.data.qpos[self._rs_env.handle_qpos_addr])

        info                   = dict(info or {})
        info["is_success"]     = is_success
        info["door_angle"]     = door_angle
        info["door_qpos"]      = door_qpos
        info["latch_qpos"]     = latch_qpos
        info["ready_retreat"]  = getattr(self, "_ready_to_retreat", False)

        return self._flatten_obs(obs), reward, terminated, truncated, info

def run_test():
    cfg     = TrainConfig(run_dir="runs/close_gen", num_envs=1, horizon=500)
    raw_env = LogOverrideGripDoorEnv(cfg)
    raw_env.curriculum_level = 1.0

    vn_path     = "runs/close_gen/vecnormalize.pkl"
    raw_env_vec = DummyVecEnv([lambda: raw_env])
    vec_env     = VecNormalize.load(vn_path, raw_env_vec)

    vec_env.training    = False
    vec_env.norm_reward = False
    
    model = SAC.load("runs/close_gen/best_model.zip", env=vec_env)

    obs   = vec_env.reset()
    done  = False
    steps = 0
    while not done and steps < 150:
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = vec_env.step(action)
        done  = dones[0]
        steps += 1

if __name__ == "__main__":
    run_test()