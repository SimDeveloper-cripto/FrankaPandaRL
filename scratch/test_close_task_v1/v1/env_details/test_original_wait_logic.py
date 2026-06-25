import os
import sys
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.train_close_config import TrainConfig
from close_generalized.env_gen import GeneralizedDoorEnv

_GRIPPER_CLOSE_THRESH = 0.65
_GRIPPER_OPEN_THRESH = -0.85

class ExactWaitDoorEnv(GeneralizedDoorEnv):
    def _calculate_reward(self, action, obs, rs_done, door_angle, prev_angle, just_succeeded):
        # We override _calculate_reward to implement the exact wait logic from 517e021c
        
        base_reward, terminated, truncated = super(GeneralizedDoorEnv, self)._calculate_reward(
            action, obs, rs_done, door_angle, prev_angle, just_succeeded
        )

        eef_pos    = obs.get("robot0_eef_pos", np.zeros(3))
        handle_pos = obs.get("handle_pos", obs.get("door_handle_pos", eef_pos))

        dist_handle = np.linalg.norm(eef_pos - handle_pos)
        dist_xy     = np.linalg.norm(eef_pos[:2] - handle_pos[:2])
        height_diff = eef_pos[2] - handle_pos[2]

        door_qpos  = self._rs_env.sim.data.qpos[self._rs_env.hinge_qpos_addr]
        is_closed  = abs(door_qpos) < 0.03
        latch_qpos = float(self._rs_env.sim.data.qpos[self._rs_env.handle_qpos_addr])

        gripper_action = action[-1] if action is not None else 0.0

        gripper_qpos = obs.get("robot0_gripper_qpos")
        if gripper_qpos is not None:
            gripper_width        = np.sum(np.abs(gripper_qpos))
            handle_diameter      = getattr(self, "_current_handle_radius", 0.02) * 2.0
            is_physically_closed = (gripper_width <= handle_diameter + 0.025) and (gripper_width >= 0.015)
        else:
            is_physically_closed = gripper_action > _GRIPPER_CLOSE_THRESH
            gripper_width        = 0.0

        alignment      = 0.0
        flat_alignment = 0.0
        eef_quat       = obs.get("robot0_eef_quat")

        if eef_quat is not None:
            delta_pos  = handle_pos - eef_pos
            norm_delta = np.linalg.norm(delta_pos)
            if norm_delta > 0:
                dir_to_handle  = delta_pos / norm_delta

                import robosuite.utils.transform_utils as T

                mat            = T.quat2mat(eef_quat)
                z_axis         = mat[:, 2]
                alignment      = float(np.dot(z_axis, dir_to_handle))
                x_axis         = mat[:, 0]
                flat_alignment = float(abs(np.dot(x_axis, np.array([0, 0, 1]))))

        rew_info = {}

        # FASE 1 & 2 reward logic remains the same (handled by calling super or implementing here)
        if not self._success_latched:
            # Let the original environment handle REACH & PUSH reward calculation
            return super()._calculate_reward(action, obs, rs_done, door_angle, prev_angle, just_succeeded)

        # FASE 3 & 4 with wait logic from 517e021c
        rew_info["base"] = base_reward
        rew_info["hold"] = 0.0

        is_waiting_latch = getattr(self, "_hold_closed_duration", 0) >= int(self.cfg.control_freq * 2.0)

        if not getattr(self, "_ready_to_retreat", False):
            # Penalizza se perde la presa fisica (disabilitato se stiamo aspettando il latch)
            if not is_physically_closed and not is_waiting_latch:
                rew_info["hold_slip"] = -5.0

            # Penalità sulla velocità angolare della porta
            door_qvel = float(self._rs_env.sim.data.qvel[self._door_hinge_dof_adr])
            if abs(door_qvel) > 0.01:
                rew_info["hold_veldamp"] = -25.0 * abs(door_qvel)

            # Soft timer reset
            if not is_closed:
                rew_info["hold_bounce"] = -20.0 * abs(door_qpos)
                penalty_steps = int(abs(door_qpos) / 0.03 * 10)
                self._hold_closed_duration = max(0, self._hold_closed_duration - penalty_steps)

        if is_closed:
            rew_info["hold"] += 1.0 - abs(door_qpos)
            if abs(door_qpos) < 0.04:
                control_freq      = self.cfg.control_freq
                target_hold_steps = int(control_freq * 2.0)

                if not hasattr(self, "_hold_closed_duration"):
                    self._hold_closed_duration = 0

                latch_is_neutral = abs(latch_qpos) < 0.15

                if self._hold_closed_duration < target_hold_steps or not latch_is_neutral:
                    is_waiting = (self._hold_closed_duration >= target_hold_steps)
                    
                    if is_waiting and not latch_is_neutral:
                        rew_info["latch_wait"] = -10.0 * abs(latch_qpos)
                    else:
                        self._hold_closed_duration += 1
                        
                    self._ready_to_retreat = False

                    if is_waiting:
                        if gripper_action < _GRIPPER_OPEN_THRESH:
                            rew_info["wait_grip"] = 2.0
                        else:
                            rew_info["wait_grip"] = -2.0 * abs(gripper_action + 1.0)
                    else:
                        if gripper_action > _GRIPPER_CLOSE_THRESH:
                            rew_info["hold_grip"] = 1.0
                        else:
                            rew_info["hold_grip"] = -2.0 * abs(gripper_action - _GRIPPER_CLOSE_THRESH)

                        if gripper_action < 0.0:
                            rew_info["hold_drop_pen"] = -10.0 * abs(gripper_action)

                    joint_vel = obs.get("robot0_joint_vel")
                    if joint_vel is not None:
                        rew_info["hold_jnt_freeze"] = -1.0 * np.linalg.norm(joint_vel)

                    if action is not None:
                        action_norm = np.linalg.norm(action[:-1])
                        if action_norm < 0.05:
                            rew_info["hold_act"] = 1.0
                        else:
                            rew_info["hold_act"] = -2.0 * action_norm

                    rew_info["hold_flat"] = -2.0 * flat_alignment

                    if dist_handle > 0.06:
                        rew_info["hold_dist"] = -3.0 * (dist_handle - 0.06)
                else:
                    # Transizione a RETREAT
                    if not getattr(self, "_ready_to_retreat", False):
                        self._ready_to_retreat = True
                        self._retreat_pos = eef_pos + np.array([-0.13, 0.0, 0.04], dtype=np.float32)

                    if gripper_action < _GRIPPER_OPEN_THRESH:
                        rew_info["ret_grip"] = 2.0
                    else:
                        rew_info["ret_grip"] = -1.0 * abs(gripper_action + 1.0)

                    if action is not None:
                        rew_info["ret_rot"] = -3.0 * np.linalg.norm(action[3:6])

                        if dist_handle < 0.12:
                            rew_info["ret_lat"] = -5.0 * abs(action[1])
                            if action[2] < 0:
                                rew_info["ret_down"] = -5.0 * abs(action[2])

                        if hasattr(self, "_retreat_pos"):
                            dist_to_target = float(np.linalg.norm(eef_pos - self._retreat_pos))

                            if dist_to_target > 0.02:
                                dir_to_target        = (self._retreat_pos - eef_pos)
                                dir_norm             = dir_to_target / (dist_to_target + 1e-6)
                                action_alignment     = float(np.dot(action[:3], dir_norm))
                                rew_info["ret_dir"]  = 3.0 * action_alignment
                                perp                 = action[:3] - action_alignment * dir_norm
                                rew_info["ret_perp"] = -2.0 * float(np.linalg.norm(perp))
                            else:
                                rew_info["ret_freeze"] = -20.0 * np.linalg.norm(action[:-1])

                            joint_vel = obs.get("robot0_joint_vel")
                            if joint_vel is not None:
                                freeze_weight = float(np.clip(1.0 - dist_to_target / 0.15, 0.1, 1.0))
                                rew_info["ret_jnt_prog"] = -5.0 * freeze_weight * np.linalg.norm(joint_vel)

                    rew_info["latch_ret"] = -1.0 * abs(latch_qpos)
        else:
            if not getattr(self, "_ready_to_retreat", False):
                self._hold_closed_duration = max(0, self._hold_closed_duration - 1)

        reward = 0.0
        for v in rew_info.values():
            reward += v

        if terminated:
            latch_is_neutral = abs(latch_qpos) < 0.08
            door_is_closed   = abs(door_qpos) < 0.03
            if not latch_is_neutral or not door_is_closed:
                terminated = False
                reward     -= 500.0
                reward     = float(np.clip(reward, -100.0, 100.0))

        return reward, terminated, truncated

def run_test():
    cfg     = TrainConfig(run_dir="runs/close_gen", num_envs=1, horizon=500)
    raw_env = ExactWaitDoorEnv(cfg)
    raw_env.curriculum_level = 1.0

    vn_path     = "runs/close_gen/vecnormalize.pkl"
    raw_env_vec = DummyVecEnv([lambda: raw_env])
    vec_env     = VecNormalize.load(vn_path, raw_env_vec)

    vec_env.training    = False
    vec_env.norm_reward = False
    
    model = SAC.load("runs/close_gen/best_model.zip", env=vec_env)

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
            
    print(f"Total Success Rate: {successes / 10 * 100:.1f}%")

if __name__ == "__main__":
    run_test()