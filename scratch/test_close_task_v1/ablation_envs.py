#!/usr/bin/env python3
# scratch/test_close_task_v1/ablation_envs.py


from __future__ import annotations

import numpy as np
from _common import GeneralizedDoorEnv

_GRIPPER_CLOSE_THRESH = 0.65
_GRIPPER_OPEN_THRESH  = -0.85


def _eef_pos(env):
    eef_site_id = env._rs_env.robots[0].eef_site_id
    site_id = (eef_site_id.get("right", list(eef_site_id.values())[0])
               if isinstance(eef_site_id, dict) else eef_site_id)
    return np.asarray(env._rs_env.sim.data.site_xpos[site_id], float)


def _smooth(env, action):
    action = np.asarray(action, np.float32).copy()
    action = np.clip(action, -1.0, 1.0)
    alpha  = getattr(env.cfg, "action_smooth_alpha", 1.0)
    if alpha < 1.0:
        action = alpha * action + (1.0 - alpha) * env._prev_action
    return action


def _finish_step(env, action, obs, rs_done, rs_info=None, gripper_for_latch=None):
    door_angle = env._get_door_angle()
    prev_angle = float(env._prev_door_angle) if env._prev_door_angle is not None else door_angle
    env._prev_door_angle = door_angle

    just_succeeded = False
    if door_angle <= env._success_angle and not env._success_latched:
        gate = True if gripper_for_latch is None else (gripper_for_latch > 0.80)
        if gate:
            env._success_latched = True
            just_succeeded       = True

    is_success                    = env._success_latched
    reward, terminated, truncated = env._calculate_reward(action, obs, rs_done, door_angle, prev_angle, just_succeeded)
    env._prev_action              = action.copy()

    door_qpos  = float(env._rs_env.sim.data.qpos[env._rs_env.hinge_qpos_addr])
    latch_qpos = float(env._rs_env.sim.data.qpos[env._rs_env.handle_qpos_addr])

    info                   = dict(rs_info or {})
    info["is_success"]     = is_success
    info["door_angle"]     = door_angle
    info["door_qpos"]      = door_qpos
    info["latch_qpos"]     = latch_qpos
    info["ready_retreat"]  = getattr(env, "_ready_to_retreat", False)

    return env._flatten_obs(obs), reward, terminated, truncated, info


# ─────────────────────────────────────────────────────────────────────────────
# Variante A — Freeze condizionato al latch (da test_freeze_logic.py)
# ─────────────────────────────────────────────────────────────────────────────
class FreezeDoorEnv(GeneralizedDoorEnv):
    def step(self, action):
        action = _smooth(self, action)
        if self._success_latched:
            if getattr(self, "_ready_to_retreat", False):
                latch_qpos = self._rs_env.sim.data.qpos[self._rs_env.handle_qpos_addr]
                if abs(latch_qpos) >= 0.15:
                    action[:-1] = 0.0
                elif self._retreat_pos is not None:
                    dist = float(np.linalg.norm(_eef_pos(self) - self._retreat_pos))
                    if dist < self.cfg.return_pos_tol or self._return_hold >= self.cfg.return_hold_steps:
                        action = np.zeros_like(action)
            else:
                action[:-1] *= 0.1
        obs, _, rs_done, rs_info = self._rs_env.step(action)
        self._step_count += 1
        return _finish_step(self, action, obs, rs_done, rs_info=rs_info)


# ─────────────────────────────────────────────────────────────────────────────
# Variante B — Hold freeze totale del braccio (da test_hold_freeze.py)
# ─────────────────────────────────────────────────────────────────────────────
class HoldFreezeDoorEnv(GeneralizedDoorEnv):
    def step(self, action):
        action = _smooth(self, action)
        if self._success_latched:
            if not getattr(self, "_ready_to_retreat", False):
                action[:-1] = 0.0
            elif self._retreat_pos is not None:
                dist = float(np.linalg.norm(_eef_pos(self) - self._retreat_pos))
                if dist < self.cfg.return_pos_tol or self._return_hold >= self.cfg.return_hold_steps:
                    action = np.zeros_like(action)
        obs, _, rs_done, rs_info = self._rs_env.step(action)
        self._step_count += 1
        return _finish_step(self, action, obs, rs_done, rs_info=rs_info)


# ─────────────────────────────────────────────────────────────────────────────
# Variante C — Hold freeze + gate di presa profonda al latch (da test_hold_freeze_grip.py)
# ─────────────────────────────────────────────────────────────────────────────
class HoldFreezeGripDoorEnv(GeneralizedDoorEnv):
    def step(self, action):
        action = _smooth(self, action)
        gripper_action = float(action[-1])
        if self._success_latched:
            if not getattr(self, "_ready_to_retreat", False):
                action[:-1] = 0.0
            elif self._retreat_pos is not None:
                dist = float(np.linalg.norm(_eef_pos(self) - self._retreat_pos))
                if dist < self.cfg.return_pos_tol or self._return_hold >= self.cfg.return_hold_steps:
                    action = np.zeros_like(action)
        obs, _, rs_done, rs_info = self._rs_env.step(action)
        self._step_count += 1
        return _finish_step(self, action, obs, rs_done, rs_info=rs_info, gripper_for_latch=gripper_action)


# ─────────────────────────────────────────────────────────────────────────────
# Variante D — Wait: annulla la transizione a RETREAT se il latch non è neutro
#              (da test_wait_logic.py)
# ─────────────────────────────────────────────────────────────────────────────
class WaitDoorEnv(GeneralizedDoorEnv):
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        latch_qpos = self._rs_env.sim.data.qpos[self._rs_env.handle_qpos_addr]
        if getattr(self, "_ready_to_retreat", False) and abs(latch_qpos) >= 0.15:
            self._ready_to_retreat = False
            info["ready_retreat"] = False
        return obs, reward, terminated, truncated, info


# ─────────────────────────────────────────────────────────────────────────────
# Variante E — Override grip: forza l'apertura del gripper in HOLD dopo 45 step
#              (da test_override_grip.py, senza i print)
# ─────────────────────────────────────────────────────────────────────────────
class OverrideGripDoorEnv(GeneralizedDoorEnv):
    def step(self, action):
        action = _smooth(self, action)
        if self._success_latched:
            if not getattr(self, "_ready_to_retreat", False):
                if getattr(self, "_hold_closed_duration", 0) >= 45:
                    action[-1] = -1.0
                action[:-1] *= 0.1
            elif self._retreat_pos is not None:
                dist = float(np.linalg.norm(_eef_pos(self) - self._retreat_pos))
                if dist < self.cfg.return_pos_tol or self._return_hold >= self.cfg.return_hold_steps:
                    action = np.zeros_like(action)
        obs, _, rs_done, rs_info = self._rs_env.step(action)
        self._step_count += 1
        return _finish_step(self, action, obs, rs_done, rs_info=rs_info)


# Registro nome → classe (BASELINE = env reale)
VARIANTS = {
    "baseline"         : GeneralizedDoorEnv,
    "freeze_cond_latch": FreezeDoorEnv,
    "hold_freeze"      : HoldFreezeDoorEnv,
    "hold_freeze_grip" : HoldFreezeGripDoorEnv,
    "wait_latch"       : WaitDoorEnv,
    "override_grip"    : OverrideGripDoorEnv,
}