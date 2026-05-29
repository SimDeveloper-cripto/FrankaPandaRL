#!/usr/bin/env python3
# close_generalized_v2/fsm_v2.py
#
# AdaptiveFSM — Context-Sensitive Finite State Machine
#
# Implements proposal:
#   "FSM con Soglie Adattive (Parametrizzazione Contestuale)"
#
# Key literature:
#   [1]  Sutton, Precup & Singh (1999) "Between MDPs and Semi-MDPs"
#        → Options framework: FSM as 4 options with (initiation, policy, termination)
#   [2]  Konidaris & Barto (2009) "Skill Chaining"
#        → Learned option preconditions; motivation for adaptive thresholds
#   [13] ManipForce (2015) "Force-Based Manipulation Primitives"
#        → Force/contact signals as FSM transition triggers
#
# Differences vs close_generalized/env_gen.py (v1):
#   ┌──────────────────────────────┬──────────────────────┬────────────────────────────────┐
#   │ Element                      │ v1 (fixed)           │ v2 (adaptive)                  │
#   ├──────────────────────────────┼──────────────────────┼────────────────────────────────┤
#   │ dist_thresh_grasp            │ 0.020 m              │ 1.5 × radius + 0.005           │
#   │ grip_thresh (REACH→PUSH)     │ 0.65                 │ 0.75 − 0.10 × norm_friction    │
#   │ hold_steps (HOLD→RETREAT)    │ 60 steps (fixed)     │ base + Δt(stiffness)           │
#   │ retreat_pos direction        │ global [-0.13,0,.04] │ door_normal × dist + z_offset  │
#   │ FSM transition triggers      │ geometric thresholds │ + optional β-network (§3.5)    │
#   └──────────────────────────────┴──────────────────────┴────────────────────────────────┘

from __future__ import annotations


import numpy as np
from typing import Optional
from dataclasses import dataclass, field
from scipy.spatial.transform import Rotation as R_scipy


PHASE_REACH   = 0
PHASE_PUSH    = 1
PHASE_HOLD    = 2
PHASE_RETREAT = 3

PHASE_NAMES = {0: "REACH", 1: "PUSH", 2: "HOLD", 3: "RETREAT"}


@dataclass
class FSMState:
    phase                : int   = PHASE_REACH
    grasp_confirm_count  : int   = 0
    hold_closed_duration : int   = 0
    return_hold          : int   = 0
    min_door_angle       : Optional[float]      = None
    retreat_pos          : Optional[np.ndarray] = None
    has_grasp_bonus      : bool  = False
    events               : list  = field(default_factory=list)

    reach_steps          : int   = 0
    push_steps           : int   = 0
    hold_steps_total     : int   = 0
    retreat_steps        : int   = 0

    # Adaptive hold target (computed once when stiffness is known)
    target_hold_steps    : Optional[int] = None

    def reset(self) -> None:
        self.phase                = PHASE_REACH
        self.grasp_confirm_count  = 0
        self.hold_closed_duration = 0
        self.return_hold          = 0
        self.min_door_angle       = None
        self.retreat_pos          = None
        self.has_grasp_bonus      = False
        self.events               = []
        self.reach_steps          = 0
        self.push_steps           = 0
        self.hold_steps_total     = 0
        self.retreat_steps        = 0
        self.target_hold_steps    = None

    @property
    def one_hot(self) -> np.ndarray:
        """
        Returns [fsm_reach, fsm_push, fsm_hold, fsm_retreat] ∈ {0,1}^4
        """
        v             = np.zeros(4, dtype=np.float32)
        v[self.phase] = 1.0
        return v

    @property
    def phase_name(self) -> str:
        return PHASE_NAMES.get(self.phase, "?")


class AdaptiveFSM:
    """
    Context-sensitive Finite State Machine for door-closing manipulation.

    Implements §3.1:
    All transition thresholds depend on the current physical context
    (handle radius, friction, latch stiffness) instead of being hard-coded.

    References
    ----------
    [1]  Sutton, Precup & Singh (1999) — Options framework.
         Each phase is an option: (initiation_set, intra_option_policy, β).
    [2]  Konidaris & Barto (2009) — Learned preconditions.
         Adaptive thresholds approximate the optimal precondition function.
    [13] ManipForce (2015) — Force-based transition triggers.
         Physical contact state (is_physically_closed) as trigger for REACH→PUSH.
    """

    _GRASP_CONFIRM_STEPS = 5
    _GRIPPER_OPEN_THRESH = -0.85

    def __init__(self, cfg):
        """
        Parameters
        ----------
        cfg : TrainConfigV2
        """
        self.cfg   = cfg
        self.state = FSMState()

    # ── Adaptive threshold computation ────────────────────────────────────────

    def grasp_dist_thresh(self, handle_radius: float) -> float:
        """
        §3.1 — Adaptive grasp distance threshold.

        v1: fixed 0.020 m
        v2: 1.5 × radius + 0.005   [Konidaris & Barto 2009, §3.1]

        A larger handle requires the fingertips to close farther from centre.

        Parameters
        ----------
        handle_radius : float — current handle radius [m] from domain randomiser.
        """
        return self.cfg.fsm_grasp_dist_k_radius * handle_radius + self.cfg.fsm_grasp_dist_k_offset

    def grip_thresh(self, handle_friction: float) -> float:
        """
        §3.1 — Adaptive gripper-closure threshold for REACH→PUSH.

        v1: fixed 0.65
        v2: 0.75 − 0.10 × norm_friction   [ManipForce 2015, §3.1]

        Higher friction → stable grip with less closure    → lower threshold.
        Lower friction  → needs deeper closure to not slip → higher threshold.

        Parameters
        ----------
        handle_friction : float — current handle friction from domain randomiser.
        """
        f_min  = self.cfg.fsm_friction_min
        f_max  = self.cfg.fsm_friction_max
        norm_f = float(np.clip((handle_friction - f_min) / (f_max - f_min + 1e-8), 0.0, 1.0))
        return float(np.clip(
            self.cfg.fsm_grip_thresh_base - self.cfg.fsm_grip_thresh_k_fric * norm_f,
            0.50, 0.90
        ))

    def compute_target_hold_steps(
        self,
        control_freq        : int,
        latch_stiffness     : float,
        base_latch_stiffness: float,
    ) -> int:
        """
        §3.1 — Adaptive HOLD timer

        v1: fixed 60 steps (2.0 s @ 30 Hz)
        v2: base_steps + Δt × control_freq
            Δt = k_stiff × (stiff_max - current_stiff) / stiff_max   [§3.1]

        A stiffer latch spring returns faster → shorter hold needed.
        A softer spring needs more hold time for the latch to stabilise.

        References: ManipForce (2015), §3.1 of 07_nuova_generalizzazione.md.
        """
        base_steps   = int(control_freq * self.cfg.fsm_hold_time_base)
        stiff_norm   = float(np.clip(latch_stiffness / (base_latch_stiffness + 1e-8), 0.5, 2.0))

        extra_seconds = self.cfg.fsm_hold_k_stiffness * max(0.0, 1.0 - stiff_norm)
        extra_steps   = int(extra_seconds * control_freq)
        return base_steps + extra_steps

    @staticmethod
    def compute_retreat_pos(
        eef_pos         : np.ndarray,
        door_quat_mujoco: np.ndarray,  # wxyz (MuJoCo convention)
        retreat_dist    : float,
        retreat_z       : float,
    ) -> np.ndarray:
        """
        §3.1 — Retreat target aligned with door surface normal.

        v1: eef_pos + [-0.13, 0.0, 0.04]  (global X axis — wrong for rotated doors)
        v2: eef_pos + retreat_dist × door_normal + [0, 0, retreat_z]

        The optimal withdrawal direction is perpendicular to the door surface,
        independent of yaw randomisation.

        Parameters
        ----------
        door_quat_mujoco : np.ndarray — door body quaternion in MuJoCo wxyz format.
        """
        # Convert MuJoCo wxyz → scipy xyzw
        w, x, y, z  = door_quat_mujoco
        door_rot    = R_scipy.from_quat([x, y, z, w])
        door_mat    = door_rot.as_matrix()

        # First column of rotation matrix = door local X axis in world frame
        # For a door in the XY plane, this is the normal pointing toward the robot
        door_normal = door_mat[:, 0]
        door_normal = door_normal / (np.linalg.norm(door_normal) + 1e-8)

        retreat    = eef_pos + retreat_dist * door_normal
        retreat[2] += retreat_z
        return retreat.astype(np.float32)

    # ── Transition logic ──────────────────────────────────────────────────────

    def update(
        self,
        *,
        door_angle       : float,
        success_angle    : float,
        gripper_action   : float,
        dist_handle      : float,
        handle_radius    : float,
        handle_friction  : float,
        is_physically_closed: bool,
        gripper_width    : float,
        prev_angle       : float,
        control_freq     : int,
        door_qpos        : float,
        eef_pos          : np.ndarray,
        door_quat_mujoco : np.ndarray,
        latch_stiffness  : float,
        base_latch_stiffness: float,
        beta_probs       : Optional[dict] = None,  # §3.5 β-network outputs
    ) -> dict:
        """
        Run one FSM step and return a dict of transition events.

        Parameters
        ----------
        beta_probs : dict or None
            Optional outputs from BetaNetwork (§3.5).
            Keys: 'beta_reach', 'beta_push', 'beta_hold'.
            If None, pure threshold logic is used.

        Returns
        -------
        events : dict with boolean flags:
            'just_grasped'     — REACH → PUSH    transition this step
            'just_succeeded'   — PUSH  → HOLD    transition this step
            'just_hold_done'   — HOLD  → RETREAT transition this step
            'grasp_lost'       — PUSH  → REACH   transition this step
        """
        s      = self.state
        events = {
            "just_grasped"  : False,
            "just_succeeded": False,
            "just_hold_done": False,
            "grasp_lost"    : False,
        }

        # ── Compute adaptive thresholds ────────────────────────────────────────
        d_thresh = self.grasp_dist_thresh(handle_radius)          # §3.1
        g_thresh = self.grip_thresh(handle_friction)              # §3.1

        if s.target_hold_steps is None:
            s.target_hold_steps = self.compute_target_hold_steps(
                control_freq, latch_stiffness, base_latch_stiffness
            )                                                     # §3.1

        # ── Phase step counters ────────────────────────────────────────────────
        if   s.phase == PHASE_REACH  : s.reach_steps      += 1
        elif s.phase == PHASE_PUSH   : s.push_steps       += 1
        elif s.phase == PHASE_HOLD   : s.hold_steps_total += 1
        elif s.phase == PHASE_RETREAT: s.retreat_steps    += 1

        # ══════════════════════════════════════════════════════════════════════
        # PHASE 1 — REACH: wait for stable grasp
        # Transition condition (v2):
        #   dist < 1.5×radius + 0.005   (adaptive — §3.1, Konidaris 2009)
        #   grip > 0.75 − 0.10×norm_f   (adaptive — §3.1, ManipForce 2015)
        #   is_physically_closed        (same as v1)
        #   for ≥ 5 consecutive steps   (same as v1)
        # β-network override possible   (§3.5, Sutton 1999)
        # ══════════════════════════════════════════════════════════════════════
        if s.phase == PHASE_REACH:
            grasp_cond = (
                gripper_action > g_thresh
                and is_physically_closed
                and dist_handle < d_thresh
            )

            # §3.5 — optional β-network override
            if beta_probs is not None and "beta_reach" in beta_probs:
                grasp_cond = grasp_cond and (beta_probs["beta_reach"] > 0.5)

            if grasp_cond:
                s.grasp_confirm_count += 1
            else:
                s.grasp_confirm_count = 0

            if s.grasp_confirm_count >= self._GRASP_CONFIRM_STEPS:
                s.phase               = PHASE_PUSH
                s.grasp_confirm_count = 0
                s.reach_steps         = 0
                events["just_grasped"] = True
                s.events.append(f"REACH→PUSH (d={dist_handle:.3f}, g={gripper_action:.2f})")

        # ══════════════════════════════════════════════════════════════════════
        # PHASE 2 — PUSH: maintain grip, close the door
        # Transition to HOLD (v2):
        #   door_angle <= success_angle  (same as v1)
        #   gripper_action > 0.80        (same as v1 — deep grip required)
        #
        # Transition back to REACH (grasp lost):
        #   dynamic distance tolerance   (same as v1)
        # ══════════════════════════════════════════════════════════════════════
        elif s.phase == PHASE_PUSH:
            # Track minimum door angle (for potential-based reward §3.2)
            if s.min_door_angle is None:
                s.min_door_angle = door_angle
            elif door_angle < s.min_door_angle:
                s.min_door_angle = door_angle

            # PUSH → HOLD
            push_to_hold = door_angle <= success_angle and gripper_action > 0.80
            if beta_probs is not None and "beta_push" in beta_probs:
                push_to_hold = push_to_hold and (beta_probs["beta_push"] > 0.5)

            if push_to_hold:
                s.phase      = PHASE_HOLD
                s.push_steps = 0
                events["just_succeeded"] = True
                s.events.append(f"PUSH→HOLD (angle={door_angle:.3f})")
                return events

            # PUSH → REACH (grasp lost)  — dynamic tolerance (same as v1)
            door_speed         = abs(prev_angle - door_angle) * control_freq
            effective_lose_tol = float(np.clip(0.05 + door_speed * 0.5, 0.05, 0.12))
            near_latch         = door_angle < 0.05
            if near_latch:
                effective_lose_tol = 0.10

            grip_lost = gripper_action < g_thresh
            if not near_latch:
                grip_lost = grip_lost or not is_physically_closed
            dist_lost = dist_handle > effective_lose_tol

            if grip_lost or dist_lost:
                s.phase               = PHASE_REACH
                s.grasp_confirm_count = 0
                s.push_steps          = 0
                events["grasp_lost"]  = True
                reason = f"dist={dist_handle:.3f}" if dist_lost else f"grip={gripper_action:.2f}"
                s.events.append(f"PUSH→REACH ({reason})")

        # ══════════════════════════════════════════════════════════════════════
        # PHASE 3 — HOLD: keep door closed for adaptive duration
        # Timer (v2): base_steps + Δt(latch_stiffness)   [§3.1]
        # Bounce detection: same as v1 (soft timer reset)
        # ══════════════════════════════════════════════════════════════════════
        elif s.phase == PHASE_HOLD:
            is_closed = abs(door_qpos) < 0.03

            # Soft timer reset on bounce (same as v1)
            if not is_closed:
                penalty = int(abs(door_qpos) / 0.03 * 10)
                s.hold_closed_duration = max(0, s.hold_closed_duration - penalty)

            if is_closed and abs(door_qpos) < 0.04:
                s.hold_closed_duration += 1

            # HOLD → RETREAT
            hold_done = s.hold_closed_duration >= s.target_hold_steps
            if beta_probs is not None and "beta_hold" in beta_probs:
                hold_done = hold_done and (beta_probs["beta_hold"] > 0.5)

            if hold_done:
                # §3.1 — Retreat pos aligned with door normal (not global X)
                s.retreat_pos = self.compute_retreat_pos(
                    eef_pos,
                    door_quat_mujoco,
                    self.cfg.fsm_retreat_dist,
                    self.cfg.fsm_retreat_z_off,
                )
                s.phase      = PHASE_RETREAT
                s.hold_steps_total = 0
                events["just_hold_done"] = True
                s.events.append(
                    f"HOLD→RETREAT (dur={s.hold_closed_duration}, "
                    f"target={s.target_hold_steps})"
                )

        # PHASE_RETREAT Transitions are handled by the environment
        return events

    def reset(
        self,
        latch_stiffness      : float,
        base_latch_stiffness : float,
        control_freq         : int,
    ) -> None:

        self.state.reset()

        # Pre-compute adaptive hold duration for this episode
        self.state.target_hold_steps = self.compute_target_hold_steps(
            control_freq, latch_stiffness, base_latch_stiffness
        )