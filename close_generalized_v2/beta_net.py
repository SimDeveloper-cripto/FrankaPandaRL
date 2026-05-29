#!/usr/bin/env python3
# close_generalized_v2/beta_net.py
#
# BetaNetwork — Learned FSM Termination Functions
#
# Implements proposal:
#   "FSM a Terminazione Appresa (Learned Termination Functions)"
#
# Key literature:
#   [1]  Sutton, Precup & Singh (1999) "Between MDPs and Semi-MDPs"
#        → Options framework: β(s) is the termination function of an option.
#        → Our FSM phases are options with deterministic β; this module makes β stochastic.
#   [2]  Konidaris & Barto (2009) "Skill Chaining"
#        → β learned from data; preconditions of next option ≈ support of β.
#
# Design:
#   One small MLP per FSM phase (REACH, PUSH, HOLD), each outputting β ∈ (0,1).
#   β > 0.5 signals the FSM to transition from the current phase.
#   β is trained in the environment loop via a binary cross-entropy loss against
#   a heuristic "optimal transition" label derived from future returns.
#
# Status: DISABLED by default (cfg.use_beta_net = False).
#         Enable in Phase 4 of the implementation roadmap (§5).
#
# Differences vs close_generalized/env_gen.py (v1):
#   ┌──────────────────────────┬─────────────────────────┬───────────────────────────────────┐
#   │ Aspect                   │ v1                      │ v2 (with use_beta_net=True)       │
#   ├──────────────────────────┼─────────────────────────┼───────────────────────────────────┤
#   │ Transition: REACH→PUSH   │ dist<thresh (fixed)     │ β_reach(s) > 0.5 AND thresh       │
#   │ Transition: PUSH→HOLD    │ angle<success (fixed)   │ β_push(s)  > 0.5 AND condition    │
#   │ Transition: HOLD→RETREAT │ timer≥target (adaptive) │ β_hold(s)  > 0.5 AND timer        │
#   │ Generalisation           │ Requires re-tuning      │ Learned interpolation             │
#   │ Interpretability         │ High                    │ Reduced (inspect β output)        │
#   └──────────────────────────┴─────────────────────────┴───────────────────────────────────┘

from __future__ import annotations

import numpy as np
from typing import Optional, Dict


class BetaNetwork:
    """
    Three lightweight learned termination functions β_reach, β_push, β_hold.

    Each β_i : R^{n_i} → (0,1) maps a compact phase-specific feature vector
    to a termination probability.
    The FSM uses β as a soft gate on top of its hard threshold conditions.

    Implementation note:
    -  This class provides a pure-NumPy forward pass (no deep learning framework
       dependency) so that the module can run without PyTorch/TensorFlow.
    -  Weights are randomly initialised and NOT trained by default.
    -  To enable training, set cfg.use_beta_net = True and call update() each step.

    References
    ----------
    [1]  Sutton, Precup & Singh (1999) — β as option termination function.
    [2]  Konidaris & Barto (2009)      — learned β generalises preconditions.
    """

    # ── Feature dimensions for each β network ─────────────────────────────────
    # REACH: [dist_handle, handle_radius, handle_friction, gripper_width, gripper_action]
    _DIM_REACH = 5

    # PUSH:  [door_angle, door_speed, dist_handle, gripper_action, door_mass_norm]
    _DIM_PUSH  = 5

    # HOLD:  [hold_duration_norm, door_qpos, door_qvel, latch_stiffness_norm]
    _DIM_HOLD  = 4

    def __init__(self, cfg):
        self.cfg    = cfg
        self.hidden = cfg.beta_net_hidden   # default 64
        self.lr     = cfg.beta_net_lr       # default 1e-4
        self.reg    = cfg.beta_net_reg      # L2 regularisation

        # Initialise weights for each β network (Xavier uniform)
        rng = np.random.default_rng(seed=42)

        def _init(d_in, d_h):
            lim1 = np.sqrt(6.0 / (d_in + d_h))
            lim2 = np.sqrt(6.0 / (d_h + 1))
            return {
                "W1": rng.uniform(-lim1, lim1, (d_h, d_in)).astype(np.float32),
                "b1": np.zeros(d_h, dtype=np.float32),
                "W2": rng.uniform(-lim2, lim2, (1,  d_h)).astype(np.float32),
                "b2": np.zeros(1, dtype=np.float32),
            }

        self._params = {
            "reach": _init(self._DIM_REACH, self.hidden),
            "push" : _init(self._DIM_PUSH,  self.hidden),
            "hold" : _init(self._DIM_HOLD,  self.hidden),
        }
        self._grad_accum = {k: {pk: np.zeros_like(v) for pk, v in p.items()}
                            for k, p in self._params.items()}

    # ── Forward pass ──────────────────────────────────────────────────────────

    @staticmethod
    def _relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, x)

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -20.0, 20.0)))

    def _forward(self, key: str, x: np.ndarray) -> float:
        """Single forward pass: x → ReLU hidden → sigmoid output."""
        p  = self._params[key]
        h  = self._relu(p["W1"] @ x + p["b1"])
        y  = self._sigmoid(p["W2"] @ h + p["b2"])
        return float(y[0])

    # ── Feature extraction ────────────────────────────────────────────────────

    def _feat_reach(
        self,
        dist_handle    : float,
        handle_radius  : float,
        handle_friction: float,
        gripper_width  : float,
        gripper_action : float,
    ) -> np.ndarray:
        """
        Feature vector for β_reach.
        All values normalised to roughly [0,1].
        """
        return np.array([
            dist_handle / 0.30,             # ~0 when touching, 1 at 30cm
            handle_radius / 0.04,           # ~0.5 at base radius
            handle_friction / 2.0,          # normalised to [0,1]
            gripper_width / 0.08,           # 0=fully closed, 1=fully open
            (gripper_action + 1.0) / 2.0,   # [-1,+1] → [0,1]
        ], dtype=np.float32)

    def _feat_push(
        self,
        door_angle    : float,
        door_speed    : float,
        dist_handle   : float,
        gripper_action: float,
        norm_door_mass: float,
    ) -> np.ndarray:

        # Feature vector for β_push
        return np.array([
            door_angle / 0.5,                       # normalised angle
            float(np.clip(door_speed / 5.0, 0, 1)), # normalised speed
            dist_handle / 0.15,                     # closeness to handle
            (gripper_action + 1.0) / 2.0,           # [-1,+1] → [0,1]
            norm_door_mass,                         # from domain randomizer
        ], dtype=np.float32)

    def _feat_hold(
        self,
        hold_duration       : int,
        target_hold_steps   : int,
        door_qpos           : float,
        door_qvel           : float,
        norm_latch_stiffness: float,
    ) -> np.ndarray:

        # Feature vector for β_hold
        return np.array([
            float(np.clip(hold_duration / max(1, target_hold_steps), 0, 1)),
            abs(door_qpos) / 0.05,                  # how far from closed
            float(np.clip(abs(door_qvel) / 2.0, 0, 1)),
            norm_latch_stiffness,                   # from domain randomizer
        ], dtype=np.float32)

    # ── Public inference API ──────────────────────────────────────────────────

    def predict(
        self,
        *,
        dist_handle          : float,
        handle_radius        : float,
        handle_friction      : float,
        gripper_width        : float,
        gripper_action       : float,
        door_angle           : float,
        door_speed           : float,
        door_qpos            : float,
        door_qvel            : float,
        hold_duration        : int,
        target_hold_steps    : int,
        norm_latch_stiffness : float,
        norm_door_mass       : float,
    ) -> Dict[str, float]:
        """
        Compute β probabilities for all three phases.

        Returns
        -------
        dict with keys 'beta_reach', 'beta_push', 'beta_hold'.
        Each value ∈ (0,1); > 0.5 signals FSM to transition.
        """
        if not self.cfg.use_beta_net:
            return {"beta_reach": 1.0, "beta_push": 1.0, "beta_hold": 1.0}

        feat_r = self._feat_reach(
            dist_handle, handle_radius, handle_friction,
            gripper_width, gripper_action
        )
        feat_p = self._feat_push(
            door_angle, door_speed, dist_handle,
            gripper_action, norm_door_mass
        )
        feat_h = self._feat_hold(
            hold_duration, target_hold_steps,
            door_qpos, door_qvel, norm_latch_stiffness
        )

        return {
            "beta_reach": self._forward("reach", feat_r),
            "beta_push" : self._forward("push",  feat_p),
            "beta_hold" : self._forward("hold",  feat_h),
        }

    def update(
        self,
        key    : str,
        feat   : np.ndarray,
        label  : float,     # 1.0 = should have transitioned, 0.0 = should not
    ) -> float:
        """
        One gradient step of binary cross-entropy loss on the specified β network.

        L = -[y log β + (1-y) log(1-β)] + λ‖W‖²

        Parameters
        ----------
        key   : 'reach' | 'push' | 'hold'
        feat  : feature vector (pre-computed)
        label : float in {0.0, 1.0}

        Returns
        -------
        loss : float
        """
        if not self.cfg.use_beta_net:
            return 0.0

        p     = self._params[key]
        h     = self._relu(p["W1"] @ feat + p["b1"])
        y     = self._sigmoid(p["W2"] @ h + p["b2"])
        y_val = float(y[0])

        # Binary cross-entropy gradient  [Sutton 1999, §3.5]
        eps   = 1e-8
        dL_dy = -(label / (y_val + eps) - (1.0 - label) / (1.0 - y_val + eps))
        loss  = -(label * np.log(y_val + eps) + (1.0 - label) * np.log(1.0 - y_val + eps))

        # Backprop
        dy_dz2 = y * (1.0 - y)    # sigmoid derivative
        delta2  = dL_dy * dy_dz2  # [1, 1]

        dL_dW2 = delta2 * h[np.newaxis, :]  # [1, d_h]
        dL_db2 = delta2

        dL_dh   = (p["W2"].T * delta2).squeeze(axis=1) # [d_h]
        dL_dh  *= (h > 0).astype(np.float32)           # ReLU mask
        dL_dW1  = np.outer(dL_dh, feat)                # [d_h, d_in]
        dL_db1  = dL_dh

        # Gradient descent with L2 regularisation
        p["W2"] -= self.lr * (dL_dW2 + self.reg * p["W2"])
        p["b2"] -= self.lr * dL_db2
        p["W1"] -= self.lr * (dL_dW1 + self.reg * p["W1"])
        p["b1"] -= self.lr * dL_db1

        return float(loss)