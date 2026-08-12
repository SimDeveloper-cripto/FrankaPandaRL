#!/usr/bin/env python3
# open_generalized_v2/domain_rand_v2.py
# [open_generalized_v2] Modulo RIUSATO dalla v2 di chiusura: fisica/grasp
# NON dipendono dal verso del task (apertura vs chiusura) → identici.
#
# ExtendedDomainRandomizer — Extended Physics Randomization
#
# Implements proposal:
#   "Generalizzazione Fisica del Task (Nuove Dimensioni di Randomizzazione)"
#
# Key literature:
#   [8]  Tobin et al. (2017) "Domain Randomization for Transferring Deep Neural Networks"
#        → Foundation: broad physics randomization → robust policies
#   [17] Zhao et al. (2020) "Sim-to-Real Transfer in Deep RL for Robotics"
#        → Physics randomization (stiffness, damping, mass) is the most impactful
#          factor for sim-to-real transfer in manipulation tasks
#   [9]  Mehta et al. (2020) "Active Domain Randomization"
#        → Adapt randomization distribution based on policy difficulty
#   [13] ManipForce (2015) — Force-dependent stiffness/damping effects on contact
#
# Differences vs close_generalized/env_gen.py (v1):
#   ┌─────────────────────────────────┬──────────────────────┬──────────────────────────────┐
#   │ Randomized Parameter            │ v1                   │ v2                           │
#   ├─────────────────────────────────┼──────────────────────┼──────────────────────────────┤
#   │ Handle radius                   │ ×0.7–1.4 base        │ ×0.7–1.4 base (same)         │
#   │ Handle length                   │ ×0.8–1.2 base        │ ×0.8–1.2 base (same)         │
#   │ Handle friction                 │ ×0.3–1.2 base        │ ×0.3–1.2 base (same)         │
#   │ Door position XY                │ ±15 cm × curriculum  │ same                         │
#   │ Door yaw                        │ ±17.2° × curriculum  │ same                         │
#   │ Latch stiffness                 │ Fixed                │ ×0.5–2.0 base  [NEW §3.4]    │
#   │ Hinge damping                   │ Fixed                │ ×0.3–1.5 base  [NEW §3.4]    │
#   │ Door body mass                  │ Fixed                │ ×0.5–2.0 base  [NEW §3.4]    │
#   └─────────────────────────────────┴──────────────────────┴──────────────────────────────┘
#
# New physics parameters are exposed in the observation (+3 features):
#   [norm_latch_stiffness, norm_hinge_damping, norm_door_mass]

from __future__ import annotations

import numpy as np
from typing import Optional


class ExtendedDomainRandomizer:
    """
    The current episode's physics parameters are:
        - Stored as instance attributes (readable by FSM and Reward)
        - Exposed in the observation as normalized values in [0,1]

    References
    ----------
    [8]  Tobin et al. (2017) — domain randomization foundation.
    [17] Zhao et al. (2020)  — physics randomization for sim-to-real.
    [9]  Mehta et al. (2020) — active domain randomization (future extension).
    """

    def __init__(self, cfg, sim_model):
        self.cfg   = cfg
        self.model = sim_model

        # ── Discover MuJoCo element IDs ──────────────────────────────────────
        self.handle_geom_id: Optional[int] = None
        self.door_body_id  : Optional[int] = None
        self.latch_joint_id: Optional[int] = None
        self.hinge_joint_id: Optional[int] = None

        for i, name in enumerate(sim_model.geom_names):
            if "handle" in name.lower():
                self.handle_geom_id = i
                break

        try:
            self.door_body_id = sim_model.body_name2id("Door_main")
        except Exception:
            self.door_body_id = None

        # Find latch and hinge joints by name pattern
        for i, name in enumerate(sim_model.joint_names):
            n = name.lower()
            if "latch" in n:
                self.latch_joint_id = i
            if "hinge" in n and "latch" not in n:
                self.hinge_joint_id = i

        # ── Capture base physics values ───────────────────────────────────────
        self.base_handle_radius   = 0.02
        self.base_handle_length   = 0.08
        self.base_friction        = np.array([0.8])
        self.base_door_pos        = None
        self.base_door_quat       = None
        self.base_latch_stiffness = None
        self.base_latch_damping   = None
        self.base_hinge_damping   = None
        self.base_door_mass       = None

        if self.handle_geom_id is not None:
            self.base_friction = sim_model.geom_friction[self.handle_geom_id].copy()

        if self.door_body_id is not None:
            self.base_door_pos  = sim_model.body_pos[self.door_body_id].copy()
            self.base_door_quat = sim_model.body_quat[self.door_body_id].copy()

        # ── Resolve DOF addresses ─────────────────────────────────────────────
        self.latch_dof_adr: Optional[int] = (
            int(sim_model.jnt_dofadr[self.latch_joint_id]) if self.latch_joint_id is not None else None
        )
        self.hinge_dof_adr: Optional[int] = (
            int(sim_model.jnt_dofadr[self.hinge_joint_id]) if self.hinge_joint_id is not None else None
        )

        # Latch joint stiffness (jnt_stiffness IS per-joint -> joint id is correct)
        if self.latch_joint_id is not None:
            self.base_latch_stiffness = float(sim_model.jnt_stiffness[self.latch_joint_id])
            if self.latch_dof_adr is not None:
                self.base_latch_damping = float(sim_model.dof_damping[self.latch_dof_adr])

        # Hinge joint damping (per-DOF index)
        if self.hinge_dof_adr is not None:
            self.base_hinge_damping = float(sim_model.dof_damping[self.hinge_dof_adr])

        # Door body mass
        if self.door_body_id is not None:
            self.base_door_mass = float(sim_model.body_mass[self.door_body_id])

        # ── Current episode values (updated at each reset) ────────────────────
        self.current_handle_radius   : float = self.base_handle_radius
        self.current_handle_friction : float = float(self.base_friction[0])
        self.current_latch_stiffness : float = self.base_latch_stiffness or 1.0
        self.current_hinge_damping   : float = self.base_hinge_damping or 0.1
        self.current_door_mass       : float = self.base_door_mass or 1.0

    # ── Public API ─────────────────────────────────────────────────────────────

    def randomize_episode(self, curriculum_level: float) -> None:
        self._randomize_handle()
        self._randomize_latch_stiffness()
        self._randomize_hinge_damping()
        self._randomize_door_mass()

    def obs_features(self) -> np.ndarray:
        """
        Returns a 3-dim observation vector of normalised physics parameters:
            [norm_latch_stiffness, norm_hinge_damping, norm_door_mass]

        All values normalized to [0,1] within their randomisation range.
        The policy observes these to adapt its behavior to the current physics.

        References: Tobin et al. (2017) §3.4 — physics params in observation.
        """

        # Latch stiffness: range [0.5, 2.0] × base
        base_s       = self.base_latch_stiffness or 1.0
        s_min, s_max = base_s * 0.5, base_s * 2.0
        norm_s       = float(np.clip(
            (self.current_latch_stiffness - s_min) / (s_max - s_min + 1e-8),
            0.0, 1.0
        ))

        # Hinge damping: range [0.3, 1.5] × base
        base_d       = self.base_hinge_damping or 0.1
        d_min, d_max = base_d * 0.3, base_d * 1.5
        norm_d       = float(np.clip(
            (self.current_hinge_damping - d_min) / (d_max - d_min + 1e-8),
            0.0, 1.0
        ))

        # Door mass: range [0.5, 2.0] × base
        base_m       = self.base_door_mass or 1.0
        m_min, m_max = base_m * 0.5, base_m * 2.0
        norm_m       = float(np.clip(
            (self.current_door_mass - m_min) / (m_max - m_min + 1e-8),
            0.0, 1.0
        ))

        return np.array([norm_s, norm_d, norm_m], dtype=np.float32)

    # ── Private randomization methods ─────────────────────────────────────────

    def _randomize_handle(self) -> None:
        """
        Handle Geometry and Friction randomization (same as v1).
        """
        if self.handle_geom_id is None:
            self.current_handle_radius   = self.base_handle_radius
            self.current_handle_friction = float(self.base_friction[0])
            return

        r_scale = np.random.uniform(0.7, 1.4)
        l_scale = np.random.uniform(0.8, 1.2)

        self.current_handle_radius = self.base_handle_radius * r_scale

        geom_size = self.model.geom_size[self.handle_geom_id]
        if geom_size is not None:
            geom_size[0] = self.current_handle_radius
            geom_size[1] = self.base_handle_length * l_scale

        f_scale = np.random.uniform(0.3, 1.2)
        base_f  = float(self.base_friction[0])
        new_f   = float(np.clip(base_f * f_scale, 0.05, 2.0))

        self.model.geom_friction[self.handle_geom_id][0] = new_f
        self.current_handle_friction = new_f

    def _randomize_latch_stiffness(self) -> None:
        """
        §3.4 — Randomize latch joint spring stiffness.

        Range: [0.5 × base, 2.0 × base]

        Effect on FSM: stiffer spring → latch returns faster → shorter HOLD time needed

        Ref: Zhao et al. (2020) §3.4; ManipForce (2015).
        """
        if not self.cfg.rand_latch_stiffness or self.latch_joint_id is None:
            return

        base  = self.base_latch_stiffness or 1.0
        scale = np.random.uniform(
            self.cfg.rand_latch_stiffness_min,
            self.cfg.rand_latch_stiffness_max,
        )
        new_stiff                                     = base * scale
        self.model.jnt_stiffness[self.latch_joint_id] = new_stiff
        self.current_latch_stiffness                  = float(new_stiff)

    def _randomize_hinge_damping(self) -> None:
        """
        §3.4 — Randomize door hinge joint damping.

        Range: [0.3 × base, 1.5 × base]

        Low damping  → door bounces more → hold_veldamp reward term is more critical.
        High damping → door stops quickly → policy can use less force.

        Ref: Tobin et al. (2017); Zhao et al. (2020) §3.4.
        """
        if not self.cfg.rand_hinge_damping or self.hinge_dof_adr is None:
            return

        base  = self.base_hinge_damping or 0.1
        scale = np.random.uniform(
            self.cfg.rand_hinge_damping_min,
            self.cfg.rand_hinge_damping_max,
        )
        new_damp                                   = base * scale
        self.model.dof_damping[self.hinge_dof_adr] = new_damp
        self.current_hinge_damping                 = float(new_damp)

    def _randomize_door_mass(self) -> None:
        """
        §3.4 — Randomize door body mass.

        Range: [0.5 × base, 2.0 × base]
        Heavier door → more resistance during PUSH → policy must apply more force.

        Ref: Zhao et al. (2020) §3.4 — mass is a key sim-to-real gap factor.
        """
        if not self.cfg.rand_door_mass or self.door_body_id is None:
            return

        base  = self.base_door_mass or 1.0
        scale = np.random.uniform(
            self.cfg.rand_door_mass_min,
            self.cfg.rand_door_mass_max,
        )
        new_mass                                = base * scale
        self.model.body_mass[self.door_body_id] = new_mass
        self.current_door_mass                  = float(new_mass)