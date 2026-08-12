#!/usr/bin/env python3
# open_generalized_v2/grasp_strategy.py
# [open_generalized_v2] Modulo RIUSATO dalla v2 di chiusura: fisica/grasp
# NON dipendono dal verso del task (apertura vs chiusura) → identici.
#
# MultiApproachGrasp — K-Candidate Grasp Direction Alignment
#
# Implements proposal:
#   "Grasp Strategy Adattiva (Multi-Approach)"
#
# Key literature:
#   [15] ten Pas et al. (2017) "Grasp Pose Detection in Point Clouds"
#        → Grasp as 6D pose; multiple valid antipodal grasps per object.
#
#   [13] ManipForce (2015) "Force-Based Manipulation Primitives"
#        → Force signals validate actual grasp quality independent of direction.
#
#   [14] Handa et al. (2020) "DexPilot"
#        → Contact representation; approach direction should be contact-compatible.
#
# Differences vs close_generalized/env_gen.py (v1):
#   ┌────────────────────────────────┬────────────────────────┬─────────────────────────────────┐
#   │ Aspect                         │ v1                     │ v2                              │
#   ├────────────────────────────────┼────────────────────────┼─────────────────────────────────┤
#   │ Approach directions            │ 1 (top-down)           │ K=3 (top, lateral-L, lateral-R) │
#   │ Alignment reward               │ |dot(eef_z, d)|        │ max(|dot(eef_z, dᵢ)|) over K    │
#   │ Best direction selection       │ N/A                    │ argmax — reported in obs        │
#   │ Obs enrichment                 │ No                     │ +K alignment values             │
#   └────────────────────────────────┴────────────────────────┴─────────────────────────────────┘

from __future__ import annotations


import numpy as np
from typing import Optional
from scipy.spatial.transform import Rotation as R_scipy


class MultiApproachGrasp:
    """
    Manages K candidate grasp approach directions and computes
    the best-alignment reward term.

    For a cylindrical door handle, three natural grasps exist:
        dir_top       : EEF approaches from above, z-axis pointing down
        dir_lateral_L : EEF approaches from the left  (−Y world)
        dir_lateral_R : EEF approaches from the right (+Y world)

    The alignment reward is:
        alignment = max_i |dot(eef_z, dir_i)|

    This allows the policy to choose the least obstructed approach
    without the reward explicitly prescribing which one to use.

    References
    ----------
    [15] ten Pas et al. (2017) — antipodal grasp quality metric.
    [14] Handa et al. (2020)   — contact-compatible approach direction.
    [13] ManipForce (2015)     — force-based approach validation.
    """

    def __init__(self, cfg):
        self.cfg          = cfg
        self.n_candidates = cfg.grasp_n_candidates

    def get_candidate_directions(
        self,
        handle_pos       : np.ndarray,
        eef_pos          : np.ndarray,
        door_quat_mujoco : Optional[np.ndarray] = None,
    ) -> list[np.ndarray]:
        """
        Returns K unit vectors representing candidate approach directions
        in world frame.

        Parameters
        ----------
        handle_pos       : [3] world-frame handle centre position
        eef_pos          : [3] current EEF position (used for adaptive lateral dir)
        door_quat_mujoco : [4] wxyz quaternion of door body, or None
        """
        candidates = []

        # Direction 1 — Top-Down (same as v1 alignment metric)
        dir_top = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        candidates.append(dir_top)

        if self.n_candidates >= 2:
            # Direction 2 — Lateral Left (−Y world)
            dir_lat_L = np.array([0.0, -1.0, 0.0], dtype=np.float32)

            # If door orientation is known, rotate lateral directions to door frame
            if door_quat_mujoco is not None:
                w, x, y, z = door_quat_mujoco
                door_rot   = R_scipy.from_quat([x, y, z, w])

                # Door local Y axis in world frame
                door_y_world = door_rot.apply(np.array([0.0, 1.0, 0.0]))
                dir_lat_L    = -door_y_world.astype(np.float32)

            candidates.append(dir_lat_L / (np.linalg.norm(dir_lat_L) + 1e-8))

        if self.n_candidates >= 3:
            # Direction 3 — Lateral Right (+Y world / +door_Y)
            if door_quat_mujoco is not None:
                candidates.append(-candidates[-1])  # symmetric to dir_lat_L
            else:
                candidates.append(np.array([0.0, 1.0, 0.0], dtype=np.float32))

        return candidates

    def compute_alignment(
        self,
        eef_quat         : np.ndarray,    # xyzw scipy convention
        handle_pos       : np.ndarray,
        eef_pos          : np.ndarray,
        door_quat_mujoco : Optional[np.ndarray] = None,
    ) -> tuple[float, int, list[float]]:
        """
        Compute max-alignment across K candidate directions.
            alignment = max_i  |dot(eef_z, dir_i)|

        Parameters
        ----------
        eef_quat : [4] EEF quaternion in xyzw (scipy) convention.

        Returns
        -------
        best_alignment  : float  — max |dot| across K directions
        best_idx        : int    — index of best direction
        all_alignments  : list   — per-direction alignment values
        """
        rmat   = R_scipy.from_quat(eef_quat).as_matrix()
        eef_z  = rmat[:, 2]  # gripper pointing axis

        dirs   = self.get_candidate_directions(handle_pos, eef_pos, door_quat_mujoco)
        aligns = [float(abs(np.dot(eef_z, d))) for d in dirs]

        best_idx       = int(np.argmax(aligns))
        best_alignment = aligns[best_idx]

        return best_alignment, best_idx, aligns

    def obs_features(
        self,
        eef_quat         : Optional[np.ndarray],
        handle_pos       : np.ndarray,
        eef_pos          : np.ndarray,
        door_quat_mujoco : Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Returns a [K + 1] - dim   observation vector for the policy:
                    [best_alignment, align_0, align_1, ..., align_{K - 1}]

        Providing per-direction alignments lets the policy learn which approach
        is feasible in each configuration.
        """
        if eef_quat is None:
            return np.zeros(self.n_candidates + 1, dtype=np.float32)

        best, best_idx, aligns = self.compute_alignment(
            eef_quat, handle_pos, eef_pos, door_quat_mujoco
        )
        return np.array([best] + aligns, dtype=np.float32)