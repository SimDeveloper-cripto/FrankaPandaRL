#!/usr/bin/env python3
# scratch/test_close_task_v2/_common.py

"""
Rif. metodologici: Henderson et al. 2018 (riproducibilità); Patterson et al. 2024
(disegno controllato); Tobin et al. 2017 / Zhao et al. 2020 (randomizzazione fisica).
"""

from __future__ import annotations

import os
import sys
import json
import pickle
import contextlib
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# JSON robusto ai tipi numpy
# ─────────────────────────────────────────────────────────────────────────────
def json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def dump_json(obj, path) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=json_default)


# ─────────────────────────────────────────────────────────────────────────────
# Path / import robusti (indipendenti dalla CWD)
# ─────────────────────────────────────────────────────────────────────────────
def find_repo_root(start: Optional[str] = None) -> str:
    """Risale finché trova la cartella-pacchetto `close_generalized_v2`; inserisce la
    root in sys.path (così `from close_generalized_v2.* import ...` e il bare
    `import train_close` di env_v2 si risolvono)."""
    here = os.path.abspath(start or os.path.dirname(__file__))
    cur = here
    root = None
    for _ in range(8):
        if os.path.isdir(os.path.join(cur, "close_generalized_v2")):
            root = cur
            break
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    if root is None:
        root = os.path.abspath(os.path.join(here, "..", ".."))
    for p in (root, os.path.join(root, "close_generalized_v2")):
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)
    return root


REPO_ROOT = find_repo_root()

# Modelli forniti dall'utente (curriculum 0 = posa fissa, 1 = posa variabile)
RUN_DIR_C0 = "runs/close_gen_v2_curriculum_0_new_110626"
RUN_DIR_C1 = "runs/close_gen_v2_curriculum_1_new_110626"
DEFAULT_RUN_DIR = RUN_DIR_C1

# Coppie (etichetta, run_dir, curriculum) usate dall'orchestratore
MODEL_SPECS = [
    ("curr0_posa_fissa", RUN_DIR_C0, 0.0),
    ("curr1_posa_variabile", RUN_DIR_C1, 1.0),
]

PHASE_NAMES = ["REACH", "PUSH", "HOLD", "RETREAT"]

from close_generalized_v2.config_v2 import TrainConfigV2            # noqa: E402
from close_generalized_v2.env_v2 import AdvancedGeneralizedDoorEnv  # noqa: E402
from stable_baselines3 import SAC                                   # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Costruzione ambiente RAW + modello (pattern v2: obs_rms manuale)
# ─────────────────────────────────────────────────────────────────────────────
def make_raw_env(curriculum_level: float, run_dir: str = DEFAULT_RUN_DIR,
                 horizon: int = 500, cfg_overrides: Optional[dict] = None):
    """Crea l'env v2 raw a curriculum fissato. `cfg_overrides` permette le ablazioni
    (es. {'retreat_clean_release': False}). Ritorna (env, cfg)."""
    cfg = TrainConfigV2(run_dir=run_dir, num_envs=1, horizon=horizon)
    cfg.fixed_curriculum_level = float(curriculum_level)
    if cfg_overrides:
        for k, v in cfg_overrides.items():
            setattr(cfg, k, v)
    env = AdvancedGeneralizedDoorEnv(cfg)
    env.set_curriculum_level(float(curriculum_level))
    return env, cfg


def load_obs_rms(run_dir: str = DEFAULT_RUN_DIR):
    vn_path = os.path.join(run_dir, "vecnormalize.pkl")
    if not os.path.exists(vn_path):
        return None
    with open(vn_path, "rb") as f:
        return pickle.load(f).obs_rms


def load_model(run_dir: str = DEFAULT_RUN_DIR):
    model_path = os.path.join(run_dir, "best_model.zip")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Modello non trovato: {model_path}\n"
            f"(REPO_ROOT={REPO_ROOT}; lancia dalla root del progetto o passa --run-dir)"
        )
    model = SAC.load(model_path)
    model.policy.set_training_mode(False)
    return model


def norm_obs(obs, obs_rms):
    if obs_rms is None:
        return obs
    return np.clip((obs - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8), -10.0, 10.0)


def predict(model, obs, obs_rms, deterministic: bool = True):
    return model.predict(norm_obs(obs, obs_rms), deterministic=deterministic)


# ─────────────────────────────────────────────────────────────────────────────
# Letture fisiche e parametri di dominio
# ─────────────────────────────────────────────────────────────────────────────
def _hinge_dof(env) -> Optional[int]:
    dof = getattr(env, "_door_hinge_dof_adr", None)
    if dof is not None:
        return int(dof)
    dof = getattr(getattr(env, "_domain_rand", None), "hinge_dof_adr", None)
    return int(dof) if dof is not None else None


def read_physics(env) -> dict:
    sim = env._rs_env.sim
    door_qpos = float(sim.data.qpos[env._rs_env.hinge_qpos_addr])
    latch_qpos = float(sim.data.qpos[env._rs_env.handle_qpos_addr])
    dof = _hinge_dof(env)
    door_qvel = float(sim.data.qvel[dof]) if dof is not None else float("nan")

    eef_site_id = env._rs_env.robots[0].eef_site_id
    site_id = (eef_site_id.get("right", list(eef_site_id.values())[0])
               if isinstance(eef_site_id, dict) else eef_site_id)
    eef_pos = np.asarray(sim.data.site_xpos[site_id], float)
    hg = getattr(getattr(env, "_domain_rand", None), "handle_geom_id", None)
    handle_pos = np.asarray(sim.data.geom_xpos[hg], float) if hg is not None else eef_pos
    return dict(door_qpos=door_qpos, latch_qpos=latch_qpos, door_qvel=door_qvel,
                dist_handle=float(np.linalg.norm(eef_pos - handle_pos)))


def realized_domain_params(env) -> dict:
    """Parametri di dominio dell'episodio (costanti dopo il reset). Per stiffness/
    damping/mass riportiamo il RAPPORTO al valore base (interpretabile su tutte le
    porte); per raggio/frizione il valore assoluto."""
    dr = env._domain_rand
    sim = env._rs_env.sim
    base_s = getattr(dr, "base_latch_stiffness", None) or 1.0
    base_d = getattr(dr, "base_hinge_damping", None) or 0.1
    base_m = getattr(dr, "base_door_mass", None) or 1.0
    door_x = float(sim.model.body_pos[env.door_body_id][0])
    return dict(
        handle_friction=float(getattr(dr, "current_handle_friction", float("nan"))),
        handle_radius=float(getattr(dr, "current_handle_radius", float("nan"))),
        latch_stiffness_ratio=float(getattr(dr, "current_latch_stiffness", base_s) / base_s),
        hinge_damping_ratio=float(getattr(dr, "current_hinge_damping", base_d) / base_d),
        door_mass_ratio=float(getattr(dr, "current_door_mass", base_m) / base_m),
        door_x=door_x,
    )


def phase_idx_from_info(info: dict, fallback: int = 0) -> int:
    p = info.get("fsm_phase", None)
    if p is not None:
        return int(p)
    name = str(info.get("fsm_phase_name", "")).upper()
    for i, n in enumerate(PHASE_NAMES):
        if n in name:
            return i
    return fallback


# ─────────────────────────────────────────────────────────────────────────────
# Classificazione del fallimento
# ─────────────────────────────────────────────────────────────────────────────
FAILURE_TYPES = [
    "SUCCESS",
    "REACH timeout",
    "PUSH timeout / grasp lost",
    "HOLD bounce / timeout",
    "RETREAT door bounce",
    "RETREAT latch not neutral",
    "RETREAT timeout",
]


def classify_failure(max_phase_idx: int, door_angle: float, latch_qpos: float,
                     is_success: bool) -> str:
    if is_success:
        return "SUCCESS"
    if max_phase_idx == 0:
        return "REACH timeout"
    if max_phase_idx == 1:
        return "PUSH timeout / grasp lost"
    if max_phase_idx == 2:
        return "HOLD bounce / timeout"
    if max_phase_idx == 3:
        if abs(door_angle) >= 0.03:
            return "RETREAT door bounce"
        if abs(latch_qpos) >= 0.08:
            return "RETREAT latch not neutral"
        return "RETREAT timeout"
    return "REACH timeout"


# ─────────────────────────────────────────────────────────────────────────────
# Record di episodio + rollout seedato (env RAW, niente auto-reset)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class EpisodeRecord:
    success: bool
    true_success: bool
    length: int
    min_door_angle: float
    max_phase: str
    phase_times: dict
    failure_type: str
    door_end: float
    latch_end: float
    handle_friction: float
    handle_radius: float
    latch_stiffness_ratio: float
    hinge_damping_ratio: float
    door_mass_ratio: float
    door_x: float
    seed: int
    hold_action_norms: list = field(default_factory=list)
    retreat_wrist_rots: list = field(default_factory=list)
    bounce_events: list = field(default_factory=list)
    latch_at_transition: Optional[float] = None

    def as_dict(self) -> dict:
        d = asdict(self)
        d["phase_times"] = dict(self.phase_times)
        return d


def seed_everything(seed: int) -> None:
    np.random.seed(seed)


def rollout_episode(env, model, obs_rms, deterministic: bool = True,
                    seed: int = 0, collect_trace: bool = False) -> EpisodeRecord:
    """Un episodio riproducibile. Lo stesso seed su cfg diversi (baseline vs ablazione)
    dà le STESSE condizioni iniziali → confronto appaiato (Patterson et al. 2024)."""
    seed_everything(seed)
    obs, info = env.reset(seed=seed)
    dom0 = realized_domain_params(env)

    phase_time = {n: 0 for n in PHASE_NAMES}
    max_phase = 0
    steps = 0
    min_door = np.inf
    prev_phase = 0
    last_dist = float("nan")
    latch_at_transition = None
    hold_norms, wrist_rots, bounces = [], [], []

    done = False
    info0 = info
    while not done:
        action, _ = predict(model, obs, obs_rms, deterministic=deterministic)
        obs, _r, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        info0 = info
        steps += 1

        step_door = abs(float(info.get("door_angle", np.inf)))
        if np.isfinite(step_door):
            min_door = min(min_door, step_door)

        pidx = phase_idx_from_info(info, fallback=prev_phase)
        phase_time[PHASE_NAMES[pidx]] += 1
        max_phase = max(max_phase, pidx)

        # env RAW: nessun auto-reset → la fisica live è valida anche allo step finale
        phys = read_physics(env)
        last_dist = phys["dist_handle"]
        if collect_trace:
            a = np.asarray(action, float).reshape(-1)
            if pidx == 2:  # HOLD
                hold_norms.append(float(np.linalg.norm(a[:-1])))
                if abs(phys["door_qvel"]) > 0.05:
                    bounces.append((steps, phys["door_qpos"], phys["door_qvel"]))
            elif pidx == 3:  # RETREAT
                wrist_rots.append(float(np.linalg.norm(a[3:6])))
            if pidx == 3 and prev_phase != 3 and latch_at_transition is None:
                latch_at_transition = phys["latch_qpos"]
        prev_phase = pidx

    is_success = bool(info0.get("is_success", False))
    door_end = float(info0.get("door_qpos", min_door))
    latch_end = float(info0.get("latch_qpos", 0.0))
    true_success = is_success and abs(door_end) < 0.03 and abs(latch_end) < 0.08

    return EpisodeRecord(
        success=is_success, true_success=true_success, length=steps,
        min_door_angle=float(min_door), max_phase=PHASE_NAMES[max_phase],
        phase_times=phase_time,
        failure_type=classify_failure(max_phase, door_end, latch_end, is_success),
        door_end=door_end, latch_end=latch_end,
        handle_friction=dom0["handle_friction"], handle_radius=dom0["handle_radius"],
        latch_stiffness_ratio=dom0["latch_stiffness_ratio"],
        hinge_damping_ratio=dom0["hinge_damping_ratio"],
        door_mass_ratio=dom0["door_mass_ratio"], door_x=dom0["door_x"], seed=seed,
        hold_action_norms=hold_norms, retreat_wrist_rots=wrist_rots,
        bounce_events=bounces, latch_at_transition=latch_at_transition,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Output helpers
# ─────────────────────────────────────────────────────────────────────────────
def results_dir(subdir: str = "") -> str:
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    path = os.path.join(base, subdir) if subdir else base
    os.makedirs(path, exist_ok=True)
    return path


def setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), "scratch"))
    import matplotlib.pyplot as plt
    return plt