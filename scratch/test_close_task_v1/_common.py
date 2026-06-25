#!/usr/bin/env python3
# scratch/test_close_task_v1/_common.py
"""
_common — Helper condivisi della suite di test (task di chiusura v1).

Responsabilità (un solo punto di verità per ognuna):
  • risoluzione robusta della radice del repository e dei path di import,
    così che gli script funzionino indipendentemente dalla cartella di lancio
    (riproducibilità — Henderson et al. 2018; Patterson et al. 2024);
  • costruzione dell'ambiente vettorizzato + VecNormalize e caricamento del modello SAC,
    con le STESSE convenzioni dei test originali (runs/close_gen, best_model.zip,
    vecnormalize.pkl);
  • un rollout di episodio *seedato* che raccoglie un record metrico ricco e,
    opzionalmente, tracce passo-passo per la diagnostica di fase.

Nessuna logica statistica qui: quella è in `stats_utils.py`.
"""

from __future__ import annotations

import os
import sys
import contextlib
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# 1. Path / import robusti (niente dipendenza dalla CWD)
# ─────────────────────────────────────────────────────────────────────────────
def find_repo_root(start: Optional[str] = None) -> str:
    """
    Risale dalle cartelle genitore finché trova sia `config/` sia `close_generalized/`.
    Fallback: due livelli sopra questo file (scratch/test_close_task_v1/ → root) e infine
    la CWD. Inserisce in sys.path SIA la root SIA `close_generalized/` (quest'ultima serve
    perché `env_gen.py` importa `from train_close import ...` in forma "piatta").
    """
    here = os.path.abspath(start or os.path.dirname(__file__))
    cur = here
    for _ in range(8):
        if os.path.isdir(os.path.join(cur, "config")) and os.path.isdir(
            os.path.join(cur, "close_generalized")
        ):
            root = cur
            break
        parent = os.path.dirname(cur)
        if parent == cur:
            root = os.path.abspath(os.path.join(here, "..", ".."))
            break
        cur = parent
    else:
        root = os.getcwd()

    for p in (root, os.path.join(root, "close_generalized")):
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)
    return root


REPO_ROOT = find_repo_root()
DEFAULT_RUN_DIR = "runs/close_gen"

# Import del progetto (dopo aver sistemato sys.path)
from config.train_close_config import TrainConfig            # noqa: E402
from close_generalized.env_gen import GeneralizedDoorEnv     # noqa: E402

from stable_baselines3 import SAC                            # noqa: E402
from stable_baselines3.common.vec_env import (               # noqa: E402
    DummyVecEnv, VecNormalize,
)

PHASE_NAMES = ["1:REACH", "2:PUSH", "3:HOLD", "4:RETREAT"]


# ─────────────────────────────────────────────────────────────────────────────
# 2. Costruzione ambiente + modello
# ─────────────────────────────────────────────────────────────────────────────
def make_cfg(run_dir: str = DEFAULT_RUN_DIR, horizon: int = 500, **kw) -> TrainConfig:
    return TrainConfig(run_dir=run_dir, num_envs=1, horizon=horizon, **kw)


def make_vec_env(cfg: TrainConfig, curriculum_level: float = 1.0,
                 env_cls=GeneralizedDoorEnv):
    """
    Costruisce DummyVecEnv([env_cls(cfg)]) + VecNormalize (eval: training=False,
    norm_reward=False) caricando le statistiche da run_dir/vecnormalize.pkl.
    `env_cls` permette di iniettare sottoclassi di ablazione (vedi ablation_study.py).
    Ritorna (venv, raw_env).
    """
    def _init():
        env = env_cls(cfg)
        env.set_curriculum_level(curriculum_level)
        return env

    venv = DummyVecEnv([_init])
    vn_path = os.path.join(cfg.run_dir, "vecnormalize.pkl")
    if os.path.exists(vn_path):
        venv = VecNormalize.load(vn_path, venv)
        venv.training = False
        venv.norm_reward = False
    raw_env = venv.envs[0]  # VecEnvWrapper.__getattr__ inoltra a DummyVecEnv
    return venv, raw_env


def load_model(cfg: TrainConfig, venv):
    model_path = os.path.join(cfg.run_dir, "best_model.zip")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Modello non trovato: {model_path}\n"
            f"(REPO_ROOT={REPO_ROOT}; lancia dalla root del progetto o passa --run-dir)"
        )
    model = SAC.load(model_path, env=venv)
    model.policy.set_training_mode(False)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 3. Letture fisiche dal simulatore
# ─────────────────────────────────────────────────────────────────────────────
def _hinge_dof(raw_env) -> Optional[int]:
    dof = getattr(raw_env, "_door_hinge_dof_adr", None)
    if dof is not None:
        return int(dof)
    try:
        sim = raw_env._rs_env.sim
        jid = sim.model.joint_name2id("Door_hinge")
        return int(sim.model.jnt_dofadr[jid])
    except Exception:
        return None


def read_physics(raw_env) -> dict:
    """Stato fisico istantaneo (letto live, non dalla cache di robosuite)."""
    sim = raw_env._rs_env.sim
    door_qpos = float(sim.data.qpos[raw_env._rs_env.hinge_qpos_addr])
    latch_qpos = float(sim.data.qpos[raw_env._rs_env.handle_qpos_addr])
    dof = _hinge_dof(raw_env)
    door_qvel = float(sim.data.qvel[dof]) if dof is not None else float("nan")

    eef_site_id = raw_env._rs_env.robots[0].eef_site_id
    site_id = (eef_site_id.get("right", list(eef_site_id.values())[0])
               if isinstance(eef_site_id, dict) else eef_site_id)
    eef_pos = np.asarray(sim.data.site_xpos[site_id], float)
    if getattr(raw_env, "handle_geom_id", None) is not None:
        handle_pos = np.asarray(sim.data.geom_xpos[raw_env.handle_geom_id], float)
    else:
        handle_pos = eef_pos
    return dict(
        door_qpos=door_qpos, latch_qpos=latch_qpos, door_qvel=door_qvel,
        dist_handle=float(np.linalg.norm(eef_pos - handle_pos)),
    )


def realized_domain_params(raw_env) -> dict:
    """Parametri di dominio effettivamente campionati per l'episodio corrente."""
    sim = raw_env._rs_env.sim
    door_x = float(sim.model.body_pos[raw_env.door_body_id][0])
    return dict(
        handle_friction=float(getattr(raw_env, "_current_handle_friction", float("nan"))),
        handle_radius=float(getattr(raw_env, "_current_handle_radius", float("nan"))),
        door_x=door_x,
    )


def phase_idx(raw_env) -> int:
    """0=REACH, 1=PUSH, 2=HOLD, 3=RETREAT — dalla FSM dell'env (stato live)."""
    if getattr(raw_env, "_success_latched", False):
        return 3 if getattr(raw_env, "_ready_to_retreat", False) else 2
    if getattr(raw_env, "_grasp_phase", False):
        return 1
    return 0


def phase_idx_from_info(info: dict, fallback: int = 0) -> int:
    """
    Fase ricavata dal dict `info` (necessaria sullo step TERMINALE, quando DummyVecEnv
    ha già auto-resettato l'env e lo stato live di raw_env è del nuovo episodio).
    `info` distingue HOLD/RETREAT via is_success + ready_retreat; REACH/PUSH non sono
    distinguibili da info → si usa `fallback` (l'ultima fase nota pre-terminale).
    """
    if info.get("is_success", False):
        return 3 if info.get("ready_retreat", False) else 2
    return fallback


# ─────────────────────────────────────────────────────────────────────────────
# 4. Classificazione del fallimento (raffinata da eval_stats_close.py)
# ─────────────────────────────────────────────────────────────────────────────
FAILURE_TYPES = [
    "SUCCESS",
    "REACH timeout",
    "GRASP lost",
    "PUSH timeout",
    "HOLD bounce / timeout",
    "RETREAT door bounce",
    "RETREAT latch not neutral",
    "RETREAT timeout",
]


def classify_failure(max_phase_idx: int, dist_handle: float, door_angle: float,
                     latch_qpos: float, is_success: bool) -> str:
    if is_success:
        return "SUCCESS"
    if max_phase_idx == 0:
        return "REACH timeout"
    if max_phase_idx == 1:
        return "GRASP lost" if dist_handle > 0.08 else "PUSH timeout"
    if max_phase_idx == 2:
        return "HOLD bounce / timeout"
    if max_phase_idx == 3:
        if door_angle >= 0.03:
            return "RETREAT door bounce"
        if abs(latch_qpos) >= 0.08:
            return "RETREAT latch not neutral"
        return "RETREAT timeout"
    return "REACH timeout"


# ─────────────────────────────────────────────────────────────────────────────
# 5. Record di episodio + rollout seedato
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class EpisodeRecord:
    success: bool                 # permissivo: ha raggiunto HOLD/RETREAT (info is_success)
    true_success: bool            # porta chiusa (|door|<0.03) E latch neutro (|latch|<0.08)
    length: int
    min_door_angle: float
    max_phase: str
    phase_times: dict
    failure_type: str
    door_end: float
    latch_end: float
    handle_friction: float
    handle_radius: float
    door_x: float
    seed: int
    # tracce opzionali (riempite se collect_trace=True)
    hold_action_norms: list = field(default_factory=list)
    retreat_wrist_rots: list = field(default_factory=list)
    bounce_events: list = field(default_factory=list)
    latch_at_transition: Optional[float] = None

    def as_dict(self) -> dict:
        d = asdict(self)
        d["phase_times"] = dict(self.phase_times)
        return d


def seed_everything(seed: int, venv=None) -> None:
    """Seed globale di numpy (la domain randomization usa np.random) + venv se possibile."""
    np.random.seed(seed)
    if venv is not None:
        with contextlib.suppress(Exception):
            venv.seed(seed)


def rollout_episode(venv, model, raw_env, deterministic: bool = True,
                    seed: int = 0, collect_trace: bool = False) -> EpisodeRecord:
    """
    Esegue UN episodio in modo riproducibile (seed → domain randomization fissata).
    Lo stesso seed su env diversi (baseline vs ablazione) dà le STESSE condizioni
    iniziali → confronto appaiato (Patterson et al. 2024).
    """
    seed_everything(seed, venv)
    obs = venv.reset()
    # Parametri di dominio REALIZZATI: costanti nell'episodio, fissati al reset.
    # Vanno letti ORA (l'env è appena resettato), non a fine episodio (dopo
    # l'auto-reset di DummyVecEnv conterrebbero i valori del PROSSIMO episodio).
    dom0 = realized_domain_params(raw_env)

    phase_time = {n: 0 for n in PHASE_NAMES}
    max_phase = 0
    steps = 0
    min_door = np.inf
    prev_phase = 0
    last_dist = float("nan")
    latch_at_transition = None
    hold_norms, wrist_rots, bounces = [], [], []

    done = False
    info0 = {}
    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, _r, dones, infos = venv.step(action)
        done = bool(dones[0])
        info0 = infos[0]
        steps += 1

        # Angolo porta dello step: info["door_angle"] è impostato dall'env PRIMA
        # dell'eventuale auto-reset di DummyVecEnv → valido anche sullo step terminale.
        step_door = abs(float(info0.get("door_angle", np.inf)))
        if np.isfinite(step_door):
            min_door = min(min_door, step_door)

        if not done:
            # Stato live valido: l'env non è stato resettato.
            phys = read_physics(raw_env)
            last_dist = phys["dist_handle"]
            pidx = phase_idx(raw_env)
            phase_time[PHASE_NAMES[pidx]] += 1
            max_phase = max(max_phase, pidx)
            if collect_trace:
                a = np.asarray(action[0], float)
                if pidx == 2:  # HOLD
                    hold_norms.append(float(np.linalg.norm(a[:-1])))
                    if abs(phys["door_qvel"]) > 0.05:
                        bounces.append((steps, phys["door_qpos"], phys["door_qvel"]))
                elif pidx == 3:  # RETREAT
                    wrist_rots.append(float(np.linalg.norm(a[3:6])))
                if pidx == 3 and prev_phase != 3 and latch_at_transition is None:
                    latch_at_transition = phys["latch_qpos"]
            prev_phase = pidx
        else:
            # Step terminale: raw_env già auto-resettato → uso info (stato terminale).
            pidx = phase_idx_from_info(info0, fallback=prev_phase)
            phase_time[PHASE_NAMES[pidx]] += 1
            max_phase = max(max_phase, pidx)

    is_success = bool(info0.get("is_success", False))
    door_end = float(info0.get("door_qpos", min_door))
    latch_end = float(info0.get("latch_qpos", 0.0))
    true_success = is_success and abs(door_end) < 0.03 and abs(latch_end) < 0.08

    return EpisodeRecord(
        success=is_success,
        true_success=true_success,
        length=steps,
        min_door_angle=float(min_door),
        max_phase=PHASE_NAMES[max_phase],
        phase_times=phase_time,
        failure_type=classify_failure(max_phase, last_dist,
                                      abs(door_end), latch_end, is_success),
        door_end=door_end,
        latch_end=latch_end,
        handle_friction=dom0["handle_friction"],
        handle_radius=dom0["handle_radius"],
        door_x=dom0["door_x"],
        seed=seed,
        hold_action_norms=hold_norms,
        retreat_wrist_rots=wrist_rots,
        bounce_events=bounces,
        latch_at_transition=latch_at_transition,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 6. Output helpers
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
