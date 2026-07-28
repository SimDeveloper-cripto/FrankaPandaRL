#!/usr/bin/env python3
# scratch/test_open_task_v2/_common.py
"""
_common — Helper condivisi della suite di test scientifica (task di APERTURA v2).

Speculare a `scratch/test_close_task_v2/_common.py`, adattato a `open_generalized_v2`.

Differenze strutturali rispetto alla chiusura (tutte verificate sul sorgente):
  • Pacchetto `open_generalized_v2`; config `TrainConfigV2Open`; env
    `AdvancedGeneralizedOpenDoorEnv`. Env RAW gymnasium (niente VecEnv/auto-reset):
    lo stato fisico è leggibile anche allo step terminale.
  • L'obiettivo NON è door_angle ≈ 0 ma door_angle ≈ goal_angle, con `goal_angle`
    RICAMPIONATO a ogni reset (cfg.goal_frac_min..max × range effettivo). Il goal è
    quindi una variabile di episodio: entra nel record ed è un asse di robustezza.
  • Fasi FSM: REACH → PULL → HOLD_OPEN → RETREAT (`info["fsm_phase"]`, 0..3).
  • L'env espone un'info molto ricca (open_error, latch_qpos, door_qvel,
    gripper_width, retreat_steps/free_steps/moved/restoring): la usiamo come fonte
    primaria, riducendo gli accessi fragili al simulatore ai soli parametri di dominio.
  • `set_retreat_to_start()` NON esiste in env_v2.py (il vecchio diagnose_phase.py lo
    chiama e crasherebbe): qui NON viene chiamato.

Rif. metodologici: Henderson et al. 2018 (riproducibilità); Patterson et al. 2024
(disegno controllato/appaiato); Tobin et al. 2017, Zhao et al. 2020 (randomizzazione
fisica come asse di valutazione).
"""

from __future__ import annotations

import os
import sys
import json
import glob
import pickle
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# JSON robusto ai tipi numpy (bug noto: int64 non serializzabile)
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
    """Risale finché trova la cartella-pacchetto `open_generalized_v2` e mette la root
    in sys.path (così sia gli import qualificati sia i fallback 'piatti' di env_v2.py
    si risolvono)."""
    here = os.path.abspath(start or os.path.dirname(__file__))
    cur = here
    root = None
    for _ in range(8):
        if os.path.isdir(os.path.join(cur, "open_generalized_v2")):
            root = cur
            break
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    if root is None:
        root = os.path.abspath(os.path.join(here, "..", ".."))
    for p in (root, os.path.join(root, "open_generalized_v2")):
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)
    return root


REPO_ROOT = find_repo_root()

# ── Modello ──────────────────────────────────────────────────────────────────
# L'apertura v2 è addestrata SOLO al curriculum 1 (posa variabile): un run, un modello.
# `runs/open_gen_v2` è il default di TrainConfigV2Open (il training non espone --run-dir).
# Se la cartella non esiste, `resolve_run_dir()` cerca automaticamente un run
# `runs/open_gen_v2*` che contenga un modello, e stampa quale ha scelto.
# Ogni script accetta comunque --run-dir per un override esplicito.
DEFAULT_RUN_DIR = "runs/open_gen_v2"
CURRICULUM = 1.0
MODEL_SPECS = [("curr1_posa_variabile", DEFAULT_RUN_DIR, CURRICULUM)]

PHASE_NAMES = ["REACH", "PULL", "HOLD_OPEN", "RETREAT"]
MODEL_FILE_PREFERENCE = ("best_model.zip", "final_model.zip", "latest_model.zip")

# Soglia diagnostica §1.55: sotto questo allontanamento il braccio è "fermo sulla maniglia".
STUCK_MOVE_THRESH = 0.06

from open_generalized_v2.config_v2 import TrainConfigV2Open              # noqa: E402
from open_generalized_v2.env_v2 import AdvancedGeneralizedOpenDoorEnv    # noqa: E402
from stable_baselines3 import SAC                                        # noqa: E402


def resolve_run_dir(run_dir: Optional[str] = None, verbose: bool = True) -> str:
    """Restituisce una run-dir che contiene un modello. Ordine: valore esplicito →
    default → autodiscovery `runs/open_gen_v2*` (più recente con un .zip)."""
    if run_dir:
        return run_dir
    cand = DEFAULT_RUN_DIR
    if find_model(cand, quiet=True):
        return cand
    root = os.path.join(REPO_ROOT, "runs")
    matches = []
    for d in sorted(glob.glob(os.path.join(root, "open_gen_v2*"))):
        if os.path.isdir(d) and find_model(d, quiet=True):
            matches.append(d)
    if matches:
        chosen = max(matches, key=os.path.getmtime)
        rel = os.path.relpath(chosen, REPO_ROOT)
        if verbose:
            print(f"[run-dir] '{DEFAULT_RUN_DIR}' non trovata → uso '{rel}'")
        return rel
    return cand


def find_model(run_dir: str, quiet: bool = False) -> Optional[str]:
    """best_model.zip → final_model.zip → latest_model.zip → .zip più recente."""
    base = run_dir if os.path.isabs(run_dir) else os.path.join(REPO_ROOT, run_dir)
    if not os.path.isdir(base):
        return None
    for name in MODEL_FILE_PREFERENCE:
        p = os.path.join(base, name)
        if os.path.exists(p):
            return p
    zips = [os.path.join(base, f) for f in os.listdir(base) if f.endswith(".zip")]
    if zips:
        p = max(zips, key=os.path.getmtime)
        if not quiet:
            print(f"[modello] nessun best/final/latest: uso il più recente {os.path.basename(p)}")
        return p
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Costruzione ambiente RAW + modello
# ─────────────────────────────────────────────────────────────────────────────
def make_raw_env(curriculum_level: float = CURRICULUM, run_dir: str = DEFAULT_RUN_DIR,
                 horizon: int = 600, cfg_overrides: Optional[dict] = None):
    """Env di apertura raw a curriculum fissato. `cfg_overrides` serve alle ablazioni
    (es. {'retreat_escape_enabled': False}). Ritorna (env, cfg).

    NB: l'orizzonte di default resta 600 come in training — l'apertura ha un RETREAT
    lungo (riporto leva + escape + rampa) e tagliarlo falserebbe le terminazioni.
    """
    cfg = TrainConfigV2Open(run_dir=run_dir, num_envs=1, horizon=horizon)
    cfg.fixed_curriculum_level = float(curriculum_level)
    if cfg_overrides:
        for k, v in cfg_overrides.items():
            if not hasattr(cfg, k):
                raise AttributeError(f"cfg_override sconosciuto: {k}")
            setattr(cfg, k, v)
    env = AdvancedGeneralizedOpenDoorEnv(cfg)
    env.set_curriculum_level(float(curriculum_level))
    # NOTA: NON chiamiamo set_retreat_to_start(): il metodo non esiste in env_v2.py.
    return env, cfg


def load_obs_rms(run_dir: str = DEFAULT_RUN_DIR):
    base = run_dir if os.path.isabs(run_dir) else os.path.join(REPO_ROOT, run_dir)
    vn_path = os.path.join(base, "vecnormalize.pkl")
    if not os.path.exists(vn_path):
        print(f"[attenzione] vecnormalize.pkl assente in {run_dir}: osservazioni NON normalizzate")
        return None
    with open(vn_path, "rb") as f:
        return pickle.load(f).obs_rms


def load_model(run_dir: str = DEFAULT_RUN_DIR):
    path = find_model(run_dir)
    if path is None:
        raise FileNotFoundError(
            f"Nessun modello (.zip) in {run_dir}\n"
            f"(REPO_ROOT={REPO_ROOT}; lancia dalla root del progetto o passa --run-dir)"
        )
    model = SAC.load(path)
    model.policy.set_training_mode(False)
    return model


def norm_obs(obs, obs_rms):
    if obs_rms is None:
        return obs
    return np.clip((obs - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8), -10.0, 10.0)


def predict(model, obs, obs_rms, deterministic: bool = True):
    return model.predict(norm_obs(obs, obs_rms), deterministic=deterministic)


# ─────────────────────────────────────────────────────────────────────────────
# Parametri di dominio realizzati (costanti dopo il reset)
# ─────────────────────────────────────────────────────────────────────────────
def realized_domain_params(env) -> dict:
    """Fisica campionata per l'episodio. Per stiffness/damping/massa il RAPPORTO al
    valore base (confrontabile fra porte); per raggio/frizione il valore assoluto.
    `goal_angle` è specifico dell'apertura: è il bersaglio campionato al reset."""
    dr = env._domain_rand
    base_s = getattr(dr, "base_latch_stiffness", None) or 1.0
    base_d = getattr(dr, "base_hinge_damping", None) or 0.1
    base_m = getattr(dr, "base_door_mass", None) or 1.0
    try:
        door_x = float(env._rs_env.sim.model.body_pos[env.door_body_id][0])
    except Exception:
        door_x = float("nan")
    return dict(
        handle_friction=float(getattr(dr, "current_handle_friction", float("nan"))),
        handle_radius=float(getattr(dr, "current_handle_radius", float("nan"))),
        latch_stiffness_ratio=float(getattr(dr, "current_latch_stiffness", base_s) / base_s),
        hinge_damping_ratio=float(getattr(dr, "current_hinge_damping", base_d) / base_d),
        door_mass_ratio=float(getattr(dr, "current_door_mass", base_m) / base_m),
        door_x=door_x,
        goal_angle=float(getattr(env, "_goal_angle", float("nan"))),
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
# Classificazione esiti (speculare alla chiusura, con i modi propri dell'apertura)
# ─────────────────────────────────────────────────────────────────────────────
FAILURE_TYPES = [
    "SUCCESS",
    "REACH timeout",                     # non afferra
    "PULL timeout / grasp lost",         # afferra ma non porta la porta al goal
    "HOLD_OPEN regress / timeout",       # arriva al goal ma non lo mantiene
    "RETREAT overshoot (oltre il goal)", # aperta TROPPO: finisce oltre goal+tol
    "RETREAT door regress (sotto il goal)",  # si richiude sotto goal-tol
    "RETREAT latch not neutral",         # leva non tornata (rilascio incompleto)
    "RETREAT stuck on handle",           # §1.55: braccio mai allontanato dalla maniglia
    "RETREAT timeout",
]

TERMINATION_TYPES = ["PULITA", "ESOGENA", "HARD-CAP", "troncata a orizzonte"]


def classify_failure(max_phase_idx: int, open_error_end: float, latch_end: float,
                     retreat_moved_max: float, is_success: bool,
                     open_tol: float, latch_tol: float,
                     signed_error_end: float = 0.0) -> str:
    """Mirror della classificazione della chiusura. L'ordine dei test riflette la
    catena causale: si guarda prima DOVE si è fermata la FSM, poi — se è arrivata al
    RETREAT — quale delle tre condizioni di uscita pulita è mancata."""
    if is_success and open_error_end <= open_tol and abs(latch_end) < latch_tol \
            and retreat_moved_max >= STUCK_MOVE_THRESH:
        return "SUCCESS"
    if max_phase_idx == 0:
        return "REACH timeout"
    if max_phase_idx == 1:
        return "PULL timeout / grasp lost"
    if max_phase_idx == 2:
        return "HOLD_OPEN regress / timeout"
    # max_phase_idx == 3 (RETREAT)
    if open_error_end > open_tol:
        # `open_error` è BILATERALE: distinguere il segno è essenziale, perché
        # "apre troppo" e "si richiude" hanno cause e rimedi opposti (il primo è un
        # problema di modulazione sul goal, il secondo di tenuta della presa/fisica).
        return ("RETREAT overshoot (oltre il goal)" if signed_error_end > 0
                else "RETREAT door regress (sotto il goal)")
    if abs(latch_end) >= latch_tol:
        return "RETREAT latch not neutral"
    if retreat_moved_max < STUCK_MOVE_THRESH:
        return "RETREAT stuck on handle"
    return "RETREAT timeout"


def classify_termination(terminated: bool, latch_end: float, retreat_steps: int,
                         retreat_free_steps: int, cfg) -> str:
    """Mirror della lettura di diagnose_phase.py §1.43/§1.45, allineata alla logica
    reale di reward_v2 (terminated = (min_release ∧ latch_home) ∨ exo_exit ∨ hardcap)."""
    if not terminated:
        return "troncata a orizzonte"
    hard_cap = int(getattr(cfg, "retreat_hard_cap", 120))
    latch_tol = float(getattr(cfg, "retreat_latch_term_tol", 0.15))
    min_release = int(retreat_free_steps) >= int(getattr(cfg, "fsm_retreat_target_steps", 40))
    if retreat_steps >= hard_cap:
        return "HARD-CAP"
    if abs(latch_end) < latch_tol and min_release:
        return "PULITA"
    return "ESOGENA"


# ─────────────────────────────────────────────────────────────────────────────
# Record di episodio + rollout seedato
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class EpisodeRecord:
    # esiti
    success: bool                 # permissivo: info["is_success"] (fase ∈ HOLD_OPEN/RETREAT)
    true_success: bool            # + porta ancora al goal E leva neutra a fine episodio
    clean_success: bool           # + ritiro effettivo (braccio allontanato) e uscita PULITA
    failure_type: str
    termination_type: str
    length: int
    # metriche di apertura
    min_open_error: float         # miglior avvicinamento al goal nell'episodio
    open_error_end: float         # errore al goal a fine episodio
    door_angle_end: float
    latch_end: float
    # ritiro
    retreat_moved_max: float
    retreat_steps: int
    retreat_free_steps: int
    # fasi
    max_phase: str
    phase_times: dict
    # dominio (assi di robustezza)
    handle_friction: float
    handle_radius: float
    latch_stiffness_ratio: float
    hinge_damping_ratio: float
    door_mass_ratio: float
    door_x: float
    goal_angle: float
    seed: int
    # tracce (solo con collect_trace=True)
    hold_action_norms: list = field(default_factory=list)
    retreat_wrist_rots: list = field(default_factory=list)
    regress_events: list = field(default_factory=list)
    retreat_moved_trace: list = field(default_factory=list)
    open_error_at_transition: Optional[float] = None
    latch_at_transition: Optional[float] = None

    def as_dict(self) -> dict:
        d = asdict(self)
        d["phase_times"] = dict(self.phase_times)
        return d


def seed_everything(seed: int) -> None:
    np.random.seed(seed)


def rollout_episode(env, model, obs_rms, deterministic: bool = True,
                    seed: int = 0, collect_trace: bool = False) -> EpisodeRecord:
    """Un episodio riproducibile.

    Lo stesso seed su configurazioni diverse (baseline vs ablazione) riproduce le
    STESSE condizioni iniziali — goal, posa e fisica sono campionati dal RNG di numpy
    che riseminiamo qui — quindi i confronti sono APPAIATI (Patterson et al. 2024;
    Colas et al. 2018: riduce la varianza a parità di episodi).
    """
    cfg = env.cfg
    open_tol = float(getattr(cfg, "open_tol_rad", 0.05))
    latch_tol = float(getattr(cfg, "retreat_latch_term_tol", 0.15))

    seed_everything(seed)
    obs, _info_reset = env.reset(seed=seed)
    dom0 = realized_domain_params(env)

    phase_time = {n: 0 for n in PHASE_NAMES}
    max_phase = 0
    steps = 0
    min_open_err = np.inf
    prev_phase = 0
    retreat_moved_max = 0.0
    hold_norms, wrist_rots, regress, moved_trace = [], [], [], []
    open_err_at_transition = None
    latch_at_transition = None

    done = False
    terminated = False
    info = {}
    while not done:
        action, _ = predict(model, obs, obs_rms, deterministic=deterministic)
        obs, _r, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        steps += 1

        oe = float(info.get("open_error", np.inf))
        if np.isfinite(oe):
            min_open_err = min(min_open_err, oe)

        pidx = phase_idx_from_info(info, fallback=prev_phase)
        phase_time[PHASE_NAMES[pidx]] += 1
        max_phase = max(max_phase, pidx)

        moved = float(info.get("retreat_moved", 0.0))
        retreat_moved_max = max(retreat_moved_max, moved)

        if collect_trace:
            a = np.asarray(action, float).reshape(-1)
            if pidx == 2:                                    # HOLD_OPEN
                hold_norms.append(float(np.linalg.norm(a[:-1])))
                if oe > open_tol:
                    # errore CON SEGNO: >0 = aperta oltre il goal, <0 = ricaduta sotto.
                    # Registrarne il segno è indispensabile: le due cose hanno rimedi opposti.
                    regress.append((steps,
                                    float(info.get("door_angle", 0.0)) - dom0["goal_angle"],
                                    float(info.get("door_qvel", 0.0))))
            elif pidx == 3:                                  # RETREAT
                if a.shape[0] >= 7:
                    wrist_rots.append(float(np.linalg.norm(a[3:6])))
                moved_trace.append((steps, moved, float(info.get("latch_qpos", 0.0)),
                                    float(info.get("door_angle", 0.0))))
            if pidx == 3 and prev_phase != 3 and open_err_at_transition is None:
                open_err_at_transition = oe
                latch_at_transition = float(info.get("latch_qpos", 0.0))
        prev_phase = pidx

    is_success = bool(info.get("is_success", False))
    open_error_end = float(info.get("open_error", np.inf))
    door_angle_end = float(info.get("door_angle", float("nan")))
    latch_end = float(info.get("latch_qpos", 0.0))
    retreat_steps = int(info.get("retreat_steps", 0))
    retreat_free = int(info.get("retreat_free_steps", 0))

    term_type = classify_termination(bool(terminated), latch_end, retreat_steps,
                                     retreat_free, cfg)
    true_success = bool(is_success and open_error_end <= open_tol
                        and abs(latch_end) < latch_tol)
    clean_success = bool(true_success and retreat_moved_max >= STUCK_MOVE_THRESH
                         and term_type == "PULITA")

    return EpisodeRecord(
        success=is_success, true_success=true_success, clean_success=clean_success,
        failure_type=classify_failure(max_phase, open_error_end, latch_end,
                                      retreat_moved_max, is_success, open_tol, latch_tol,
                                      signed_error_end=(door_angle_end - dom0["goal_angle"])),
        termination_type=term_type, length=steps,
        min_open_error=float(min_open_err), open_error_end=open_error_end,
        door_angle_end=door_angle_end, latch_end=latch_end,
        retreat_moved_max=float(retreat_moved_max), retreat_steps=retreat_steps,
        retreat_free_steps=retreat_free,
        max_phase=PHASE_NAMES[max_phase], phase_times=phase_time,
        handle_friction=dom0["handle_friction"], handle_radius=dom0["handle_radius"],
        latch_stiffness_ratio=dom0["latch_stiffness_ratio"],
        hinge_damping_ratio=dom0["hinge_damping_ratio"],
        door_mass_ratio=dom0["door_mass_ratio"], door_x=dom0["door_x"],
        goal_angle=dom0["goal_angle"], seed=seed,
        hold_action_norms=hold_norms, retreat_wrist_rots=wrist_rots,
        regress_events=regress, retreat_moved_trace=moved_trace,
        open_error_at_transition=open_err_at_transition,
        latch_at_transition=latch_at_transition,
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


def safe_hist(ax, data, bins=30, **kwargs):
    """Istogramma robusto ai campioni DEGENERI.

    Caso reale e frequente: tutti i valori identici (100% di successi nel bootstrap,
    una randomizzazione disattivata, una traccia costante). `hist(bins=30)` in quel caso
    solleva "Too many bins for data range" e farebbe fallire l'intera batteria dopo aver
    già speso i rollout. Qui si degrada a un singolo bin e si va avanti.
    """
    a = np.asarray(list(data), dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        ax.text(0.5, 0.5, "nessun dato", ha="center", va="center", transform=ax.transAxes)
        return
    lo, hi = float(a.min()), float(a.max())
    if (hi - lo) <= 1e-12:
        pad = max(abs(lo) * 1e-3, 1e-6)
        ax.hist(a, bins=1, range=(lo - pad, hi + pad), **kwargs)
        return

    # Dati su SUPPORTO DISCRETO (es. la media bootstrap di 100 esiti binari può valere
    # solo multipli di 1 punto percentuale). Chiedendo più bin dei valori possibili, una
    # parte dei bin resta strutturalmente vuota e l'istogramma sembra "a pettine": è un
    # artefatto del binning, non una proprietà dei dati. In quel caso si allineano i bin
    # al reticolo dei valori.
    uniq = np.unique(a)
    if 1 < uniq.size <= bins:
        step = float(np.min(np.diff(uniq)))
        n_edges = int(np.floor((uniq[-1] - uniq[0]) / step)) + 2 if step > 0 else 0
        if step > 0 and n_edges <= 4 * bins:
            edges = np.arange(uniq[0] - step / 2.0, uniq[-1] + step, step)
            ax.hist(a, bins=edges, **kwargs)
            return
    ax.hist(a, bins=int(max(1, min(bins, a.size))), range=(lo, hi), **kwargs)
