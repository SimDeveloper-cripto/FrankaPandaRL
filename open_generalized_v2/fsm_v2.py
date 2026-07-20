#!/usr/bin/env python3
# open_generalized_v2/fsm_v2.py
#
# AdaptiveFSMOpen — Macchina a stati a soglie adattive per l'APERTURA generalizzata.
#
# E' la differenza architetturale principale rispetto alla v1 (come per la chiusura):
# la transizione tra fasi non usa soglie fisse ma soglie CONTEST-SENSITIVE, funzione
# della fisica corrente (frizione/raggio maniglia, rigidità latch) fornita dal domain
# randomizer. Questo è ciò che permette la "generalizzazione della generalizzazione".
#
# Fasi (SPECULARI alla chiusura REACH→PUSH→HOLD→RETREAT):
#   REACH      0  — avvicinati e afferra la maniglia
#   PULL       1  — tira/spingi la porta verso l'angolo-obiettivo (apertura)
#   HOLD_OPEN  2  — mantieni la porta aperta al goal per un tempo adattivo
#   RETREAT    3  — rilascia e allontanati (transizioni gestite dall'environment)
#
# Inversione chiave chiusura ↔ apertura:
#   chiusura:  PUSH→HOLD   quando  door_angle <= success_angle      (porta chiusa)
#   apertura:  PULL→HOLD   quando  door_angle >= goal_angle - tol    (porta aperta al goal)
#
# Riferimenti:
#   [1] Sutton, Precup & Singh (1999) — fasi come opzioni con terminazione.
#   [2] Konidaris & Barto (2009)      — precondizioni di opzione.
#   [13] ManipForce (2015)            — soglie adattive di presa/forza.

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional, List
from scipy.spatial.transform import Rotation as R_scipy

PHASE_REACH     = 0
PHASE_PULL      = 1
PHASE_HOLD_OPEN = 2
PHASE_RETREAT   = 3

PHASE_NAMES = {0: "REACH", 1: "PULL", 2: "HOLD_OPEN", 3: "RETREAT"}


@dataclass
class FSMStateOpen:
    phase                : int = PHASE_REACH
    grasp_confirm_count  : int = 0
    hold_open_duration   : int = 0
    reach_steps          : int = 0
    pull_steps           : int = 0
    hold_steps_total     : int = 0
    retreat_steps        : int = 0
    retreat_free_steps   : int = 0   # §1.38: step di RETREAT dopo il rilascio delle dita
    return_hold          : int = 0
    retreat_pos          : Optional[np.ndarray] = None
    target_hold_steps    : Optional[int] = None

    def reset(self):
        self.phase               = PHASE_REACH
        self.grasp_confirm_count = 0
        self.hold_open_duration  = 0
        self.reach_steps         = 0
        self.pull_steps          = 0
        self.hold_steps_total    = 0
        self.retreat_steps       = 0
        self.return_hold         = 0
        self.retreat_pos         = None
        self.target_hold_steps   = None

    @property
    def one_hot(self) -> np.ndarray:
        """§1.34 — [fsm_reach, fsm_pull, fsm_hold_open, fsm_retreat] ∈ {0,1}^4.
        Mirror ESATTO della chiusura (FSMState.one_hot): è la feature che dice alla policy
        IN CHE FASE È. Senza di essa lo stato fisico di HOLD_OPEN (tieni la maniglia) e di
        RETREAT (molla e allontanati) è INDISTINGUIBILE per la policy — che quindi non può
        imparare due comportamenti diversi nello stesso stato osservato: il braccio resta
        attaccato alla maniglia. La chiusura ha sempre avuto questa feature nell'obs; nel
        porting dell'apertura era andata persa."""
        v             = np.zeros(4, dtype=np.float32)
        v[self.phase] = 1.0
        return v

    @property
    def phase_name(self) -> str:
        return PHASE_NAMES.get(self.phase, "?")


class AdaptiveFSMOpen:
    """
    FSM a soglie adattive per l'apertura. API speculare a AdaptiveFSM (chiusura):
      - grip_thresh(friction)            : soglia di chiusura gripper adattiva
      - grasp_dist_thresh(handle_radius) : distanza di presa adattiva
      - compute_target_hold_steps(...)   : timer HOLD_OPEN adattivo
      - update(...)                      : avanza la FSM, ritorna lista di eventi (log)
    """

    _GRASP_CONFIRM_STEPS = 5   # step consecutivi di presa valida per confermare REACH→PULL

    def __init__(self, cfg):
        self.cfg = cfg
        self.state = FSMStateOpen()

    # ── Soglie adattive (§3.1) ──────────────────────────────────────────────────

    def grip_thresh(self, handle_friction: float) -> float:
        """Soglia di chiusura gripper adattiva alla frizione (ManipForce 2015, §3.1)."""
        f_min = self.cfg.fsm_friction_min
        f_max = self.cfg.fsm_friction_max
        norm_f = float(np.clip((handle_friction - f_min) / (f_max - f_min + 1e-8), 0.0, 1.0))
        return float(np.clip(
            self.cfg.fsm_grip_thresh_base - self.cfg.fsm_grip_thresh_k_fric * norm_f,
            0.50, 0.90,
        ))

    def grasp_dist_thresh(self, handle_radius: float) -> float:
        """Distanza di presa adattiva al raggio maniglia (§3.1)."""
        return float(
            self.cfg.fsm_grasp_dist_base
            + self.cfg.fsm_grasp_dist_k_radius * handle_radius
            + self.cfg.fsm_grasp_dist_offset
        )

    @staticmethod
    def rotate_quat_z_mujoco(quat_wxyz: np.ndarray, angle: float) -> np.ndarray:
        """§1.33 — orientazione CORRENTE della porta: ruota il quaternione statico del modello
        di `angle` attorno alla Z del mondo (l'asse del cardine è verticale). Necessario perché
        `model.body_quat` è l'orientazione STATICA (porta chiusa): a porta aperta di θ la
        normale vera è Rz(+θ)·n_statica (verificato in MuJoCo: match esatto con la xmat del
        pannello; errore usando la statica a θ=0.35: 20.1°)."""
        w, x, y, z = quat_wxyz
        r = R_scipy.from_euler("z", float(angle)) * R_scipy.from_quat([x, y, z, w])
        xq, yq, zq, wq = r.as_quat()
        return np.array([wq, xq, yq, zq], dtype=float)

    @staticmethod
    def compute_retreat_pos(
        eef_pos         : np.ndarray,
        door_quat_mujoco: np.ndarray,  # wxyz (MuJoCo)
        retreat_dist    : float,
        retreat_z       : float,
        door_pos        : Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """§3.1/§1.32 — target di ritiro allineato alla NORMALE della porta (copia ESATTA di
        close fsm_v2.compute_retreat_pos): eef + dist × door_normal + [0,0,z]. La normale è la
        prima colonna della matrice di rotazione del corpo porta (asse X locale): punta verso
        il robot, quindi è la direzione di allontanamento corretta a QUALSIASI yaw."""
        w, x, y, z = door_quat_mujoco
        door_mat    = R_scipy.from_quat([x, y, z, w]).as_matrix()
        door_normal = door_mat[:, 0]
        door_normal = door_normal / (np.linalg.norm(door_normal) + 1e-8)

        # §1.40 — ORIENTA la normale VERSO IL ROBOT invece di ASSUMERLO.
        # La formula (condivisa con la chiusura) documenta: "this is the normal pointing
        # toward the robot". MISURATO nell'env reale: e' FALSO. La normale punta a -Y, cioe'
        # VIA dal robot, DENTRO la porta: retreat_pos finiva 13 cm PIU' VICINO alla maniglia
        # (0.338 -> 0.210) invece che piu' lontano. Il "ritiro" ordinava al braccio di
        # attraversare il pannello -> il braccio SPINGEVA la porta richiudendola e si bloccava.
        # Nella CHIUSURA il difetto e' invisibile: li' la porta e' gia' a fine-corsa contro il
        # telaio, spingerla non produce movimento. Nell'APERTURA la porta e' libera -> disastro.
        # Fix: scegliere il verso col segno del prodotto scalare verso l'eef. Cosi' il codice
        # fa ESATTAMENTE cio' che la sua docstring dichiara, per qualunque posa/yaw.
        _eef = np.asarray(eef_pos, dtype=float)
        if door_pos is not None:
            if float(np.dot(door_normal, _eef - np.asarray(door_pos, dtype=float))) < 0.0:
                door_normal = -door_normal

        retreat     = np.asarray(eef_pos, dtype=np.float32) + float(retreat_dist) * door_normal.astype(np.float32)
        retreat[2] += float(retreat_z)
        return retreat.astype(np.float32)

    def compute_target_hold_steps(self, control_freq, latch_stiffness, base_latch_stiffness):
        """
        Timer di HOLD_OPEN adattivo alla rigidità del latch (§3.1).
        Latch più rigido tende a richiudere → serve mantenere l'apertura un po' di più.
        """
        base = float(base_latch_stiffness) if base_latch_stiffness else 1.0
        stiff = float(latch_stiffness) if latch_stiffness else base
        # frazione [0,1]: 0 = molla debole, 1 = molla forte (max 2x base)
        norm = float(np.clip((stiff - base) / (base + 1e-8), 0.0, 1.0))
        extra = self.cfg.fsm_hold_k_stiff * norm * control_freq * 0.5
        return int(self.cfg.fsm_hold_base_steps + extra)

    # ── Update ───────────────────────────────────────────────────────────────────

    def update(
        self,
        *,
        door_angle       : float,
        goal_angle       : float,
        open_tol         : float,
        gripper_action   : float,
        dist_handle      : float,
        handle_radius    : float,
        handle_friction  : float,
        is_physically_closed: bool,
        gripper_width    : float,
        prev_angle       : float,
        control_freq     : int,
        door_qpos        : float,
        latch_stiffness  : float,
        base_latch_stiffness: float,
        eef_pos          : Optional[np.ndarray] = None,
        door_quat_mujoco : Optional[np.ndarray] = None,
        door_pos         : Optional[np.ndarray] = None,   # §1.40: per orientare la normale
        wrist_align_ok   : bool = True,
        beta_probs       : Optional[dict] = None,
    ) -> List[str]:
        """
        Avanza la FSM di uno step. Ritorna una lista di stringhe-evento per il logging.
        SPECULARE alla chiusura, con l'unica inversione nella condizione PULL→HOLD_OPEN.
        """
        s = self.state
        events: List[str] = []

        g_thresh = self.grip_thresh(handle_friction)         # §3.1
        d_thresh = self.grasp_dist_thresh(handle_radius)     # §3.1

        if s.target_hold_steps is None:
            s.target_hold_steps = self.compute_target_hold_steps(
                control_freq, latch_stiffness, base_latch_stiffness
            )

        if   s.phase == PHASE_REACH    : s.reach_steps      += 1
        elif s.phase == PHASE_PULL     : s.pull_steps       += 1
        elif s.phase == PHASE_HOLD_OPEN: s.hold_steps_total += 1
        elif s.phase == PHASE_RETREAT  : s.retreat_steps    += 1

        # ── REACH → PULL : presa confermata (soglie adattive + gate polso opz.) ──
        if s.phase == PHASE_REACH:
            grasp_cond = (
                gripper_action > g_thresh
                and is_physically_closed
                and dist_handle < d_thresh
                and wrist_align_ok
            )
            if grasp_cond:
                s.grasp_confirm_count += 1
            else:
                s.grasp_confirm_count = 0

            if s.grasp_confirm_count >= self._GRASP_CONFIRM_STEPS:
                s.phase = PHASE_PULL
                events.append(f"REACH→PULL (d={dist_handle:.3f}, g={gripper_action:.2f})")

        # ── PULL → HOLD_OPEN : porta aperta al goal (INVERSIONE vs chiusura) ──────
        elif s.phase == PHASE_PULL:
            # apertura: door_angle deve aver RAGGIUNTO il goal (entro tolleranza) MENTRE la
            # presa è chiusa. §1.30 — il gate di presa usa la soglia ADATTIVA g_thresh (come
            # REACH→PULL), NON il letterale 0.80 di prima. Diagnosi su 20 episodi reali: la
            # porta raggiunge SEMPRE il goal (open_error min ≈ 0), ma il grip-lock §1.18 floora
            # il comando a g_thresh + grip_lock_margin; per maniglie a bassa frizione questo
            # vale ~0.75 < 0.80, quindi con il vecchio gate la transizione NON scattava e
            # l'episodio restava bloccato in PULL fino all'orizzonte (8/20 falliti, tutti con
            # floor < 0.80). Allineando il gate alle soglie adattive l'incoerenza sparisce.
            opened_enough = (door_angle >= goal_angle - open_tol) and (gripper_action > g_thresh)
            if opened_enough:
                s.phase = PHASE_HOLD_OPEN
                s.hold_open_duration = 0
                events.append(f"PULL→HOLD_OPEN (angle={door_angle:.3f}, goal={goal_angle:.3f})")
            elif not is_physically_closed:
                # presa persa durante il PULL → torna a REACH
                s.phase = PHASE_REACH
                s.grasp_confirm_count = 0
                events.append(f"PULL→REACH (grip lost, width={gripper_width:.3f})")

        # ── HOLD_OPEN → RETREAT : mantenuto aperto per il tempo adattivo ──────────
        elif s.phase == PHASE_HOLD_OPEN:
            open_ok = door_angle >= goal_angle - open_tol
            if open_ok:
                s.hold_open_duration += 1
            else:
                # è ri-richiusa sotto il goal: scala il timer (specchio del bounce chiusura)
                penalty = int(abs(goal_angle - door_angle) / max(open_tol, 1e-6) * 5)
                s.hold_open_duration = max(0, s.hold_open_duration - penalty)

            hold_done = s.hold_open_duration >= s.target_hold_steps
            if hold_done:
                # §1.32 — fissa il target di ritiro lungo la normale (mirror ESATTO chiusura)
                if eef_pos is not None and door_quat_mujoco is not None:
                    s.retreat_pos = self.compute_retreat_pos(
                        eef_pos, door_quat_mujoco,
                        self.cfg.fsm_retreat_dist, self.cfg.fsm_retreat_z_off,
                        door_pos=door_pos,
                    )
                s.phase = PHASE_RETREAT
                events.append(
                    f"HOLD_OPEN→RETREAT (dur={s.hold_open_duration}, "
                    f"target={s.target_hold_steps})"
                )

        # PHASE_RETREAT: transizioni/terminazione gestite dall'environment (§1.17/§1.21)
        return events

    def reset(self):
        self.state.reset()