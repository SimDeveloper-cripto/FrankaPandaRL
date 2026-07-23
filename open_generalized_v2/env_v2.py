#!/usr/bin/env python3
# open_generalized_v2/env_v2.py
#
# AdvancedGeneralizedOpenDoorEnv — environment top-level dell'APERTURA generalizzata (v2),
# SPECULARE a close_generalized_v2/env_v2.py ma per il task di apertura, e tarato per il
# SOLO curriculum 1 (posa variabile + soglie adattive + fisica randomizzata).
#
# Pipeline di uno step (specchio della chiusura):
#   1) smoothing azione (EMA)                                            [come v1 apertura]
#   2) override DETERMINISTICI env-level a successo già acquisito:
#        §1.18 grip-lock in PULL/HOLD_OPEN  (blocca aperture accidentali della presa)
#        §1.17 rilascio pulito in RETREAT   (apri gripper, congela braccio fino a dita libere)
#        §1.21 rampa di avvio del ritiro     (scala 0→1 l'azione del braccio dopo il rilascio)
#   3) step del simulatore robosuite
#   4) lettura stato (door_angle, latch, presa) e update della FSM adattiva
#   5) reward potential-based + terminazione
#   6) info ricca (fase FSM, latch, door, goal, hold, curriculum) per test/log
#
# Inversione chiusura ↔ apertura: l'obiettivo è door_angle ≈ goal_angle (porta aperta al
# valore richiesto), non door_angle ≈ 0. Il goal è campionato a ogni reset (curriculum 1).
#
# Riferimenti: [1][2] opzioni/precondizioni, [3] shaping invariante, [8][17] domain rand,
#              [13] presa al contatto, [15] grasp 6-D.

from __future__ import annotations

import os
import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Any, Dict, List, Optional
from scipy.spatial.transform import Rotation as R_scipy

# Path setup PRIMA di qualsiasi import interno, così funziona sia lanciato come
# script (mjpython open_generalized_v2/train_curriculum_v2.py) sia come modulo (-m).
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import logging
from robosuite.utils.log_utils import ROBOSUITE_DEFAULT_LOGGER
ROBOSUITE_DEFAULT_LOGGER.setLevel(logging.ERROR)

# Import robusti: prima qualificati col package, poi fallback "piatti".
try:
    from open_generalized_v2.config_v2 import TrainConfigV2Open
    from open_generalized_v2.fsm_v2 import (AdaptiveFSMOpen, PHASE_REACH, PHASE_PULL,
                                            PHASE_HOLD_OPEN, PHASE_RETREAT)
    from open_generalized_v2.reward_v2 import PotentialBasedRewardOpen
    from open_generalized_v2.domain_rand_v2 import ExtendedDomainRandomizer
    from open_generalized_v2.grasp_strategy import MultiApproachGrasp
except ModuleNotFoundError:
    from config_v2 import TrainConfigV2Open
    from fsm_v2 import (AdaptiveFSMOpen, PHASE_REACH, PHASE_PULL,
                        PHASE_HOLD_OPEN, PHASE_RETREAT)
    from reward_v2 import PotentialBasedRewardOpen
    from domain_rand_v2 import ExtendedDomainRandomizer
    from grasp_strategy import MultiApproachGrasp


class AdvancedGeneralizedOpenDoorEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 20}

    def __init__(self, cfg: TrainConfigV2Open, render_mode: Optional[str] = None):
        super().__init__()
        self.cfg = cfg
        self.render_mode = render_mode
        self.curriculum_level = float(cfg.fixed_curriculum_level)

        import robosuite as suite
        from robosuite.controllers import load_composite_controller_config
        controller_config = load_composite_controller_config(controller="BASIC")

        self._rs_env = suite.make(
            env_name               = cfg.env_name,
            robots                 = cfg.robot,
            has_renderer           = (render_mode == "human"),
            has_offscreen_renderer = False,
            use_camera_obs         = False,
            use_object_obs         = True,
            reward_shaping         = False,
            horizon                = cfg.horizon,
            control_freq           = cfg.control_freq,
            controller_configs     = controller_config,
            ignore_done            = True,
        )
        sim = self._rs_env.sim

        # ── hinge della porta ──
        hinge = [n for n in sim.model.joint_names if ("door" in n.lower() and "hinge" in n.lower())]
        if not hinge:
            hinge = [n for n in sim.model.joint_names if "hinge" in n.lower()]
        self._door_hinge_name = hinge[0]
        jid = sim.model.joint_name2id(self._door_hinge_name)
        jmin, jmax = sim.model.jnt_range[jid]
        self._door_min = float(jmin)
        self._door_max = float(jmax)
        self._effective_max = float(min(self._door_max, self._door_min + cfg.door_open_cap_rad))
        self._door_hinge_qpos_adr = int(sim.model.jnt_qposadr[jid])
        self._door_hinge_dof_adr  = int(sim.model.jnt_dofadr[jid])

        # ── latch (handle joint) per lo stato di presa/aggancio ──
        latch = [n for n in sim.model.joint_names if "latch" in n.lower()]
        self._latch_qpos_adr = (
            int(sim.model.jnt_qposadr[sim.model.joint_name2id(latch[0])]) if latch else self._door_hinge_qpos_adr
        )

        # ── geom maniglia (per dist_handle e raggio) ──
        self.handle_geom_id = None
        for i, n in enumerate(sim.model.geom_names):
            if "handle" in n.lower():
                self.handle_geom_id = i
                break

        try:
            self.door_body_id = sim.model.body_name2id("Door_main")
        except Exception:
            self.door_body_id = 0

        # ── moduli v2 ──
        self._domain_rand   = ExtendedDomainRandomizer(cfg, sim.model)
        self._grasp_strategy = MultiApproachGrasp(cfg)
        self._fsm           = AdaptiveFSMOpen(cfg)
        self._reward_fn     = PotentialBasedRewardOpen(cfg, gamma=cfg.gamma)

        # ── default di stato necessari PRIMA del primo _flatten_obs ──
        # _flatten_obs() usa self._goal_angle per la feature di goal: va inizializzato
        # qui (default = apertura piena), poi reset() lo ricampiona a ogni episodio.
        self._goal_angle = self._effective_max

        # ── spazi ──
        obs = self._rs_env.reset()
        self._obs_keys = sorted(k for k, v in obs.items()
                                if isinstance(v, np.ndarray) and v.dtype != object and v.ndim == 1)
        flat = self._flatten_obs(obs)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=flat.shape, dtype=np.float32)
        low, high = self._rs_env.action_spec
        self.action_space = spaces.Box(low.astype(np.float32), high.astype(np.float32), dtype=np.float32)

        # ── stato episodio ──
        self._step_count = 0
        self._goal_angle = self._effective_max
        self._prev_door_angle: Optional[float] = None
        self._prev_action = None
        self._start_eef_pos: Optional[np.ndarray] = None
        # override deterministici (specchio della chiusura)
        self._prev_gripper_width = 0.08      # §1.17
        self._prev_is_phys_closed = False    # §1.18
        self._retreat_ramp_step = 0          # §1.21
        self._retreat_free_steps = 0         # §1.38: step di RETREAT DOPO il rilascio
        self._retreat_restore_done  = False  # §1.46: riporto attivo leva completato?
        self._retreat_restore_steps = 0      # §1.46: step spesi nel riporto
        self._retreat_escape_eef0   = None   # §1.46: ancora dell'escape (post-riporto)
        self._retreat_restore_cage   = False # §1.48: riporto in modalità GABBIA (morsa→gabbia)
        self._retreat_restore_latch0 = None  # §1.48: latch a inizio riporto (misura progresso)

    # ── helpers ──────────────────────────────────────────────────────────────────

    def set_curriculum_level(self, level: float) -> None:
        self.curriculum_level = float(level)

    def _door_angle(self) -> float:
        a = float(self._rs_env.sim.data.qpos[self._door_hinge_qpos_adr])
        return float(np.clip(a, self._door_min, self._effective_max))

    def _door_qvel(self) -> float:
        # §1.28 — velocità angolare del cardine della porta. Letta dal DOF del cardine,
        # simmetrica a _door_angle che legge il qpos dello stesso giunto. Serve al termine
        # hold_veldamp (damping anti-rimbalzo) in HOLD_OPEN/RETREAT, mirror ESATTO della
        # chiusura v2 (che osserva door_qvel e lo usa per fermare la porta sul bersaglio).
        return float(self._rs_env.sim.data.qvel[self._door_hinge_dof_adr])

    def _latch_qpos(self) -> float:
        return float(self._rs_env.sim.data.qpos[self._latch_qpos_adr])

    def _gripper_width_live(self) -> float:
        """§1.38 — larghezza REALE del gripper al passo corrente (non il proxy
        _prev_gripper_width, in ritardo di un passo). Serve al gate 'dita libere' del RETREAT."""
        try:
            obs = self._rs_env._get_observations()
            gq  = obs.get("robot0_gripper_qpos")
            if gq is not None:
                return float(np.sum(np.abs(gq)))
        except Exception:
            pass
        return float(self._prev_gripper_width)

    def _latch_restore_dir(self):
        """§1.46 — direzione (mondo) che RIPORTA la leva verso latch=0: tangente all'arco
        della leva nel punto eef. In MuJoCo il qpos di un hinge cresce con la regola della
        mano destra attorno all'asse del giunto (data.xaxis, ancora in data.xanchor): la
        velocità del punto afferrato per latch DECRESCENTE è v = (-axis) × r, con
        r = eef − anchor. Nessuna stima geometrica: assi e ancora letti dal simulatore."""
        try:
            jid = getattr(self._domain_rand, "latch_joint_id", None)
            if jid is None:
                return None
            sim    = self._rs_env.sim
            axis   = np.asarray(sim.data.xaxis[jid],   dtype=float)
            anchor = np.asarray(sim.data.xanchor[jid], dtype=float)
            r = self._eef_pos() - anchor
            v = -np.cross(axis, r)
            sgn = -1.0
            if self._latch_qpos() < 0.0:      # leva oltre lo zero: verso opposto
                v = -v
                sgn = 1.0
            n = float(np.linalg.norm(v))
            if n < 1e-8:
                return None
            # §1.49 — ritorna anche l'asse di rotazione FIRMATO (ω̂ = sgn·axis): il riporto
            # è un moto RIGIDO attorno all'asse del latch, quindi serve sia la traslazione
            # tangente sia la rotazione del polso coerente (v. nota nel ramo RETREAT).
            return v / n, (sgn * axis)
        except Exception:
            return None

    def _eef_pos(self) -> np.ndarray:
        eef_site = self._rs_env.robots[0].eef_site_id
        sid = eef_site.get('right', list(eef_site.values())[0]) if isinstance(eef_site, dict) else eef_site
        return np.array(self._rs_env.sim.data.site_xpos[sid], dtype=float)

    def _handle_pos(self) -> np.ndarray:
        if self.handle_geom_id is not None:
            return np.array(self._rs_env.sim.data.geom_xpos[self.handle_geom_id], dtype=float)
        return self._eef_pos()

    def _flatten_obs(self, obs: dict) -> np.ndarray:
        parts = [obs[k].ravel().astype(np.float32) for k in self._obs_keys]
        base = np.concatenate(parts, axis=0)
        eef_quat = obs.get("robot0_eef_quat")
        eef_pos  = obs.get("robot0_eef_pos", np.zeros(3))
        handle_pos = obs.get("handle_pos", obs.get("door_handle_pos", eef_pos))
        try:
            dq = self._rs_env.sim.model.body_quat[self.door_body_id]
        except Exception:
            dq = None
        grasp_feats = self._grasp_strategy.obs_features(eef_quat, np.asarray(handle_pos), np.asarray(eef_pos), dq)
        physics_feats = self._domain_rand.obs_features()
        # §1.34 — FASE FSM nell'osservazione (mirror ESATTO della chiusura, che la ha sempre
        # avuta). Senza il one-hot la policy non può distinguere HOLD_OPEN (tieni) da RETREAT
        # (molla e allontanati): stesso stato fisico, azioni ottime opposte → il braccio
        # restava attaccato alla maniglia qualunque fosse il reward del RETREAT.
        fsm_onehot = self._fsm.state.one_hot
        # feature di goal (apertura): goal normalizzato nel range effettivo
        goal_norm = np.array([
            (self._goal_angle - self._door_min) / (self._effective_max - self._door_min + 1e-8)
        ], dtype=np.float32)
        return np.concatenate([base, grasp_feats, physics_feats, fsm_onehot, goal_norm], axis=0)

    # ── reset ─────────────────────────────────────────────────────────────────────

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        obs = self._rs_env.reset()

        # domain randomization fisica (sempre attiva)
        self._domain_rand.randomize_episode(self.curriculum_level)

        # curriculum 1: campiona un GOAL di apertura variabile nel range alto
        f = np.random.uniform(self.cfg.goal_frac_min, self.cfg.goal_frac_max)
        self._goal_angle = float(self._door_min + f * (self._effective_max - self._door_min))

        self._fsm.reset()
        self._reward_fn.reset()
        self._step_count = 0
        self._prev_door_angle = self._door_angle()
        self._prev_action = None
        self._start_eef_pos = self._eef_pos().copy()
        self._prev_gripper_width = 0.08
        self._prev_is_phys_closed = False
        self._retreat_ramp_step = 0
        self._retreat_free_steps = 0
        self._retreat_eef0 = None
        self._retreat_restore_done  = False   # §1.46
        self._retreat_restore_steps = 0       # §1.46
        self._retreat_escape_eef0   = None    # §1.46
        self._retreat_restore_cage   = False  # §1.48
        self._retreat_restore_latch0 = None   # §1.48
        info = {
            "goal_angle": self._goal_angle,
            "door_min": self._door_min,
            "effective_max": self._effective_max,
            "curriculum_level": self.curriculum_level,
        }
        return self._flatten_obs(obs), info

    # ── step ──────────────────────────────────────────────────────────────────────

    def step(self, action: np.ndarray):
        action = np.clip(np.asarray(action, dtype=np.float32).copy(), -1.0, 1.0)
        if self._prev_action is None:
            self._prev_action = np.zeros_like(action)
        a = float(self.cfg.action_smooth_alpha)
        action = a * action + (1.0 - a) * self._prev_action

        dr = self._domain_rand
        handle_radius = dr.current_handle_radius
        handle_friction = dr.current_handle_friction

        # presa fisica corrente (specchio chiusura): dita attorno alla maniglia
        gw = self._prev_gripper_width
        handle_diam = handle_radius * 2.0
        is_phys_closed = (gw <= handle_diam + 0.025) and (gw >= 0.015)

        phase = self._fsm.state.phase

        # azzera la rampa quando NON siamo in RETREAT (§1.21)
        if phase != PHASE_RETREAT:
            self._retreat_ramp_step = 0

        # ── §1.18 GRIP-LOCK in PULL e HOLD_OPEN ──
        if phase in (PHASE_PULL, PHASE_HOLD_OPEN):
            if getattr(self.cfg, "grip_lock_enabled", True) and self._prev_is_phys_closed:
                grip_floor = min(1.0, self._fsm.grip_thresh(handle_friction) + self.cfg.grip_lock_margin)
                action[-1] = max(float(action[-1]), grip_floor)

        # ── §1.22 ACCOMPAGNA LEVA + §1.17 RILASCIO PULITO + §1.21 RAMPA in RETREAT ──
        elif phase == PHASE_RETREAT:
            # §1.41-diag — memorizza la posa eef all'INGRESSO in RETREAT per misurare quanto
            # il braccio arretra davvero (l'unica variabile che mancava alla traccia).
            if getattr(self, "_retreat_eef0", None) is None:
                self._retreat_eef0 = self._eef_pos().copy()
            # §1.51 — GUARDIA DI VELOCITÀ DELLA PORTA (anti-bounce, sicurezza). Se la porta
            # sta muovendo (|dθ/dt| oltre una piccola soglia) l'azione cartesiana INIETTATA
            # dall'env (riporto/escape) viene SCALATA giù fino a un pavimento: il braccio
            # "molla" invece di inseguire e la porta non entra in avanti-indietro. La
            # traccia post-training mostra lo spike da spezzare (door_qvel −0.54 a rs1).
            # _door_qvel() legge la velocità del cardine PRIMA dello step → reagisce al
            # movimento in corso. Vale sia in riporto sia in escape (stesso frame OSC).
            _qv_ref  = float(getattr(self.cfg, "retreat_door_qvel_ref", 0.15))
            _qv_flr  = float(getattr(self.cfg, "retreat_door_qvel_floor", 0.25))
            _qvel_damp = float(np.clip(1.0 - abs(self._door_qvel()) / max(_qv_ref, 1e-6),
                                       _qv_flr, 1.0))
            # §1.38 — CAUSA MISURATA (traccia diagnostica reale): il RETREAT (30 step) è
            # troppo CORTO per la sua sequenza fisica. La traccia mostra: gripper che parte a
            # width~0.046 e apre LENTO (supera la soglia 'dita libere' solo verso lo step ~10),
            # leva a FONDO-CORSA (+1.571) che a fine RETREAT è ancora ~1.25 perché le dita la
            # tengono, braccio che non ha tempo di allontanarsi. Non è un rimbalzo passivo
            # (misurato in MuJoCo: leva che scarica da 1.571 muove la porta di 0.0004 rad;
            # nessun contatto latch-telaio a porta aperta) né trascinamento del braccio: è il
            # BUDGET DI TEMPO. Fix: (a) gate 'dita libere' sulla larghezza REALE del passo
            # corrente e braccio FERMO finché non è vero (niente trascinamento); (b) la durata
            # utile parte DOPO il rilascio (_retreat_free_steps); (c) target allungato (config).
            # §1.46 — RIPORTO ATTIVO DELLA LEVA (prima del rilascio, presa CHIUSA).
            # Diag §1.45: con z_off=0 la porta resta al goal ma lo sfilamento sotto CARICO
            # è friction-limited: la molla (θ≈1.35) preme la barra sul dito e il braccio
            # si arresta esponenzialmente a ~3.7 cm (equilibrio forza OSC ↔ attrito), leva
            # mai a casa (0/5). Il §1.22 (attesa passiva a braccio congelato) era il
            # tentativo giusto sbagliato: la leva non può tornare MENTRE è impugnata da un
            # braccio FERMO. Qui il braccio la ACCOMPAGNA lungo il suo arco (direzione
            # esatta da xaxis/xanchor, vedi _latch_restore_dir) finché la molla è scarica
            # (|latch| ≤ tol) → il successivo rilascio+escape avviene SENZA carico.
            # È la sequenza della chiusura: porta ferma → maniglia riportata → stacco.
            _rst_on  = bool(getattr(self.cfg, "retreat_restore_enabled", True))
            _rst_tol = float(getattr(self.cfg, "retreat_restore_tol", 0.35))
            _rst_cap = int(getattr(self.cfg, "retreat_restore_max_steps", 20))
            _restoring = False
            if _rst_on and not self._retreat_restore_done:
                _latch_now = abs(self._latch_qpos())
                if _latch_now <= _rst_tol or self._retreat_restore_steps >= _rst_cap:
                    self._retreat_restore_done = True
                    # §1.47 — retreat_pos era ancorato all'eef d'INGRESSO in RETREAT, ma il
                    # riporto ha spostato l'eef di ~10 cm lungo l'arco della leva: quel
                    # target ormai punta ALL'INDIETRO (misurato, ep.2 §1.46: ARRETRATO che
                    # cala 0.100→0.096 e latch che RISALE 0.68→0.98 — l'escape riportava il
                    # dito dentro la leva). Invalidandolo qui, il blocco post-step lo
                    # RICALCOLA dalla posa corrente (fine riporto) lungo la normale attuale.
                    self._fsm.state.retreat_pos = None
                else:
                    _dr = self._latch_restore_dir()
                    if _dr is None:
                        self._retreat_restore_done = True   # dati giunto non disponibili
                        self._fsm.state.retreat_pos = None  # §1.47 (v. sopra)
                    else:
                        _dir, _omega = _dr
                        _restoring = True
                        if self._retreat_restore_steps == 0:
                            self._retreat_restore_latch0 = _latch_now
                        self._retreat_restore_steps += 1
                        # §1.48 — MORSA→GABBIA. Caso misurato (ep.4 §1.47, maniglia grossa
                        # r=0.0252): seguire l'arco a presa chiusa richiede che la barra
                        # RUOTI DENTRO la pinza (polso bloccato); su una barra spessa la
                        # coppia d'attrito della morsa vince la spinta OSC → latch a
                        # 0.001 rad/step (30x più lento), riporto inutile, poi stallo.
                        # Se il progresso è insufficiente, le dita si SEMIAPRONO appena
                        # oltre il diametro: la barra ruota libera nella gabbia e il dito
                        # la ACCOMPAGNA lungo l'arco spingendola — niente slip richiesto.
                        _cg_after = int(getattr(self.cfg, "retreat_restore_cage_after", 12))
                        _cg_prog  = float(getattr(self.cfg, "retreat_restore_cage_progress", 0.10))
                        if (not self._retreat_restore_cage
                                and self._retreat_restore_steps >= _cg_after
                                and self._retreat_restore_latch0 is not None
                                and (self._retreat_restore_latch0 - _latch_now) < _cg_prog):
                            self._retreat_restore_cage = True
                        _gain = float(getattr(self.cfg, "retreat_restore_gain", 2.0))
                        _mag  = float(min(0.6, _gain * _latch_now))
                        # §1.51 — avvio MORBIDO del riporto (rampa 0→1 sui primi step) +
                        # guardia door_qvel: elimina lo strappo alla porta all'ingresso in
                        # RETREAT (la morsa parte con presa chiusa, massimo accoppiamento
                        # col cardine). La rotazione del polso sotto eredita _mag, quindi
                        # si smorza in modo coerente.
                        _rr = int(getattr(self.cfg, "retreat_restore_ramp", 4))
                        _restore_ramp = 1.0 if _rr <= 0 else min(1.0, float(self._retreat_restore_steps) / float(_rr))
                        _mag *= _restore_ramp * _qvel_damp
                        action[:3] = np.clip(_dir * _mag, -1.0, 1.0)
                        # §1.49 — ROTAZIONE DEL POLSO coerente col moto rigido attorno
                        # all'asse del latch. Prima le rotazioni erano AZZERATE → l'OSC
                        # teneva l'orientazione RIGIDA mentre la traslazione seguiva
                        # l'arco: un vincolo contraddittorio che nelle pose scomode
                        # (ep.4 §1.48: 5 mm in 13 step pur comandando 0.6) pianta il
                        # braccio. Il comando axis-angle segue ω̂ = -axis (verso che
                        # riduce |latch|), stessa fonte MuJoCo di v = ω̂ × r.
                        if action.shape[0] >= 7:
                            _rg = float(getattr(self.cfg, "retreat_restore_rot_gain", 0.5))
                            action[3:6] = np.clip(_omega * (_rg * _mag), -1.0, 1.0)
                        # §1.50 — GABBIA SEMPRE (default). Trace §1.49 ep.1: in morsa le
                        # dita SCIVOLANO dalla barra durante il moto d'arco e la pinza
                        # speed-based si chiude A PUGNO nel vuoto (width misurata 0.0012);
                        # la leva viene accompagnata dalle nocche ma la porta oscilla di
                        # ±0.13 rad (RIMBALZO 0.086-0.143). La gabbia in retroazione —
                        # la stessa che ha risolto ep.4 — cradle la barra senza stringerla:
                        # niente pugno, niente trascinamento.
                        if (self._retreat_restore_cage
                                or getattr(self.cfg, "retreat_restore_cage_always", True)):
                            # §1.49 — GABBIA IN RETROAZIONE. Il gripper robosuite è
                            # SPEED-BASED (l'azione è una direzione, non una posizione):
                            # il comando costante del §1.48 lo apriva FINO IN FONDO
                            # (misurato: width 0.081 ≫ target 0.065). Bang-bang sulla
                            # larghezza REALE: tiene le dita a ~diam+margine, la barra
                            # ruota libera nella gabbia mentre il dito la accompagna.
                            _w_tgt = float(handle_diam + getattr(self.cfg, "retreat_restore_cage_margin", 0.015))
                            _gw_live = self._gripper_width_live()
                            action[-1] = 1.0 if _gw_live > _w_tgt else -1.0
                        else:
                            action[-1] = 1.0     # MORSA: accompagna, non rilasciare
                        self._retreat_ramp_step  = 0
                        self._retreat_free_steps = 0

            gw_now        = self._gripper_width_live()
            fingers_clear = gw_now > (handle_diam + self.cfg.retreat_clear_margin)

            if _restoring:
                pass   # azione già impostata dal riporto §1.46
            elif getattr(self.cfg, "retreat_clean_release", True) and not fingers_clear:
                action[:-1] = 0.0      # braccio fermo: non trascinare la porta
                action[-1]  = -1.0     # gripper aperto → rilascio
                self._retreat_ramp_step = 0
                self._retreat_free_steps = 0
            else:
                # dita libere: da qui parte la durata utile; gripper aperto.
                self._retreat_free_steps = getattr(self, "_retreat_free_steps", 0) + 1
                action[-1] = -1.0

                # §1.43 — ESCAPE GUIDATO post-rilascio (env-level, deterministico, mirror
                # metodologico di §1.17/§1.18). CAUSA MISURATA (traccia RETREAT, 5/5 episodi):
                # dopo il rilascio la policy arretra solo ~0.10 m e si ferma; §1.41 ha misurato
                # che per liberare la leva servono fino a 0.128 m → le punte delle dita restano
                # nel piano di rotazione della leva, che torna da 1.57 a ~0.5 rad e SI APPOGGIA
                # sul dito (residuo libero misurato ≤0.20): incastro. La molla, premendo sul
                # dito, SPINGE la porta (door_qvel +0.16 al rilascio). Con la terminazione
                # §1.42 gatata su |latch|<0.08 l'episodio muore sempre al hard-cap (RETREAT=201)
                # e il reddito per-step del RETREAT rende l'incastro la strategia OTTIMA.
                # Fix: finché la leva NON è tornata neutra E il braccio NON si è allontanato di
                # almeno retreat_escape_dist, l'azione del braccio è GUIDATA verso retreat_pos
                # (normale della porta, §1.40) — stessa tecnica della mano guidata del
                # diagnostico. È motion-quality a successo già acquisito: zero reward, zero
                # retraining necessario. Poi rampa §1.21 e policy, come prima.
                _esc_on   = bool(getattr(self.cfg, "retreat_escape_enabled", True))
                _esc_dist = float(getattr(self.cfg, "retreat_escape_dist", 0.15))
                _lt_tol   = float(getattr(self.cfg, "retreat_latch_neutral_tol", 0.05))
                _rp       = getattr(self._fsm.state, "retreat_pos", None)
                _eef_now  = self._eef_pos()
                # §1.46 — l'escape si misura dal punto di FINE RIPORTO, non dall'ingresso
                # in RETREAT: il riporto sposta l'eef lungo l'arco della leva (fino a
                # ~15 cm) e col vecchio ancoraggio il gate `moved < escape_dist` sarebbe
                # già scaduto prima ancora di iniziare a sfilarsi.
                if getattr(self, "_retreat_escape_eef0", None) is None:
                    self._retreat_escape_eef0 = _eef_now.copy()
                _moved    = float(np.linalg.norm(_eef_now - self._retreat_escape_eef0))
                _latch_now = abs(self._latch_qpos())
                if _esc_on and _rp is not None and _moved < _esc_dist and _latch_now > _lt_tol:
                    _dir = np.asarray(_rp, dtype=np.float32) - _eef_now.astype(np.float32)
                    _n   = float(np.linalg.norm(_dir))
                    if _n > 1e-6:
                        _gain = float(getattr(self.cfg, "retreat_escape_gain", 5.0))
                        # §1.52 — NIENTE guardia door_qvel qui: durante lo SFILAMENTO la porta
                        # è già rilasciata e il braccio si muove lungo la normale (VIA dalla
                        # porta), quindi non la può far rimbalzare. Con la guardia (come in §1.51)
                        # il comando veniva frenato dalle micro-oscillazioni di qvel (RIMBALZO=0
                        # ma dθ/dt≈±0.02–0.07 per rumore di contatto) → il braccio arretrava solo
                        # ~0.03 m prima della terminazione: allontanamento quasi invisibile. La
                        # guardia RESTA nel riporto (righe sopra, presa chiusa = accoppiamento
                        # reale col cardine, dove il rimbalzo NASCE). Qui: comando pieno → il
                        # braccio arretra ~0.10 m come nei run pre-§1.51 (RIMBALZO comunque ~0).
                        action[:3]  = np.clip(_dir / _n * min(1.0, _gain * _n), -1.0, 1.0)
                        if action.shape[0] > 4:
                            action[3:-1] = 0.0   # niente torsioni del polso durante l'escape
                    self._retreat_ramp_step = 0   # la rampa §1.21 parte DOPO l'escape
                else:
                    # braccio libero (ritiro pagato dal reward ret_dir), avvio morbido §1.21.
                    _R = int(getattr(self.cfg, "retreat_rampup_steps", 0))
                    if getattr(self.cfg, "retreat_rampup_enabled", True) and _R > 0 \
                            and self._retreat_ramp_step < _R:
                        _scale = float(self._retreat_ramp_step + 1) / float(_R)
                        action[:-1] = action[:-1] * _scale
                        self._retreat_ramp_step += 1

        self._prev_action = action.copy()

        # ── step simulatore ──
        obs, _, rs_done, _ = self._rs_env.step(action)
        self._step_count += 1

        door_angle = self._door_angle()
        prev_angle = float(self._prev_door_angle) if self._prev_door_angle is not None else door_angle
        latch_qpos = self._latch_qpos()
        door_qpos = float(self._rs_env.sim.data.qpos[self._door_hinge_qpos_adr])

        # dist_handle dall'OSSERVAZIONE robosuite (come l'env di chiusura che funziona:
        # env_gen.py righe 70-72). I metodi site_xpos/geom_xpos davano un handle_geom
        # errato (primo geom con "handle" nel nome ≠ centro afferrabile), con dist ~0.3 m
        # costante → REACH→PULL mai raggiunto. Fallback ai metodi solo se le chiavi mancano.
        obs_eef    = obs.get("robot0_eef_pos", None)
        obs_handle = obs.get("handle_pos", obs.get("door_handle_pos", None))
        if obs_eef is not None and obs_handle is not None:
            eef_pos    = np.asarray(obs_eef, dtype=float)
            handle_pos = np.asarray(obs_handle, dtype=float)
            _handle_src = "obs"
        else:
            eef_pos    = self._eef_pos()
            handle_pos = self._handle_pos()
            _handle_src = "fallback_methods"
        dist_handle = float(np.linalg.norm(eef_pos - handle_pos))
        # componenti per i termini densi di REACH (mirror chiusura): xy e dislivello z
        dist_xy     = float(np.linalg.norm(eef_pos[:2] - handle_pos[:2]))
        height_diff = float(eef_pos[2] - handle_pos[2])

        gripper_qpos = obs.get("robot0_gripper_qpos")
        gripper_width = float(np.sum(np.abs(gripper_qpos))) if gripper_qpos is not None else gw
        is_phys_closed = (gripper_width <= handle_diam + 0.025) and (gripper_width >= 0.015)

        # gate del polso (riusa la geometria del grasp; opzionale e robusto)
        wrist_align_ok = True

        prev_phase = self._fsm.state.phase
        try:
            # §1.33 — orientazione CORRENTE della porta: model.body_quat è STATICA (porta
            # chiusa); a porta aperta di door_angle la normale vera è ruotata di Rz(+door_angle)
            # (misurato: 20.1° di errore a 0.35 rad usando la statica → il ritiro scivolava di
            # lato, il dito restava sotto la leva e la bloccava). Nella CHIUSURA la statica va
            # bene perché al RETREAT la porta È chiusa (door≈0 → correzione nulla).
            _q_static  = self._rs_env.sim.model.body_quat[self.door_body_id]
            _door_quat = self._fsm.rotate_quat_z_mujoco(_q_static, door_angle)
            # §1.40 — posizione del pannello: serve a ORIENTARE la normale verso il robot
            # (invece di assumerne il verso). Usa la posa CORRENTE del corpo nel sim.
            _door_pos  = np.array(self._rs_env.sim.data.body_xpos[self.door_body_id], dtype=float)
        except Exception:
            _door_quat = None
            _door_pos  = None
        fsm_events = self._fsm.update(
            door_angle           = door_angle,
            goal_angle           = self._goal_angle,
            open_tol             = self.cfg.open_tol_rad,
            gripper_action       = float(action[-1]),
            dist_handle          = dist_handle,
            handle_radius        = handle_radius,
            handle_friction      = handle_friction,
            is_physically_closed = is_phys_closed,
            gripper_width        = gripper_width,
            prev_angle           = prev_angle,
            control_freq         = self.cfg.control_freq,
            door_qpos            = door_qpos,
            latch_stiffness      = dr.current_latch_stiffness,
            base_latch_stiffness = dr.base_latch_stiffness or 1.0,
            eef_pos              = eef_pos,
            door_quat_mujoco     = _door_quat,
            door_pos             = _door_pos,
            wrist_align_ok       = wrist_align_ok,
            beta_probs           = None,
        )

        just_succeeded = (prev_phase == PHASE_PULL and self._fsm.state.phase == PHASE_HOLD_OPEN)

        # §1.32 — retreat_pos: fissato dalla FSM alla transizione HOLD_OPEN→RETREAT come
        # back-off lungo la NORMALE della porta (compute_retreat_pos, mirror chiusura).
        # Fallback robusto (quaternione mancante in quel frame): calcolo qui, stessa formula.
        if self._fsm.state.phase == PHASE_RETREAT and self._fsm.state.retreat_pos is None:
            if _door_quat is not None:
                self._fsm.state.retreat_pos = self._fsm.compute_retreat_pos(
                    eef_pos, _door_quat, self.cfg.fsm_retreat_dist, self.cfg.fsm_retreat_z_off,
                    door_pos=_door_pos)
            else:
                rp = np.asarray(eef_pos, dtype=np.float32) + np.array(
                    [-self.cfg.fsm_retreat_dist, 0.0, self.cfg.fsm_retreat_z_off], dtype=np.float32)
                self._fsm.state.retreat_pos = rp
        dist_retreat = (
            float(np.linalg.norm(eef_pos - self._fsm.state.retreat_pos))
            if self._fsm.state.retreat_pos is not None else 1.0
        )

        # contatore return_hold: step consecutivi entro return_pos_tol (mirror chiusura)
        if self._fsm.state.phase == PHASE_RETREAT and self._fsm.state.retreat_pos is not None:
            if dist_retreat < self.cfg.return_pos_tol:
                self._fsm.state.return_hold += 1
            else:
                self._fsm.state.return_hold = 0

        # §1.38 — porta la durata post-rilascio sullo stato FSM così il reward può usarla per
        # la terminazione senza cambiare firma (retreat_steps totali resta invariato per la
        # diagnostica e la simmetria con la chiusura).
        self._fsm.state.retreat_free_steps = int(getattr(self, "_retreat_free_steps", 0))
        # §1.46 — il reward sospende i termini di rilascio/direzione durante il riporto
        self._fsm.state.retreat_restoring = bool(
            self._fsm.state.phase == PHASE_RETREAT
            and getattr(self.cfg, "retreat_restore_enabled", True)
            and not getattr(self, "_retreat_restore_done", True)
        )
        reward, terminated, truncated, rew_info = self._reward_fn.compute(
            fsm_state      = self._fsm.state,
            phase_consts   = (PHASE_REACH, PHASE_PULL, PHASE_HOLD_OPEN, PHASE_RETREAT),
            door_angle     = door_angle,
            door_qvel      = self._door_qvel(),
            goal_angle     = self._goal_angle,
            door_min       = self._door_min,
            open_tol       = self.cfg.open_tol_rad,
            prev_angle     = prev_angle,
            gripper_action = float(action[-1]),
            grip_thresh    = self._fsm.grip_thresh(handle_friction),
            dist_handle    = dist_handle,
            dist_xy        = dist_xy,
            height_diff    = height_diff,
            dist_retreat   = dist_retreat,
            eef_pos        = eef_pos,
            target_steps   = self._fsm.state.target_hold_steps or 30,
            curriculum_lvl = self.curriculum_level,
            is_physically_closed = is_phys_closed,
            action         = action,
            latch_qpos     = latch_qpos,
            just_succeeded = just_succeeded,
            rs_done        = bool(rs_done),
            step_count     = self._step_count,
            horizon        = self.cfg.horizon,
        )

        self._prev_door_angle = door_angle
        self._prev_gripper_width = float(gripper_width)
        self._prev_is_phys_closed = bool(is_phys_closed)

        # info ricca (stessa interfaccia dei test v2: nomi speculari)
        info: Dict[str, Any] = {}
        info["is_success"]        = self._fsm.state.phase in (PHASE_HOLD_OPEN, PHASE_RETREAT)
        info["door_angle"]        = door_angle
        info["door_qpos"]         = door_qpos
        info["goal_angle"]        = self._goal_angle
        info["open_error"]        = float(abs(self._goal_angle - door_angle))
        info["latch_qpos"]        = latch_qpos
        info["dist_handle"]       = dist_handle
        info["eef_pos"]           = [round(float(x),4) for x in np.asarray(eef_pos).ravel()[:3]]
        info["handle_pos"]        = [round(float(x),4) for x in np.asarray(handle_pos).ravel()[:3]]
        info["vec_eef_to_handle"] = [round(float(x),4) for x in (np.asarray(handle_pos).ravel()[:3] - np.asarray(eef_pos).ravel()[:3])]
        info["handle_src"]        = _handle_src
        info["obs_keys_sample"]   = self._obs_keys[:12]
        info["fsm_phase"]         = self._fsm.state.phase
        info["door_qvel"]         = float(self._door_qvel())
        info["gripper_width"]     = float(gripper_width)
        info["retreat_steps"]     = int(self._fsm.state.retreat_steps)
        info["retreat_free_steps"]= int(getattr(self, "_retreat_free_steps", 0))
        info["retreat_restoring"] = bool(getattr(self._fsm.state, "retreat_restoring", False))
        _e0 = getattr(self, "_retreat_eef0", None)
        info["retreat_moved"]     = (float(np.linalg.norm(self._eef_pos() - _e0))
                                     if _e0 is not None else 0.0)
        info["door_qvel"]         = float(self._door_qvel())
        info["gripper_width"]     = float(gripper_width)
        info["retreat_steps"]     = int(self._fsm.state.retreat_steps)
        info["fsm_phase_name"]    = self._fsm.state.phase_name
        info["hold_duration"]     = self._fsm.state.hold_open_duration
        info["target_hold_steps"] = self._fsm.state.target_hold_steps
        info["curriculum_level"]  = self.curriculum_level
        info["fsm_events"]        = fsm_events
        info["reward_terms"]      = rew_info

        return self._flatten_obs(obs), reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            return self._rs_env.render()
        return None

    def close(self):
        self._rs_env.close()