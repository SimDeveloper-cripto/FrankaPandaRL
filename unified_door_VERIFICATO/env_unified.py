#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
env_unified.py — l'ambiente Gymnasium unico per i due compiti. ESEGUIBILE.

Documento di riferimento: tabella_reward_machine_unificata.md
  §6  lo spazio di osservazione unificato: 126 dimensioni
  §8  le differenze geometriche del punto di ritiro
  §6    randomizzazione del dominio e moduli esistenti: INVARIATI

I moduli che il §6 dichiara invariati (MultiApproachGrasp, ExtendedDomainRandomizer)
NON sono riscritti: sono importati dal progetto di chiusura. Se non li trova,
l'ambiente funziona lo stesso con sostituti neutri, così il codice resta
eseguibile anche fuori dalla cartella dei progetti.

[Tobin et al. 2017]   domain randomization + parametri fisici in osservazione
[ten Pas et al. 2017] feature di presa multi-approccio
"""
import os
import sys
from typing import Optional, Dict, Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from config_unified import UnifiedConfig
from state_unified import DoorState
from fsm_unified import UnifiedFSM, Fase, Controllore
from reward_unified import UnifiedReward

# ── §6: i moduli invarianti, importati e non riscritti ──────────────────────
_ORIG = os.environ.get("PROGETTI_ORIGINALI", "")
if _ORIG and _ORIG not in sys.path:
    sys.path.insert(0, _ORIG)
try:
    from close_generalized_v2.grasp_strategy import MultiApproachGrasp
    from close_generalized_v2.domain_rand_v2 import ExtendedDomainRandomizer
    MODULI_ORIGINALI = True
except Exception as _e:                              # sostituti neutri
    MODULI_ORIGINALI = False
    # Il fallback e' comodo per i test, ma se scatta durante un addestramento
    # vero le feature di presa e il contesto fisico sarebbero tutti zeri. Meglio
    # dirlo forte, con la diagnosi del perche'.
    _perche = (f"PROGETTI_ORIGINALI non impostata" if not _ORIG
               else f"{_ORIG!r} non contiene close_generalized_v2/"
               if not os.path.isdir(os.path.join(_ORIG, "close_generalized_v2"))
               else f"manca train_close.py in {_ORIG!r}"
               if not os.path.exists(os.path.join(_ORIG, "train_close.py"))
               else f"{type(_e).__name__}: {_e}")
    print("\n" + "!" * 78)
    print("  ATTENZIONE: i moduli originali NON sono stati caricati.")
    print(f"  Causa: {_perche}")
    print("  Sto usando sostituti neutri: le feature di presa e il contesto")
    print("  fisico saranno tutti ZERI. Va bene per i test, NON per addestrare.")
    print("  PROGETTI_ORIGINALI deve puntare alla cartella che contiene INSIEME:")
    print("      close_generalized_v2/   open_generalized_v2/   train_close.py   config/")
    print("!" * 78 + "\n")

    class MultiApproachGrasp:                        # type: ignore
        def __init__(self, cfg): self.n_candidates = 3
        def obs_features(self, *a, **k): return np.zeros(4, dtype=np.float32)

    class ExtendedDomainRandomizer:                  # type: ignore
        def __init__(self, cfg, sim_model):
            self.current_handle_radius = 0.02
            self.current_handle_friction = 1.0
            self.current_latch_stiffness = 1.0
            self.base_latch_stiffness = 1.0
        def randomize_episode(self, curriculum_level): pass
        def obs_features(self): return np.zeros(3, dtype=np.float32)


# §6 — LA COMPOSIZIONE DELL'OSSERVAZIONE UNIFICATA
#
# | componente                        | chius. | apert. | unif. |
# |-----------------------------------|--------|--------|-------|
# | chiavi robosuite vettoriali       |   112  |   112  |  112  |
# | feature di presa multi-approccio  |     4  |     4  |    4  |
# | contesto fisico                   |     3  |     3  |    3  |
# | one-hot della fase FSM            |     4  |     4  |    4  |
# | scalari handle_qpos, hinge_qpos   |     2  |     0  |    0  |  <- imballaggio
# | distanza mano-maniglia, cerniera  |     2  |     0  |    0  |  <- imballaggio
# | raggio e attrito della maniglia   |     2  |     0  |    2  |  <- informazione reale
# | bersaglio θ* normalizzato         |     0  |     1  |    1  |  <- parametro di compito
# | TOTALE                            |   129  |   124  |  126  |
OBS_DIM = 126
OBS_BLOCCHI = (("robosuite_vettoriali", 112), ("presa_multi_approccio", 4),
               ("contesto_fisico", 3), ("fase_one_hot", 4),
               ("raggio_attrito", 2), ("bersaglio_norm", 1))


class UnifiedDoorEnv(gym.Env):
    """Un ambiente, due compiti. La differenza è tutta in `cfg.task` (§6)."""

    metadata = {"render_modes": ["human"], "render_fps": 20}

    def __init__(self, cfg: UnifiedConfig, render_mode: Optional[str] = None):
        super().__init__()
        self.cfg = cfg
        self.render_mode = render_mode

        self.door = DoorState(theta_star=0.0, theta_zero=0.0, tol=cfg.task.tol)
        self.fsm = UnifiedFSM(cfg)
        self.reward = UnifiedReward(cfg)

        self.action_space = spaces.Box(-1.0, 1.0, shape=(7,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(OBS_DIM,), dtype=np.float32)

        self._rs = None
        self._obs_keys = None
        self._passi = 0
        self._contatto_prec = False
        self._larghezza = 0.0
        self._latch = 0.0
        self._passi_riporto = 0
        self._retreat_pos: Optional[np.ndarray] = None
        self._eef_rilascio: Optional[np.ndarray] = None   # §7 posa al rilascio
        self._ritiro_spostato: float = 0.0                # §7 spostamento massimo
        self._eef = np.zeros(3)
        self._handle = np.zeros(3)
        self._prev_cmd = None

        self._costruisci_robosuite()

    # ══ robosuite: stessi parametri dei due progetti (§6) ═══════════════
    def _costruisci_robosuite(self) -> None:
        import robosuite as suite
        from robosuite.controllers import load_composite_controller_config
        h = self.cfg.sac
        self._rs = suite.make(
            env_name="Door", robots="Panda",
            has_renderer=(self.render_mode == "human"),
            has_offscreen_renderer=False, use_camera_obs=False, use_object_obs=True,
            reward_shaping=False, horizon=h.horizon, control_freq=h.control_freq,
            controller_configs=load_composite_controller_config(controller="BASIC"),
            ignore_done=True,
        )
        # §4 la corsa reale del giunto: converte le frazioni in radianti
        jid = self._rs.sim.model.jnt_qposadr.tolist().index(self._rs.hinge_qpos_addr)
        lo, hi = self._rs.sim.model.jnt_range[jid]
        self.corsa_min, self.corsa_max = float(lo), float(hi)
        self._grasp = MultiApproachGrasp(self.cfg)
        self._rand = ExtendedDomainRandomizer(self.cfg, self._rs.sim.model)

    # ══ ciclo ═══════════════════════════════════════════════════════════
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        # NOTA: come nei progetti originali la randomizzazione di dominio pesca
        # dal generatore GLOBALE di numpy, che reset(seed=) non semina. Per
        # riprodurre un episodio va imposto np.random.seed() dall'esterno.
        self._rand.randomize_episode(self.cfg.curriculum_level)
        obs = self._rs.reset()

        if self._obs_keys is None:      # §6 le 112 chiavi vettoriali, fissate una volta
            self._obs_keys = sorted(k for k, v in obs.items()
                                    if isinstance(v, np.ndarray) and v.ndim == 1)

        # §4 la porta viene POSATA all'angolo iniziale del compito: la
        # chiusura parte aperta (70-100 % della corsa), l'apertura parte chiusa.
        theta_zero = self.cfg.task.campiona_posa_iniziale(self.corsa_min, self.corsa_max, np.random)
        self._rs.sim.data.qpos[self._rs.hinge_qpos_addr] = theta_zero
        self._rs.sim.forward()
        theta_star = self.cfg.task.campiona_bersaglio(self.corsa_min, self.corsa_max, np.random)
        self.door.reset(quota_leva=self.cfg.thr.quota_leva,
                        theta_zero=theta_zero, theta_star=theta_star,
                        tol=self.cfg.task.tol,
                        leva_rilascio=self.cfg.thr.leva_rilascio)

        self.fsm.reset()
        self.reward.reset()
        self._passi = 0
        self._retreat_pos = None
        self._eef_rilascio = None
        self._ritiro_spostato = 0.0
        self._prev_cmd = None
        self._contatto_prec = False
        return self._osserva(obs), {}

    def step(self, action: np.ndarray):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)

        # §6 (tesi §6.2.1) — FILTRO sull'azione, invariato rispetto agli
        # originali: media mobile esponenziale con alpha = 0.95. Smorza lo
        # strappo fra passi consecutivi prima che il comando arrivi al
        # simulatore. Senza, il braccio riceve comandi discontinui.
        eseguita = action if self._prev_cmd is None else (
            self.cfg.action_alpha * action + (1.0 - self.cfg.action_alpha) * self._prev_cmd)
        self._prev_cmd = eseguita.copy()

        # §8 — i controllori sostituiscono l'azione PRIMA che raggiunga il
        # simulatore. Non sono ricompensa: sono controllo classico.
        eseguita = self._applica_controllori(eseguita)
        obs, _, _, _ = self._rs.step(eseguita)
        self._passi += 1

        m = self._leggi(obs, eseguita)
        self.door.step(m["theta"], m["latch"])
        self._contatto_prec = bool(m["contatto"])

        hold_target = self.cfg.hold_steps(self._rand.current_latch_stiffness,
                                          self._rand.base_latch_stiffness or 1.0)
        self.fsm.step(door=self.door, presa_ok=m["presa_ok"], contatto=m["contatto"],
                      dist_mano=m["dist_mano"], latch=m["latch"],
                      hold_target=hold_target, grip_cmd=m["grip_cmd"],
                      soglia=m["soglia"], door_qvel=m["door_qvel"],
                      dist_ritiro=m["dist_ritiro"])

        # §7 — il cricchetto vive nella fase di moto: si posa il riferimento
        # entrando in MOVE e si incassa solo con la presa comandata sopra la
        # soglia adattiva, come nei due sorgenti.
        delta_p = 0.0
        if self.fsm.s.fase == Fase.MOVE:
            self.door.ancora()
            if m["grip_cmd"] > m["soglia"]:
                delta_p = self.door.incassa()

        terminated = self.fsm.s.fase == Fase.FINE
        truncated = self._passi >= self.cfg.sac.horizon

        termini = self.reward.compute(
            fsm=self.fsm, door=self.door, action=eseguita, delta_p=delta_p,
            hold_target=hold_target, terminato=terminated,
            raggio=self._rand.current_handle_radius,
            soglia_presa=self.fsm.soglia_presa(self._rand.current_handle_friction),
            **m["reward_args"])
        reward = self.reward.total(termini, terminato=terminated)

        info = {
            "reward_terms": termini,                 # pubblicati: niente da intercettare
            "fsm_phase": int(self.fsm.s.fase),
            "controllore": int(self.fsm.s.controllore),
            "mascherati": sorted(self.fsm.termini_mascherati()),
            "door_error": self.door.e, "A": bool(self.door.A),
            "latch_finale": float(m["latch"]),
            "theta_star": self.door.theta_star, "progress": self.door.p,
            # §9 — I TRE LIVELLI DI SUCCESSO della tesi (§6.3.4), non uno solo:
            #   permissivo  la politica ha raggiunto la fase di mantenimento
            #   true        a fine episodio la porta e' al bersaglio E la leva
            #               e' tornata a riposo
            #   clean       in piu', il ritiro e' stato effettivamente completato
            # `is_success` e' il TRUE SUCCESS, perche' e' quello con cui la
            # baseline si misura (apertura 83.0 %, chiusura 100 %). Fermarsi a
            # `terminato ∧ A` sarebbe piu' permissivo — non guarderebbe la leva —
            # e i numeri non sarebbero confrontabili con quelli della tesi.
            "successo_permissivo": bool(self.fsm.s.fase >= Fase.HOLD),
            "is_success": bool(terminated and self.door.A
                               and abs(m["latch"]) <= self.cfg.thr.latch_term_tol),
            "ritiro_spostato": float(self._ritiro_spostato),
            # §7 — definizione del progetto originale (`_common.py` riga 382):
            #   true success ∧ spostamento massimo >= 6 cm ∧ uscita PULITA
            "successo_pulito": bool(terminated and self.door.A
                                    and abs(m["latch"]) <= self.cfg.thr.latch_term_tol
                                    and self._ritiro_spostato >= self.cfg.thr.ritiro_spostamento_min
                                    and self.fsm.s.uscita_pulita),
        }
        return self._osserva(obs), reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            self._rs.render()

    def close(self):
        if self._rs is not None:
            self._rs.close()

    # ══ lettura dello stato ═════════════════════════════════════════════
    def _angolo_porta(self) -> float:
        return float(np.clip(self._rs.sim.data.qpos[self._rs.hinge_qpos_addr],
                             self.corsa_min, self.corsa_max))

    def _leggi(self, obs: Dict[str, Any], action: np.ndarray) -> dict:
        """Tutte le grandezze che servono al reward, lette una volta sola."""
        rs = self._rs
        theta = self._angolo_porta()
        latch = float(rs.sim.data.qpos[rs.handle_qpos_addr])
        qvel = float(rs.sim.data.qvel[rs.hinge_qpos_addr])

        self._eef = np.asarray(obs.get("robot0_eef_pos", np.zeros(3)), dtype=np.float64)
        self._handle = np.asarray(obs.get("handle_pos", self._eef), dtype=np.float64)
        d3 = float(np.linalg.norm(self._handle - self._eef))
        dxy = float(np.linalg.norm(self._handle[:2] - self._eef[:2]))
        dz = float(self._eef[2] - self._handle[2])

        grip_cmd = float(action[-1])
        soglia = self.fsm.soglia_presa(self._rand.current_handle_friction)
        raggio = self._rand.current_handle_radius
        vicino = d3 <= self.fsm.soglia_distanza(raggio)
        # Contatto FISICO: SOLO la larghezza reale delle dita confrontata col
        # diametro della maniglia, esattamente come `is_physically_closed` nei
        # due sorgenti. NON deve dipendere dalla vicinanza: `vicino` entra
        # separatamente in `presa_ok`, dove serve a CONFERMARE la presa.
        # Mettendolo dentro `contatto` si rompono tre cose insieme — la morsa
        # sul gripper si sgancia appena la mano si scosta di 3.5 cm, il comando
        # grezzo (basso) torna a passare, e la presa risulta persa: misurato
        # sulla traiettoria che risolve l'apertura, la mano supera i 3.5 cm
        # nell'11 % dei passi di MOVE mentre gira la leva, e il comando grezzo
        # sta sotto soglia−0.20 nel 94 %.
        larghezza = float(np.sum(np.abs(np.asarray(obs["robot0_gripper_qpos"]))))
        self._larghezza = larghezza
        self._latch = latch
        contatto = bool(0.015 <= larghezza <= 2.0 * raggio + 0.025)
        # Condizione di presa: le TRE condizioni degli originali (fsm_v2.py,
        # ramo PHASE_REACH). Senza `contatto` — la chiusura FISICA delle dita —
        # l'agente puo' comandare il gripper chiuso a vuoto, farsi confermare la
        # presa, incassare il bonus di transizione e riperderla subito: un ciclo
        # REACH->MOVE->REACH che paga senza mai muovere la porta.
        presa_ok = vicino and grip_cmd >= soglia and contatto

        # allineamento del polso: prima componente delle feature multi-approccio
        gf = self._feature_presa(obs)
        allin = float(gf[0])
        piatto = self._polso_piatto(obs)

        vel_giunti = float(np.linalg.norm(obs.get("robot0_joint_vel", np.zeros(7))))

        # §8 il punto di ritiro, calcolato una volta all'ingresso in RELEASE
        # §7 — quanto la mano si e' allontanata dalla posa in cui ha lasciato la
        # maniglia. E' la grandezza con cui il progetto originale misura il
        # ritiro (`retreat_moved`, open_generalized_v2/env_v2.py). Massimo
        # sull'episodio, come fa l'originale.
        if self.fsm.s.fase == Fase.RELEASE:
            if self._eef_rilascio is None:
                self._eef_rilascio = self._eef.copy()
            self._ritiro_spostato = max(
                self._ritiro_spostato,
                float(np.linalg.norm(self._eef - self._eef_rilascio)))

        if self.fsm.s.fase == Fase.RELEASE and self._retreat_pos is None:
            self._retreat_pos = self._punto_ritiro(self._eef, *self._normale_porta())
        if self._retreat_pos is not None:
            verso = self._retreat_pos - self._eef
            dist_ret = float(np.linalg.norm(verso))
        else:
            verso, dist_ret = None, 1.0

        return dict(
            theta=theta, latch=latch, presa_ok=presa_ok, contatto=contatto,
            dist_mano=d3, dist_ritiro=dist_ret,
            grip_cmd=grip_cmd, soglia=soglia, door_qvel=qvel,
            reward_args=dict(
                dist_mano=d3, dist_xy=dxy, dz=dz, allineamento=allin, polso_piatto=piatto,
                grip_cmd=grip_cmd, contatto=contatto,
                door_qvel=qvel, latch=latch, vel_giunti=vel_giunti,
                dist_ritiro=dist_ret, dir_ritiro=verso,
            ),
        )

    def _normale_porta(self):
        """Normale del pannello e posizione del corpo porta.

        La normale è la prima colonna della matrice di rotazione del corpo
        `Door_main`, come nei due sorgenti (`compute_retreat_pos`). Il nome del
        corpo conta: con `door` al posto di `Door_main` si legge un quaternione
        diverso e la via di fuga risulta quasi ORTOGONALE a quella vera —
        misurato, cos = 0.10 fra le due direzioni.
        """
        try:
            bid = self._rs.sim.model.body_name2id("Door_main")
            w, x, y, z = np.asarray(self._rs.sim.model.body_quat[bid], dtype=np.float64)
            n = np.array([1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)])
            return n, np.asarray(self._rs.sim.data.body_xpos[bid], dtype=np.float64)
        except Exception:
            return np.array([1.0, 0.0, 0.0]), np.zeros(3)

    # ══ §6 osservazione ═══════════════════════════════════════════════
    def _osserva(self, obs: Dict[str, Any]) -> np.ndarray:
        parti = [
            self._chiavi_vettoriali(obs),                                   # 112
            self._feature_presa(obs),                                       # 4
            np.asarray(self._rand.obs_features(), dtype=np.float32),        # 3
            self.fsm.s.one_hot,                                             # 4
            # informazione reale che oggi manca all'apertura: si adotta la
            # scelta della chiusura, che la comunica alla politica
            np.array([self._rand.current_handle_radius,
                      self._rand.current_handle_friction], dtype=np.float32),   # 2
            # parametro di compito: nella chiusura vale 0 per definizione
            np.array([self._bersaglio_normalizzato()], dtype=np.float32),   # 1
        ]
        v = np.concatenate(parti).astype(np.float32)
        if v.shape[0] != OBS_DIM:                    # allinea a 126 in modo esplicito
            v = (v[:OBS_DIM] if v.shape[0] > OBS_DIM
                 else np.concatenate([v, np.zeros(OBS_DIM - v.shape[0], np.float32)]))
        return v

    def _chiavi_vettoriali(self, obs: Dict[str, Any]) -> np.ndarray:
        return np.concatenate([np.asarray(obs[k], dtype=np.float32).ravel()
                               for k in self._obs_keys]) if self._obs_keys else np.zeros(112, np.float32)

    def _feature_presa(self, obs: Dict[str, Any]) -> np.ndarray:
        """[ten Pas et al. 2017] — modulo invariato (§6)."""
        try:
            bid = self._rs.sim.model.body_name2id("door")
            dq = self._rs.sim.model.body_quat[bid]
        except Exception:
            dq = None
        f = self._grasp.obs_features(obs.get("robot0_eef_quat"), self._handle, self._eef, dq)
        return np.asarray(f, dtype=np.float32)

    def _bersaglio_normalizzato(self) -> float:
        """θ* normalizzato sulla corsa della porta.

        §6 — esiste solo nell'apertura perché nella chiusura vale 0 per
        definizione; qui è presente in entrambi, costante nel primo caso.
        """
        amp = self.corsa_max - self.corsa_min
        return float(np.clip((self.door.theta_star - self.corsa_min) / amp, 0.0, 1.0)) if amp > 1e-9 else 0.0

    # ══ §8 geometria del punto di ritiro ══════════════════════════════
    def _punto_ritiro(self, pos_eef: np.ndarray, normale: np.ndarray,
                      pos_porta: np.ndarray) -> np.ndarray:
        """Tre parametri geometrici, una sola causa fisica (§8).

        L'ancora e' la posa della MANO all'ingresso in RELEASE, non la maniglia:
        `p_eef + d_ret·n + z_ret·ẑ`, come nei due sorgenti (tesi §6.2.4,
        tab. 6.5). Ancorandolo alla maniglia la direzione di fuga risulta
        diversa e il termine `retreat` punisce la traiettoria giusta: misurato
        sulla soluzione della chiusura, RELEASE −3.99 per passo invece di +2.30.

        A porta chiusa il pannello è contro il telaio e la leva quasi a riposo;
        a porta aperta la leva è a fondo corsa con la molla carica. Da qui:
          · d_ret quasi doppio: serve più corsa per uscire dall'arco della leva
          · z_ret azzerato: salire o scendere farebbe rientrare il dito nell'arco
          · normale riorientata: a porta aperta è ruotata insieme al pannello
        """
        t = self.cfg.task
        n = np.asarray(normale, dtype=np.float64)
        n = n / max(float(np.linalg.norm(n)), 1e-9)
        # §8 — la normale va ORIENTATA verso il robot, non assunta tale: il
        # sorgente dell'apertura lo ha misurato («punta a −Y, cioe' VIA dal
        # robot: il ritiro ordinava al braccio di attraversare il pannello»).
        # Nella chiusura il difetto e' invisibile perche' la porta e' contro il
        # telaio; nell'apertura la porta e' libera.
        if t.orienta_normale and float(np.dot(n, pos_eef - pos_porta)) < 0.0:
            n = -n
        return pos_eef + t.d_ret * n + np.array([0.0, 0.0, t.z_ret])

    # ══ §8 i controllori: override sull'azione ════════════════════════
    def _applica_controllori(self, action: np.ndarray) -> np.ndarray:
        a = action.copy()

        # ── override di fase, presenti in ENTRAMBI gli originali ────────────
        # Non sono ricompensa: sono controllo classico, e senza di loro due
        # fasi non chiudono.
        #
        # BLOCCO DEL BRACCIO IN HOLD (close env_v2 riga 175, open riga 451).
        # In HOLD il compito e' tenere ferma la porta al bersaglio: se il
        # braccio resta comandato dalla policy la porta oscilla, `still` non
        # paga e HOLD diventa una fase in cui non conviene entrare.
        if self.fsm.s.fase == Fase.HOLD:
            a[:-1] = 0.0

        # GRIP-LOCK in MOVE e HOLD (close riga 180, open riga 318): se la presa
        # era fisicamente chiusa, il comando del gripper non puo' scendere sotto
        # la soglia adattiva piu' un margine. Impedisce di perdere la presa per
        # rumore di esplorazione proprio mentre si muove la porta.
        if self.fsm.s.fase in (Fase.MOVE, Fase.HOLD) and self._contatto_prec:
            soglia = self.fsm.soglia_presa(self._rand.current_handle_friction)
            a[-1] = max(float(a[-1]), min(1.0, soglia + self.cfg.thr.grip_lock_margin))

        c = self.fsm.s.controllore
        if c != Controllore.C0_RIPORTO_LEVA:
            self._passi_riporto = 0          # la rampa riparte a ogni riporto
        if c == Controllore.NESSUNO:
            return np.clip(a, -1.0, 1.0)
        if c == Controllore.C0_RIPORTO_LEVA:
            # il braccio accompagna la leva lungo il suo arco, presa ancora chiusa
            # §8 C0 — RIPORTO DELLA LEVA, nella forma del sorgente
            # (open env_v2, rami §1.46-§1.51). Tre pezzi, tutti misurati li':
            #   · ampiezza proporzionale a |leva|, con tetto 0.6 e RAMPA sui
            #     primi passi: senza, l'ingresso in RELEASE da' uno strappo
            #     alla porta, perche' la presa e' ancora chiusa e accoppiata
            #     al cardine;
            #   · ROTAZIONE DEL POLSO coerente con il moto rigido attorno
            #     all'asse della leva: azzerarla impone all'OSC di tenere
            #     l'orientazione rigida mentre la traslazione segue l'arco, un
            #     vincolo contraddittorio che pianta il braccio;
            #   · GABBIA invece della morsa: bang-bang sulla larghezza REALE
            #     delle dita attorno a diametro + margine. A presa chiusa la
            #     barra deve RUOTARE dentro la pinza, e su una maniglia grossa
            #     la coppia d'attrito della morsa vince la spinta: il riporto
            #     rallenta di 30 volte e si pianta, con la porta che rimbalza
            #     di ±0.13 rad. Nella gabbia la barra ruota libera e il dito
            #     l'accompagna.
            verso, omega = self._direzione_leva()
            self._passi_riporto += 1
            rampa = min(1.0, self._passi_riporto / self.cfg.thr.riporto_rampa)
            mag = min(self.cfg.thr.riporto_mag_max,
                      self.cfg.thr.riporto_guadagno * abs(self._latch)) * rampa
            a[:3] = np.clip(verso * mag, -1.0, 1.0)
            a[3:6] = np.clip(omega * (self.cfg.thr.riporto_rot_gain * mag), -1.0, 1.0)
            largo_tgt = 2.0 * self._rand.current_handle_radius + self.cfg.thr.gabbia_margine
            a[-1] = 1.0 if self._larghezza > largo_tgt else -1.0
        elif c == Controllore.C1_ESCAPE:
            # l'ambiente comanda il braccio verso il punto di ritiro
            if self._retreat_pos is not None:
                v = self._retreat_pos - self._eef
                n = float(np.linalg.norm(v))
                if n > 1e-9:
                    a[:3] = np.clip(v / n, -1.0, 1.0)
        return np.clip(a, -1.0, 1.0)

    def _polso_piatto(self, obs) -> float:
        """Quanto il polso è «piatto»: |x_eef[2]| dalla matrice di rotazione.

        Stesso calcolo di `flat_alignment` in close_generalized_v2/env_v2.py.
        """
        q = obs.get("robot0_eef_quat")
        if q is None:
            return 0.0
        x, y, z, w = np.asarray(q, dtype=np.float64)     # robosuite: xyzw
        return float(abs(2.0 * (x * z - w * y)))          # terza riga, prima colonna

    def _direzione_leva(self):
        """§8 C0 — tangente all'arco della leva nel punto afferrato.

        Non è una stima geometrica: asse e ancora del giunto sono letti dal
        simulatore. In MuJoCo il qpos di un hinge cresce con la regola della
        mano destra attorno all'asse, quindi la velocità del punto afferrato
        per `latch` DECRESCENTE è v = −(asse × r), con r = eef − ancora.
        Stesso metodo di `_latch_restore_dir` in open_generalized_v2/env_v2.py.
        """
        try:
            jid = getattr(self._rand, "latch_joint_id", None)
            if jid is None:
                return np.zeros(3), np.zeros(3)
            sim = self._rs.sim
            asse = np.asarray(sim.data.xaxis[jid], dtype=np.float64)
            ancora = np.asarray(sim.data.xanchor[jid], dtype=np.float64)
            v = -np.cross(asse, self._eef - ancora)
            segno = -1.0
            if float(sim.data.qpos[self._rs.handle_qpos_addr]) < 0.0:
                v, segno = -v, +1.0             # leva oltre lo zero: verso opposto
            n = float(np.linalg.norm(v))
            if n < 1e-8:
                return np.zeros(3), np.zeros(3)
            # il riporto e' un moto RIGIDO attorno all'asse: serve la tangente
            # per la traslazione e l'asse firmato per la rotazione del polso
            return v / n, segno * asse
        except Exception:
            return np.zeros(3), np.zeros(3)
