#!/usr/bin/env python3
# open_generalized_v2/train_curriculum_v2.py
#
# Entry-point di training per l'APERTURA generalizzata v2 — SOLO curriculum 1
# (posa variabile, soglie adattive, fisica randomizzata). Speculare a
# close_generalized_v2/train_gen_v2.py, con la stessa logica di eval/best-model/
# VecNormalize e lo stesso schema di --play (env raw + obs_rms manuale).
#
# Uso:
#   # training (curriculum 1)
#   python -m open_generalized_v2.train_curriculum_v2 --total-steps 1500000
#
#   # play (visualizza la policy migliore)
#   python -m open_generalized_v2.train_curriculum_v2 --play
#
# Riferimenti: SAC (sb3); potential-based shaping [3]; domain randomization [8][17].

from __future__ import annotations

import os
import sys
import time
import argparse
import numpy as np

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import robusti (package-qualified con fallback piatto): funziona con
# `mjpython open_generalized_v2/train_curriculum_v2.py` e con `python -m ...`.
try:
    from open_generalized_v2.config_v2 import TrainConfigV2Open
    from open_generalized_v2.env_v2 import AdvancedGeneralizedOpenDoorEnv
except ModuleNotFoundError:
    from config_v2 import TrainConfigV2Open
    from env_v2 import AdvancedGeneralizedOpenDoorEnv


def make_env_fn(cfg, render_mode=None):
    def _thunk():
        env = AdvancedGeneralizedOpenDoorEnv(cfg, render_mode=render_mode)
        env.set_curriculum_level(cfg.fixed_curriculum_level)  # curriculum 1 fisso
        return env
    return _thunk


# ─────────────────────────────────────────────────────────────────────────────
# Eval callback: salva best_model + vecnormalize.pkl quando migliora il success.
# ─────────────────────────────────────────────────────────────────────────────
def build_eval_callback():
    from stable_baselines3.common.callbacks import BaseCallback

    class EvalBestCallback(BaseCallback):
        def __init__(self, eval_env, save_path, eval_freq, n_eval_episodes, verbose=1):
            super().__init__(verbose)
            self.eval_env = eval_env
            self.save_path = save_path
            self.eval_freq = eval_freq
            self.n_eval_episodes = n_eval_episodes
            self.best_success = -1.0
            self._next_eval = eval_freq
            os.makedirs(save_path, exist_ok=True)

        def _evaluate(self):
            succ, lengths = [], []
            for _ in range(self.n_eval_episodes):
                obs = self.eval_env.reset()
                done = np.array([False])
                steps = 0
                last_info = {}
                while not done[0]:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, _r, done, infos = self.eval_env.step(action)
                    last_info = infos[0]
                    steps += 1
                succ.append(int(bool(last_info.get("is_success", False))))
                lengths.append(steps)
            return float(np.mean(succ)), float(np.mean(lengths))

        def _on_step(self) -> bool:
            # trigger basato su num_timesteps (robusto a num_envs): n_calls*num_envs.
            # Usiamo una soglia progressiva così l'eval scatta a multipli REALI di eval_freq.
            if self.num_timesteps >= self._next_eval:
                self._next_eval += self.eval_freq
                if self.model.get_vec_normalize_env() is not None:
                    self.eval_env.obs_rms = self.model.get_vec_normalize_env().obs_rms
                sr, ml = self._evaluate()
                print(f"\n--- [EVAL OPEN v2] step {self.num_timesteps} ---")
                print(f"Success: {sr*100:.1f}%  (best {max(sr,self.best_success)*100:.1f}%)  ep_len {ml:.1f}\n")
                # checkpoint SEMPRE aggiornato (così interrompere a metà lascia un modello caricabile dal diagnostico)
                self.model.save(os.path.join(self.save_path, "latest_model.zip"))
                if self.model.get_vec_normalize_env() is not None:
                    self.model.get_vec_normalize_env().save(
                        os.path.join(self.save_path, "vecnormalize.pkl"))
                if sr > self.best_success:
                    self.best_success = sr
                    self.model.save(os.path.join(self.save_path, "best_model.zip"))
                    if self.model.get_vec_normalize_env() is not None:
                        self.model.get_vec_normalize_env().save(
                            os.path.join(self.save_path, "vecnormalize.pkl"))
                    print(f"[BEST OPEN v2] nuovo best: {sr*100:.1f}%")
            return True

    return EvalBestCallback


def train(cfg, total_steps):
    from stable_baselines3 import SAC
    from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize

    os.makedirs(cfg.run_dir, exist_ok=True)

    venv = DummyVecEnv([make_env_fn(cfg) for _ in range(cfg.num_envs)])
    venv = VecMonitor(venv)
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)

    eval_env = DummyVecEnv([make_env_fn(cfg)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    eval_env.training = False

    EvalBestCallback = build_eval_callback()
    cb = EvalBestCallback(eval_env, cfg.run_dir, cfg.eval_freq, cfg.n_eval_episodes)

    model = SAC(
        "MlpPolicy", venv,
        learning_rate=cfg.learning_rate, buffer_size=cfg.buffer_size,
        batch_size=cfg.batch_size, gamma=cfg.gamma, tau=cfg.tau,
        train_freq=cfg.train_freq, gradient_steps=cfg.gradient_steps,
        learning_starts=cfg.learning_starts, ent_coef=cfg.ent_coef,
        target_entropy=cfg.target_entropy,
        policy_kwargs=dict(net_arch=list(cfg.policy_net_arch)),
        tensorboard_log=cfg.tb_dir, seed=cfg.seed, verbose=1,
    )
    model.learn(total_timesteps=int(total_steps), callback=cb)
    model.save(os.path.join(cfg.run_dir, "final_model.zip"))
    venv.save(os.path.join(cfg.run_dir, "vecnormalize.pkl"))
    print("[OPEN v2] Training complete.")


# ─────────────────────────────────────────────────────────────────────────────
# PLAY — visualizzazione + HUD tabellare completo.
#   • Render DEFAULT: renderer robosuite (render_mode="human"); sotto mjpython la
#     finestra resta aperta e la camera è già controllabile col mouse.
#   • Render --free-viewer (opt-in): viewer nativo mujoco.viewer.launch_passive con
#     i PANNELLI impostazioni (UI) visibili; un solo viewer, nessun conflitto.
#   • HUD: a ogni step stampa TUTTO lo stato FSM + il breakdown del reward diviso in
#     PREMI/PENALITÀ (colonne allineate, colori auto su TTY), con barra di apertura,
#     tag di fase colorati e, a fine episodio, il contributo CUMULATO per termine.
#     I colori si disattivano da soli se rediriggi l'output su file.
# ─────────────────────────────────────────────────────────────────────────────

# etichette leggibili dei termini di reward (chiave interna → descrizione breve)
_REW_LABEL = {
    "base": "time-penalty", "phi_shape": "shaping Φ (Ng)",
    "dist_3d": "dist. maniglia 3D", "dist_xy": "dist. maniglia XY",
    "dist_z": "disliv. Z", "app_blw": "sotto la maniglia", "app_top": "sopra la maniglia",
    "grip": "comando presa", "grip_contact": "contatto in PULL",
    "door_prog": "PROGRESSO apertura", "hold": "porta al goal",
    "hold_slip": "presa persa", "hold_grip": "presa in HOLD",
    "hold_drop_pen": "gripper aperto in HOLD", "hold_act": "braccio fermo",
    "hold_dist": "maniglia lontana", "ret_grip": "rilascio gripper",
    "ret_rot": "torsione polso", "ret_dir": "direzione ritiro",
    "ret_perp": "deriva laterale", "ret_freeze": "settle braccio",
    "ret_release": "rilascio a porta aperta", "latch_ret": "monitor leva",
    "door_regress": "richiusura porta", "success_bonus": "BONUS successo",
}

# fase → colore ANSI (solo se il terminale è un TTY; se rediriggi su file, niente codici)
_USE_COLOR = None  # lazy: valorizzato al primo uso in base a sys.stdout.isatty()
_ANSI = {"reset": "\033[0m", "dim": "\033[2m", "bold": "\033[1m",
         "green": "\033[32m", "red": "\033[31m", "yellow": "\033[33m",
         "cyan": "\033[36m", "mag": "\033[35m", "blue": "\033[34m"}
_PHASE_COLOR = {"REACH": "cyan", "PULL": "yellow", "HOLD_OPEN": "green", "RETREAT": "mag"}


def _c(txt, color):
    """Colora `txt` se siamo su un TTY, altrimenti lo lascia pulito (per redirezione su file)."""
    global _USE_COLOR
    if _USE_COLOR is None:
        try: _USE_COLOR = bool(sys.stdout.isatty())
        except Exception: _USE_COLOR = False
    if not _USE_COLOR or color not in _ANSI:
        return txt
    return f"{_ANSI[color]}{txt}{_ANSI['reset']}"


def _bar(frac, width=14, fill="█", empty="·"):
    """Barretta ASCII di riempimento in [0,1] (per progressi visivi nell'HUD)."""
    frac = 0.0 if frac != frac else max(0.0, min(1.0, float(frac)))
    n = int(round(frac * width))
    return fill * n + empty * (width - n)


def _fnum(x, w=7, p=3):
    try:    return f"{float(x):+{w}.{p}f}"
    except Exception: return " " * (w - 2) + "n/d"


def _hud_thresholds(env):
    """Soglie adattive correnti e parametri fisici dell'episodio (per l'intestazione)."""
    dr  = env._domain_rand
    fsm = env._fsm
    hr  = float(getattr(dr, "current_handle_radius", float("nan")))
    hf  = float(getattr(dr, "current_handle_friction", float("nan")))
    return {
        "handle_radius": hr, "handle_friction": hf,
        "g_thresh": float(fsm.grip_thresh(hf)),
        "d_thresh": float(fsm.grasp_dist_thresh(hr)),
        "latch_stiff": float(getattr(dr, "current_latch_stiffness", float("nan"))),
        "hinge_damp":  float(getattr(dr, "current_hinge_damping", float("nan"))),
        "door_mass":   float(getattr(dr, "current_door_mass", float("nan"))),
    }


_W = 96  # larghezza fissa dei riquadri HUD


def _hud_episode_header(env, ep_idx, info):
    th = _hud_thresholds(env)
    goal = info.get('goal_angle', float('nan'))
    top = "╔" + "═" * (_W - 2) + "╗"
    bot = "╚" + "═" * (_W - 2) + "╝"
    def _row(s):
        s = s[: _W - 4]
        print("║ " + s + " " * (_W - 4 - len(s)) + " ║")
    print("\n" + _c(top, "bold"))
    _row(_c(f"EPISODIO {ep_idx}", "bold") + f"   —   goal_angle = {goal:.3f} rad   "
         f"(door∈[{env._door_min:.3f}, {env._effective_max:.3f}], tol {env.cfg.open_tol_rad:.3f})")
    _row(f"maniglia : raggio={th['handle_radius']:.4f}  attrito={th['handle_friction']:.3f}"
         f"   →  soglie adattive  g_thresh={th['g_thresh']:.3f}  d_thresh={th['d_thresh']:.4f}")
    _row(f"fisica   : latch_stiff={th['latch_stiff']:.3f}  hinge_damp={th['hinge_damp']:.3f}  "
         f"door_mass={th['door_mass']:.3f}")
    print(_c(bot, "bold"))


def _hud_step(env, step, action, reward, info, cum_reward, hold_frac=None):
    """Stampa lo stato FSM completo + il breakdown reward (premi/penalità), allineati e colorati."""
    st   = env._fsm.state
    ph   = info.get("fsm_phase_name", st.phase_name)
    ph_c = _c(f"{ph:<9s}", _PHASE_COLOR.get(ph, "blue"))
    door   = info.get("door_angle", float("nan"))
    dqv    = info.get("door_qvel", float("nan"))
    oerr   = info.get("open_error", float("nan"))
    latch  = info.get("latch_qpos", float("nan"))
    disth  = info.get("dist_handle", float("nan"))
    gw     = info.get("gripper_width", float("nan"))
    grip_a = float(action[-1]) if action is not None and len(action) else float("nan")
    arm    = float(np.linalg.norm(np.asarray(action[:-1], float))) if action is not None else float("nan")
    goal   = info.get("goal_angle", float("nan"))
    # apertura vs goal come barra visiva
    aperture_frac = (door - env._door_min) / max(1e-6, goal - env._door_min)
    ok_open = (oerr == oerr and oerr < env.cfg.open_tol_rad)
    open_tag = _c(" AL GOAL ", "green") if ok_open else _c("  ...    ", "yellow")

    print("┌" + "─" * (_W - 1))
    print(f"│ step {_c(f'{step:4d}','bold')}  fase {ph_c}  [{_bar(aperture_frac)}] apertura {open_tag}")
    print(f"│ STATO   door={door:6.3f}/{goal:.3f}  dθ/dt={_fnum(dqv,7,3)}  open_err={oerr:6.3f}  "
          f"latch={_fnum(latch,7,3)}")
    print(f"│         dist_maniglia={disth:6.4f}  grip_width={gw:6.4f}  cmd_grip={_fnum(grip_a,6,2)}  "
          f"|azione_braccio|={arm:5.3f}")
    # ── FSM interna: TUTTI i contatori ──
    hold_s = f"{st.hold_open_duration}/{st.target_hold_steps or 0}"
    print(f"│ FSM     reach={st.reach_steps:<3d} pull={st.pull_steps:<3d} hold={hold_s:<7s} "
          f"retreat={st.retreat_steps:<3d} free={st.retreat_free_steps:<3d} "
          f"grasp_confirm={st.grasp_confirm_count}/5 return_hold={st.return_hold}")
    if int(info.get("fsm_phase", -1)) == 3:
        restoring = bool(info.get("retreat_restoring", False))
        rs_tag = _c("RIPORTO-LEVA", "yellow") if restoring else _c("SFILAMENTO", "cyan")
        print(f"│ RITIRO  {rs_tag}  riporto(done={getattr(env,'_retreat_restore_done',None)} "
              f"cage={getattr(env,'_retreat_restore_cage',None)} "
              f"step={getattr(env,'_retreat_restore_steps',0)})  "
              f"arretrato={info.get('retreat_moved',0.0):.4f} m")
    ev = info.get("fsm_events") or []
    if ev:
        print("│ " + _c("EVENTI  " + " ; ".join(ev), "bold"))
    # ── REWARD: premi (verde) vs penalità (rosso), ordinati per |valore| ──
    terms = dict(info.get("reward_terms", {}) or {})
    premi = {k: v for k, v in terms.items() if v > 1e-9}
    penal = {k: v for k, v in terms.items() if v < -1e-9}
    def _cells(d, color):
        cells = []
        for k, v in sorted(d.items(), key=lambda kv: -abs(kv[1])):
            cells.append(_c(f"{k}={v:+.2f}", color) + _c(f" {_REW_LABEL.get(k,k)}", "dim"))
        return cells
    print("│ " + "─" * (_W - 3))
    pc = _cells(premi, "green")
    nc = _cells(penal, "red")
    if pc:
        print("│ " + _c("PREMI   ", "green") + "  ".join(pc[:2]))
        for i in range(2, len(pc), 2): print("│         " + "  ".join(pc[i:i+2]))
    if nc:
        print("│ " + _c("PENALITÀ", "red") + "  ".join(nc[:2]))
        for i in range(2, len(nc), 2): print("│         " + "  ".join(nc[i:i+2]))
    tot_c = "green" if reward >= 0 else "red"
    print("└ " + _c(f"Σ step = {reward:+.3f}", tot_c) + f"     Σ episodio = {cum_reward:+.2f}\n")


def _hud_summary(ep_idx, steps, phase_time, reward_cum, info, term, trunc,
                 rew_accum=None, open_tol=0.05):
    latch = info.get("latch_qpos", float("nan"))
    oerr  = info.get("open_error", float("nan"))
    door  = info.get("door_angle", float("nan"))
    if term:
        clean = abs(latch) < 0.15
        esito = (_c("PULITA (leva a casa, ritiro completo)", "green") if clean
                 else _c("ESOGENA (uscita a tempo, leva non tornata)", "yellow"))
    else:
        esito = _c("troncata a orizzonte", "red")
    top = "╔" + "═" * (_W - 2) + "╗"
    bot = "╚" + "═" * (_W - 2) + "╝"
    print(_c(top, "bold"))
    print("║ " + _c(f"RIEPILOGO EPISODIO {ep_idx}", "bold") + f"   esito: {esito}")
    print(f"║   step totali = {steps}      per fase = {phase_time}")
    tag = _c("entro tol", "green") if (oerr == oerr and oerr < open_tol) else _c("FUORI tol", "red")
    print(f"║   door finale = {door:.3f}   open_error = {oerr:.4f} (tol {open_tol:.2f}, {tag})   "
          f"latch finale = {latch:+.4f}")
    print("║   " + _c(f"Σ reward episodio = {reward_cum:+.2f}", "bold"))
    # ── contributo CUMULATO per termine (dove è finito il reward dell'episodio) ──
    if rew_accum:
        items = sorted(rew_accum.items(), key=lambda kv: -abs(kv[1]))
        print("║   contributo cumulato per termine (top):")
        for k, v in items[:8]:
            col = "green" if v >= 0 else "red"
            lbl = _REW_LABEL.get(k, k)
            print("║     " + _c(f"{k:<14s} {v:+8.2f}", col) + _c(f"  {lbl}", "dim"))
    print(_c(bot, "bold") + "\n")


def play(cfg, model_path=None, hud_every=1, episodes=None, free_viewer=False,
         cam_dist=2.6, cam_az=135.0, cam_el=-20.0, cam_z=1.0,
         retreat_slow=2.0, end_pause=2.5):
    import pickle
    from stable_baselines3 import SAC

    # ── Scelta del renderer ──────────────────────────────────────────────────────
    # DEFAULT ("come sempre"): renderer NATIVO di robosuite (render_mode="human" +
    #   env.render()). Sotto mjpython robosuite apre la finestra e la tiene aperta;
    #   il mouse ruota/trasla/zooma già la camera.
    # --free-viewer (OPT-IN): apriamo NOI un unico viewer nativo mujoco.viewer.launch_
    #   passive con i PANNELLI impostazioni (UI sinistra/destra) visibili, e l'env NON
    #   usa il renderer robosuite (render_mode=None) → un solo viewer, nessun conflitto,
    #   nessun flicker. Dà accesso a TUTTI i settaggi del viewer (contatti, giunti,
    #   trasparenze, camere, ecc.). Percorso separato: NON tocca il default che funziona.
    #
    # macOS: qualunque finestra passa da launch_passive → serve mjpython. Lo rileviamo
    # (mujoco.viewer._MJPYTHON) per NON far crashare env.step() con python normale.
    is_mac = (sys.platform == "darwin")
    try:
        import mujoco.viewer as _mjv
        _under_mjpython = getattr(_mjv, "_MJPYTHON", None) is not None
    except Exception:
        _mjv = None
        _under_mjpython = False
    on_screen_ok = (not is_mac) or _under_mjpython

    use_free = bool(free_viewer) and on_screen_ok
    # con --free-viewer l'env NON deve avere il renderer robosuite (evita doppio viewer)
    rmode = None if (use_free or not on_screen_ok) else "human"
    env = AdvancedGeneralizedOpenDoorEnv(cfg, render_mode=rmode)
    env.set_curriculum_level(cfg.fixed_curriculum_level)

    obs_rms = None
    vn = os.path.join(cfg.run_dir, "vecnormalize.pkl")
    if os.path.exists(vn):
        with open(vn, "rb") as f:
            obs_rms = pickle.load(f).obs_rms

    model = SAC.load(model_path or os.path.join(cfg.run_dir, "best_model.zip"))

    def norm(o):
        if obs_rms is None:
            return o
        return np.clip((o - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8), -10.0, 10.0)

    PHASE_NAMES = ["REACH", "PULL", "HOLD_OPEN", "RETREAT"]
    if not on_screen_ok:
        print("\n[PLAY] ─ macOS: il rendering on-screen richiede mjpython ─────────────")
        print("  Rilancia così per avere la SCENA (finestra persistente + camera col mouse):")
        print("      mjpython -m open_generalized_v2.train_curriculum_v2 --play")
        print("  (mjpython è già nel venv: .venv/bin/mjpython — col venv attivo basta")
        print("   `mjpython`). Per ora proseguo con il SOLO HUD a terminale.\n")
    elif use_free:
        print("[PLAY] --free-viewer: viewer nativo con PANNELLI impostazioni. Controlli:")
        print("       mouse trascina = ruota • Shift+trascina = trasla • rotella = zoom")
        print("       Tab = pannelli • '[' ']' = cambia camera • H = help completo\n")

    free = None   # handle del viewer --free-viewer, aperto UNA volta dopo il primo reset
    ep_idx = 0
    try:
        while episodes is None or ep_idx < episodes:
            ep_idx += 1
            obs, info = env.reset()
            _hud_episode_header(env, ep_idx, info)

            # --free-viewer: apri il viewer UNA sola volta (dopo il primo reset, così i
            # dati sono pronti) e NON chiuderlo più (chiuderlo/riaprirlo era la causa del
            # vecchio flicker). show_left_ui / show_right_ui = pannelli impostazioni.
            if use_free and free is None:
                try:
                    m = getattr(env._rs_env.sim.model, "_model", env._rs_env.sim.model)
                    d = getattr(env._rs_env.sim.data,  "_data",  env._rs_env.sim.data)
                    free = _mjv.launch_passive(m, d, show_left_ui=True, show_right_ui=True)
                    # INQUADRATURA INIZIALE: il free-camera di MuJoCo parte troppo vicino
                    # (scena illeggibile). La impostiamo su una vista d'insieme del robot+porta.
                    # lookat ~ altezza maniglia (z≈1.0). Regolabile da CLI (--cam-*) o col mouse.
                    try:
                        hp = info.get("handle_pos", None)
                        lx, ly = (float(hp[0]) * 0.5, float(hp[1]) * 0.5) if hp else (0.0, -0.15)
                        with free.lock():
                            free.cam.lookat[0] = lx
                            free.cam.lookat[1] = ly
                            free.cam.lookat[2] = float(cam_z)
                            free.cam.distance  = float(cam_dist)
                            free.cam.azimuth   = float(cam_az)
                            free.cam.elevation = float(cam_el)
                    except Exception:
                        pass  # se l'API cam cambia, resta il default (zoom col mouse)
                    free.sync()
                except Exception as e:
                    print(f"[PLAY] --free-viewer non disponibile ({e}). Riprova col default "
                          f"(senza --free-viewer): usa il renderer robosuite.")
                    use_free = False

            steps = 0
            cum_reward = 0.0
            phase_time = {n: 0 for n in PHASE_NAMES}
            rew_accum = {}
            prev_phase = None
            done = False
            while not done:
                action, _ = model.predict(norm(obs), deterministic=True)
                obs, reward, term, trunc, info = env.step(action)
                steps += 1
                cum_reward += float(reward)
                for k, v in (info.get("reward_terms", {}) or {}).items():
                    rew_accum[k] = rew_accum.get(k, 0.0) + float(v)
                ph = int(info.get("fsm_phase", 0))
                phase_time[PHASE_NAMES[ph]] += 1

                # HUD: sempre alle transizioni di fase, altrimenti ogni hud_every step
                phase_changed = (ph != prev_phase)
                if phase_changed and prev_phase is not None:
                    print("  " + _c(f"▸▸▸ TRANSIZIONE {PHASE_NAMES[prev_phase]} → {PHASE_NAMES[ph]}"
                                     f"  (step {steps})", "bold"))
                prev_phase = ph
                if phase_changed or (steps % max(1, hud_every) == 0):
                    _hud_step(env, steps, action, float(reward), info, cum_reward)

                # render: --free-viewer → sync() del nostro viewer; altrimenti robosuite.
                if use_free and free is not None:
                    if not free.is_running():
                        print("[PLAY] Finestra viewer chiusa — esco.")
                        return
                    free.sync()
                elif on_screen_ok:
                    env.render()
                # PLAY-ONLY: rallenta la fase RETREAT (fase 3) così l'allontanamento del
                # braccio è ben visibile. NON tocca l'ambiente né il training: cambia solo
                # quanto a lungo si dorme tra un frame e l'altro.
                _slow = float(retreat_slow) if ph == 3 else 1.0
                time.sleep((1.0 / 30.0) * max(0.0, _slow))
                done = bool(term or trunc)

            _hud_summary(ep_idx, steps, phase_time, cum_reward, info, bool(term),
                         bool(trunc), rew_accum=rew_accum, open_tol=cfg.open_tol_rad)

            # PLAY-ONLY: pausa a fine episodio tenendo a video la posa finale (braccio
            # ritirato, porta aperta) prima del reset, così l'esito si vede con calma.
            _t_end = time.time()
            while (time.time() - _t_end) < float(end_pause):
                if use_free and free is not None:
                    if not free.is_running():
                        return
                    free.sync()
                elif on_screen_ok:
                    env.render()
                time.sleep(1.0 / 30.0)
    finally:
        if free is not None:
            try: free.close()
            except Exception: pass
        try: env.close()
        except Exception: pass


def main():
    ap = argparse.ArgumentParser(description="Apertura generalizzata v2 — curriculum 1")
    ap.add_argument("--total-steps", type=int, default=None)
    ap.add_argument("--play", action="store_true")
    ap.add_argument("--model", type=str, default=None)
    ap.add_argument("--episodes", type=int, default=None,
                    help="numero di episodi da riprodurre nel play (default: infinito)")
    ap.add_argument("--hud-every", type=int, default=1,
                    help="stampa l'HUD ogni N step (default 1 = ogni step). Le transizioni "
                         "di fase sono sempre stampate.")
    ap.add_argument("--free-viewer", action="store_true",
                    help="apre il viewer nativo MuJoCo con i PANNELLI impostazioni (UI) "
                         "visibili e camera col mouse. Su macOS richiede mjpython. Senza "
                         "questo flag si usa il renderer robosuite di default.")
    ap.add_argument("--cam-dist", type=float, default=2.6,
                    help="[--free-viewer] distanza iniziale della camera (default 2.6; "
                         "aumenta per zoomare INDIETRO).")
    ap.add_argument("--cam-az", type=float, default=135.0,
                    help="[--free-viewer] azimuth iniziale in gradi (default 135).")
    ap.add_argument("--cam-el", type=float, default=-20.0,
                    help="[--free-viewer] elevazione iniziale in gradi (default -20).")
    ap.add_argument("--cam-z", type=float, default=1.0,
                    help="[--free-viewer] altezza (z) del punto guardato (default 1.0 ≈ maniglia).")
    ap.add_argument("--retreat-slow", type=float, default=2.0,
                    help="rallenta la fase RETREAT di questo fattore per vedere bene "
                         "l'allontanamento del braccio (default 2.0; usa 1.0 per il TEMPO "
                         "REALE, o un valore più alto per rallentare di più). "
                         "Solo visualizzazione, NON tocca il training.")
    ap.add_argument("--end-pause", type=float, default=2.5,
                    help="secondi di pausa a fine episodio con la posa finale a video "
                         "(default 2.5). Solo visualizzazione.")
    args = ap.parse_args()

    cfg = TrainConfigV2Open()
    cfg.fixed_curriculum_level = 1.0   # questo progetto è SOLO curriculum 1
    if args.play:
        play(cfg, args.model,
             hud_every=args.hud_every,
             episodes=args.episodes,
             free_viewer=args.free_viewer,
             cam_dist=args.cam_dist, cam_az=args.cam_az,
             cam_el=args.cam_el, cam_z=args.cam_z,
             retreat_slow=args.retreat_slow, end_pause=args.end_pause)
    else:
        train(cfg, args.total_steps or cfg.total_steps)


if __name__ == "__main__":
    main()