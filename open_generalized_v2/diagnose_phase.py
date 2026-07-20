#!/usr/bin/env python3
# open_generalized_v2/diagnose_phase.py
#
# DIAGNOSTICO (non tocca il training): scopre DOVE si blocca la catena
# REACH → PULL → HOLD_OPEN → RETREAT, che nei log resta a success_rate=0 / ep_len=600.
#
# Gira N episodi con la policy addestrata (se c'è best_model.zip) oppure con azioni
# CASUALI (--random), e per ogni episodio riporta:
#   - fase massima raggiunta + step spesi per fase
#   - nei frame di REACH: gripper_width, is_physically_closed, dist_handle vs soglia adattiva
#   - escursione di door_angle (quanto si muove la porta) vs goal
# Stampa anche, a inizio episodio, i valori di calibrazione (handle_radius, handle_diam,
# finestra di "presa chiusa", soglia di distanza) per capire se le condizioni di
# transizione sono anche solo RAGGIUNGIBILI.
#
# Uso:
#   python -m open_generalized_v2.diagnose_phase --episodes 5            # policy addestrata
#   python -m open_generalized_v2.diagnose_phase --episodes 5 --random   # azioni casuali
#   python -m open_generalized_v2.diagnose_phase --episodes 5 --open-gripper  # bias presa chiusa

from __future__ import annotations

import os
import sys
import pickle
import argparse
import numpy as np

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from open_generalized_v2.config_v2 import TrainConfigV2Open
    from open_generalized_v2.env_v2 import AdvancedGeneralizedOpenDoorEnv
    from open_generalized_v2 import fsm_v2
except ModuleNotFoundError:
    from config_v2 import TrainConfigV2Open
    from env_v2 import AdvancedGeneralizedOpenDoorEnv
    import fsm_v2

PHASE_NAMES = ["REACH", "PULL", "HOLD_OPEN", "RETREAT"]


def _find_any_model(cfg):
    """Trova QUALSIASI modello su disco: best > final > checkpoint più recente > qualsiasi .zip.
    Con success=0 il best_model è salvato a 0%; se il training è stato interrotto a metà
    e non c'è best/final, recupera l'ultimo checkpoint salvato."""
    rd = cfg.run_dir
    for name in ("latest_model.zip", "best_model.zip", "final_model.zip"):
        cand = os.path.join(rd, name)
        if os.path.exists(cand):
            return cand
    if os.path.isdir(rd):
        zips = [os.path.join(rd, f) for f in os.listdir(rd) if f.endswith(".zip")]
        if zips:
            return max(zips, key=os.path.getmtime)   # il più recente
    return None


def load_policy(cfg, model_path):
    """Ritorna (predict_fn, descr). Se non c'è il modello, usa azioni casuali."""
    path = model_path or _find_any_model(cfg)
    if path is None or not os.path.exists(path):
        return None, f"(nessun modello in {cfg.run_dir})"
    from stable_baselines3 import SAC
    obs_rms = None
    vn = os.path.join(cfg.run_dir, "vecnormalize.pkl")
    if os.path.exists(vn):
        with open(vn, "rb") as f:
            obs_rms = pickle.load(f).obs_rms
    model = SAC.load(path)

    def predict(obs):
        o = obs
        if obs_rms is not None:
            o = np.clip((obs - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8), -10.0, 10.0)
        a, _ = model.predict(o, deterministic=True)
        return a
    return predict, f"policy addestrata ({path})"


def run(episodes, model_path, use_random, open_gripper, scripted=False):
    cfg = TrainConfigV2Open()
    cfg.fixed_curriculum_level = 1.0
    env = AdvancedGeneralizedOpenDoorEnv(cfg)
    env.set_curriculum_level(1.0)

    # stato condiviso per la mano guidata (legge il vettore eef->handle dall'info)
    _last = {"vec": None}

    predict, descr = load_policy(cfg, model_path)
    if scripted:
        descr = "MANO GUIDATA (scripted: punta alla maniglia + chiude)"
        act_dim = env.action_space.shape[0]
        def predict(obs):  # noqa
            a = np.zeros(act_dim, dtype=np.float32)
            v = _last["vec"]
            if v is not None:
                # OSC: primi 3 = spostamento cartesiano dell'eef verso la maniglia
                step_xyz = np.clip(np.asarray(v, dtype=np.float32) * 5.0, -1.0, 1.0)
                a[:3] = step_xyz
            a[-1] = 1.0   # gripper chiuso
            return a
    elif use_random or predict is None:
        descr = "AZIONI CASUALI" if use_random else descr + " → fallback CASUALE"
        act_dim = env.action_space.shape[0]

        def predict(obs):  # noqa
            a = np.random.uniform(-1, 1, size=act_dim).astype(np.float32)
            if open_gripper:        # bias: tieni il gripper comandato chiuso
                a[-1] = 1.0
            return a

    print(f"\n=== DIAGNOSTICO FASE — {descr} ===")
    fsm = fsm_v2.AdaptiveFSMOpen(cfg)  # solo per leggere le soglie adattive

    for ep in range(episodes):
        obs, info = env.reset()
        # calibrazione iniziale
        hr = env._domain_rand.current_handle_radius
        hd = hr * 2.0
        d_thresh = fsm.grasp_dist_thresh(hr)
        g_thresh = fsm.grip_thresh(env._domain_rand.current_handle_friction)
        print(f"\n--- EPISODIO {ep+1} ---")
        print(f"  handle_radius={hr:.4f}  handle_diam={hd:.4f}  "
              f"finestra presa-chiusa: [0.015, {hd+0.025:.4f}]")
        print(f"  soglia distanza d_thresh={d_thresh:.4f}  soglia gripper g_thresh={g_thresh:.3f}")
        print(f"  goal_angle={info['goal_angle']:.3f}  door_min={env._door_min:.3f}  "
              f"eff_max={env._effective_max:.3f}")

        phase_time = {n: 0 for n in PHASE_NAMES}
        max_phase = 0
        door_min_seen, door_max_seen = 1e9, -1e9
        gw_min, gw_max = 1e9, -1e9
        dist_min = 1e9
        phys_closed_ever = False
        reach_samples = []
        open_err_min = 1e9          # quanto vicino al goal è arrivata la porta (best)
        open_err_final = None        # errore al goal a fine episodio
        success_ever = False         # is_success visto almeno una volta
        term_clean = False           # episodio terminato pulito (RETREAT completo)
        last_retreat_steps = 0       # §1.43: per distinguere PULITA da HARD-CAP
        last_latch = float("nan")

        done = False
        steps = 0
        retreat_trace = []   # §1.38-diag: (rs, door_angle, door_qvel, gripper_width, latch)
        while not done:
            a = predict(obs)
            obs, r, term, trunc, info = env.step(a)
            done = bool(term or trunc)
            steps += 1
            # §1.38-diag — traccia il RETREAT step-by-step: i 3 numeri che discriminano la
            # causa del rimbalzo (velocità porta all'ingresso vs apertura gripper vs braccio).
            if int(info["fsm_phase"]) == 3:
                retreat_trace.append((
                    int(info.get("retreat_steps", -1)),
                    round(float(info.get("door_angle", float("nan"))), 4),
                    round(float(info.get("door_qvel", float("nan"))), 4),
                    round(float(info.get("gripper_width", float("nan"))), 4),
                    round(float(info.get("latch_qpos", float("nan"))), 4),
                    round(float(info.get("retreat_moved", 0.0)), 4),
                ))
            _last["vec"] = info.get("vec_eef_to_handle", None)
            p = int(info["fsm_phase"])
            if bool(info.get("is_success", False)):
                success_ever = True
            oe = float(info.get("open_error", float("nan")))
            if oe == oe:  # not nan
                open_err_min = min(open_err_min, oe)
                open_err_final = oe
            if done:
                term_clean = bool(term)  # term=True → terminazione pulita; trunc → orizzonte
                last_retreat_steps = int(info.get("retreat_steps", 0))
                last_latch = float(info.get("latch_qpos", float("nan")))
            if steps == 1:
                print(f"  [obs] handle_src={info.get('handle_src')}  "
                      f"eef_pos={info.get('eef_pos')}  handle_pos={info.get('handle_pos')}")
                print(f"  [obs] obs_keys(sample)={info.get('obs_keys_sample')}")
            phase_time[PHASE_NAMES[p]] += 1
            max_phase = max(max_phase, p)
            da = float(info["door_angle"])
            door_min_seen = min(door_min_seen, da)
            door_max_seen = max(door_max_seen, da)
            gw = float(env._prev_gripper_width)
            gw_min = min(gw_min, gw); gw_max = max(gw_max, gw)
            if gw <= hd + 0.025 and gw >= 0.015:
                phys_closed_ever = True
            # dist_handle letta DALL'INFO dell'env (stesso valore usato dalla FSM)
            dist = float(info.get("dist_handle", float("nan")))
            dist_min = min(dist_min, dist)
            if p == 0 and len(reach_samples) < 3:
                reach_samples.append((round(gw, 4), round(dist, 4), round(float(a[-1]), 2)))

        print(f"  → fase MAX raggiunta: {PHASE_NAMES[max_phase]}")
        print(f"  → step per fase: {phase_time}")
        print(f"  → door_angle escursione: [{door_min_seen:.3f}, {door_max_seen:.3f}] "
              f"(si è mossa di {door_max_seen-door_min_seen:.3f} rad)")
        print(f"  → gripper_width range: [{gw_min:.4f}, {gw_max:.4f}]  "
              f"presa-chiusa mai vera? {'SÌ almeno una volta' if phys_closed_ever else 'MAI'}")
        print(f"  → dist_handle minima: {dist_min:.4f} (soglia {d_thresh:.4f}) "
              f"{'RAGGIUNTA' if dist_min < d_thresh else 'MAI sotto soglia'}")
        if reach_samples:
            print(f"  → primi campioni REACH (gw, dist, grip_act): {reach_samples}")
        # §1.43/§1.45 — distingui i TRE esiti di terminazione (prima incastro e uscita
        # esogena venivano stampati come "PULITA"):
        #   PULITA   = leva tornata a casa (|latch| < tol) → il ritiro è riuscito davvero
        #   ESOGENA  = rete di sicurezza a retreat_exo_exit_steps: episodio chiuso ma leva
        #              NON tornata → il rilascio è ancora da sistemare
        #   HARD-CAP = guardia estrema
        _cap = int(getattr(env.cfg, "retreat_hard_cap", 90))
        _ltol = float(getattr(env.cfg, "retreat_latch_term_tol", 0.08))
        if term_clean and last_retreat_steps >= _cap:
            _term_lbl = f"HARD-CAP ({last_retreat_steps} step, latch={last_latch:+.3f} → INCASTRO leva)"
        elif term_clean and abs(last_latch) < _ltol:
            _term_lbl = f"PULITA (RETREAT completo in {last_retreat_steps} step, latch={last_latch:+.3f})"
        elif term_clean:
            _term_lbl = (f"ESOGENA ({last_retreat_steps} step, latch={last_latch:+.3f} → "
                         f"leva NON tornata, rilascio da sistemare)")
        else:
            _term_lbl = "troncata a orizzonte"
        print(f"  → SUCCESS (is_success visto)? {'SÌ' if success_ever else 'NO'}   "
              f"terminazione: {_term_lbl}")
        oem = f"{open_err_min:.4f}" if open_err_min < 1e8 else "n/d"
        oef = f"{open_err_final:.4f}" if open_err_final is not None else "n/d"
        print(f"  → open_error: minimo={oem}  finale={oef}  (tol={env.cfg.open_tol_rad:.3f})  "
              f"→ {'porta ARRIVATA al goal' if open_err_min <= env.cfg.open_tol_rad else 'goal MAI centrato entro tol'}")
        # §1.38-diag — traccia RETREAT: rs | door | door_qvel | gripper_width | latch
        if retreat_trace:
            d0 = retreat_trace[0][1]
            dmin = min(t[1] for t in retreat_trace)
            print(f"  → RETREAT trace (door0={d0:.3f}  door_min={dmin:.3f}  RIMBALZO={d0-dmin:.4f}):")
            print(f"       rs | door_angle | door_qvel | grip_width | latch    | ARRETRATO")
            for (rs, da, dv, gw2, lq, mv) in retreat_trace:
                print(f"      {rs:3d} |   {da:6.4f}  |  {dv:+7.4f} |   {gw2:6.4f}  | {lq:+.4f} |  {mv:.4f} m")

    env.close()
    print("\n=== LETTURA (policy addestrata) ===")
    print(" - fase MAX = REACH                  → non afferra: problema presa/avvicinamento.")
    print(" - fase MAX = PULL, goal MAI centrato → apre ma non raggiunge goal_angle entro tol.")
    print(" - SUCCESS sì ma terminazione troncata → entra in HOLD_OPEN ma non completa il RETREAT")
    print("   (o non MANTIENE l'apertura): il collo di bottiglia è HOLD_OPEN/RETREAT, non REACH.")
    print(" - open_error minimo ≈ tol ma finale ≫ tol → tocca il goal e poi si richiude.")


def main():
    ap = argparse.ArgumentParser(description="Diagnostico fase apertura v2")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--model", type=str, default=None)
    ap.add_argument("--random", action="store_true", help="usa azioni casuali invece del modello")
    ap.add_argument("--open-gripper", action="store_true", help="con --random, forza gripper chiuso (+1)")
    ap.add_argument("--scripted", action="store_true", help="mano GUIDATA: punta alla maniglia (OSC) e chiude la presa")
    args = ap.parse_args()
    run(args.episodes, args.model, args.random, args.open_gripper, args.scripted)


if __name__ == "__main__":
    main()