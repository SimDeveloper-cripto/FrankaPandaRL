#!/usr/bin/env python3
# tests_v2/smoke_retreat_log_v2.py
#
# Smoke-test con logging passo-passo della sequenza HOLD → RETREAT
# per close_generalized_v2 — porting di test_override_grip.py (v1).
#
# A differenza della v1, NON serve riscrivere step() per iniettare l'override:
# in v2 il rilascio pulito (§1.17), il grip-lock (§1.18) e la rampa di avvio (§1.21) sono GIA' dentro
# env_v2.step().
#
# Questo script si limita a far girare la policy e a stampare, istante per
# istante in HOLD/RETREAT, le grandezze che rivelano il comportamento di quei fix:
#   - fase FSM, gripper width (dall'azione), larghezza presa, latch_qpos, door_qpos
#   - distingue "rilascio pulito in corso" (braccio fermo, gripper→aperto) dal ritiro vero.
#
# Serve a CONFERMARE nel play che:
#   * in RETREAT le dita si aprono PRIMA che il braccio si muova (§1.17)
#   * il latch torna verso 0 durante il rilascio
#   * (con §1.21) l'azione del braccio parte in rampa, non a scatto
#
# Uso:
#   python -m tests_v2.smoke_retreat_log_v2 --steps 200 --curriculum 1
#
# Rif.: rilascio basato sul contatto [13]; avvio morbido dell'opzione [1]

import os
import argparse
import numpy as np

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tests_v2._common_v2 import make_raw_env, load_model, load_obs_rms, predict, phase_idx_from_info, PHASE_NAMES


def run(max_steps, curriculum, run_dir):
    env, _cfg = make_raw_env(curriculum_level=curriculum, run_dir=run_dir)
    obs_rms   = load_obs_rms(run_dir=run_dir)
    model     = load_model(run_dir=run_dir)

    obs, info      = env.reset()
    steps          = 0
    logged_retreat = 0
    prev_phase     = -1

    while steps < max_steps:
        action, _ = predict(model, obs, obs_rms, deterministic=True)
        grip_cmd  = float(np.asarray(action).reshape(-1)[-1])  # ultimo = comando gripper
        arm_norm  = float(np.linalg.norm(np.asarray(action).reshape(-1)[:-1]))

        obs, _r, terminated, truncated, info = env.step(action)

        done  = bool(terminated or truncated)
        steps += 1
        pidx  = phase_idx_from_info(info)

        if pidx != prev_phase:
            print(f"--- entra in fase {PHASE_NAMES[pidx]} (step {steps}) ---")
            prev_phase = pidx

        # Logga i passi di HOLD e RETREAT (dove agiscono §1.17/§1.18/§1.21)
        if pidx in (2, 3):
            print(f"  [{PHASE_NAMES[pidx]:<7}] step={steps:>3} "
                  f"grip_cmd={grip_cmd:+.2f} arm|a|={arm_norm:.3f} "
                  f"latch={float(info.get('latch_qpos',0)):+.3f} "
                  f"door={float(info.get('door_qpos',0)):+.3f}")
            if pidx == 3:
                logged_retreat += 1

        if done:
            print(f"=== fine episodio: is_success={info.get('is_success')} "
                  f"door_end={float(info.get('door_qpos',0)):+.3f} "
                  f"latch_end={float(info.get('latch_qpos',0)):+.3f} ===")
            if logged_retreat > 0:
                break
            obs, info = env.reset()
            prev_phase = -1


def main():
    ap = argparse.ArgumentParser(description = "Smoke-test logging RETREAT v2")
    ap.add_argument("--steps",      type = int,    default = 200)
    ap.add_argument("--curriculum", type = float,  default = 1.0)
    ap.add_argument("--run-dir",    type = str,    default = "runs/close_gen_v2")
    args = ap.parse_args()
    run(args.steps, args.curriculum, args.run_dir)

if __name__ == "__main__":
    main()