#!/usr/bin/env python3

import os
import sys
import subprocess

PYTHON       = os.path.join(os.path.dirname(__file__), ".venv", "bin", "python")
ROOT         = os.path.dirname(os.path.abspath(__file__))
RUN_C0       = "runs/close_gen_v2_curriculum_0_new_110626"
RUN_C1       = "runs/close_gen_v2_curriculum_1_new_110626"

RESULTS_FILE = os.path.join(ROOT, "results_v2_raw.txt")

def run(cmd, label):
    print(f"\n{'='*70}")
    print(f"RUNNING: {label}")
    print(f"CMD: {' '.join(cmd)}")
    print(f"{'='*70}")

    result = subprocess.run(cmd, cwd = ROOT, capture_output = True, text = True)
    out    = result.stdout + ("\n[STDERR]\n" + result.stderr if result.stderr.strip() else "")

    print(out)
    with open(RESULTS_FILE, "a") as f:
        f.write(f"\n{'='*70}\n")
        f.write(f"### {label}\n")
        f.write(f"CMD: {' '.join(cmd)}\n")
        f.write(f"EXIT CODE: {result.returncode}\n")
        f.write(f"{'='*70}\n")
        f.write(out)
        f.write("\n")
    return result.returncode


if __name__ == "__main__":
    # Wipe previous results
    with open(RESULTS_FILE, "w") as f:
        f.write("# Risultati test v2\n\n")

    # --- TEST FUNZIONALE OFFLINE (white-box, no robosuite, no modello) ---
    # 1. Verifica le 7 proprietà della rampa §1.21 direttamente sul codice di env_v2.
    #    Non dipende dal curriculum né dal modello addestrato.
    run([PYTHON, "scratch/test_retreat_ramp_v2_wrapper.py"], "test_retreat_ramp_v2 (curriculum-indipendente)")

    # --- TEST END-TO-END BLACK-BOX — valutazione statistica (30 episodi) ---
    # 2-3. Carica il modello SAC e lo valuta in modalità det + sto su N episodi completi.
    #      Misura success rate permissivo e «true success» (porta chiusa + latch neutro).
    run([PYTHON, "eval_stats_v2.py",
        "--episodes", "30", "--curriculum", "0", "--run-dir", RUN_C0],
        "eval_stats_v2 — curriculum 0 (posa fissa)")

    run([PYTHON, "eval_stats_v2.py",
        "--episodes", "30", "--curriculum", "1", "--run-dir", RUN_C1],
        "eval_stats_v2 — curriculum 1 (posa variabile)")

    # --- TEST DIAGNOSTICO END-TO-END — analisi delle fasi (10 episodi) ---
    # 4-5. Come eval_stats ma logga per ogni episodio la durata per fase e il valore
    #      di latch_qpos all'istante della transizione HOLD→RETREAT.
    run([PYTHON, "close_generalized_v2/diag_phase34_v2.py",
        "--episodes", "10", "--curriculum", "0", "--run-dir", RUN_C0],
        "diag_phase34_v2 — curriculum 0 (posa fissa)")

    run([PYTHON, "close_generalized_v2/diag_phase34_v2.py",
        "--episodes", "10", "--curriculum", "1", "--run-dir", RUN_C1],
        "diag_phase34_v2 — curriculum 1 (posa variabile)")

    # --- SMOKE TEST END-TO-END — log visivo passo-passo (nessuna asserzione) ---
    # 6-7. Esegue un singolo episodio e stampa ogni step di HOLD/RETREAT in tempo reale.
    #      Serve a ispezionare visivamente il rilascio §1.17 e la rampa §1.21.
    run([PYTHON, "scratch/smoke_retreat_log_v2.py",
        "--steps", "300", "--curriculum", "0", "--run-dir", RUN_C0],
        "smoke_retreat_log_v2 — curriculum 0 (posa fissa)")

    run([PYTHON, "scratch/smoke_retreat_log_v2.py",
        "--steps", "300", "--curriculum", "1", "--run-dir", RUN_C1],
        "smoke_retreat_log_v2 — curriculum 1 (posa variabile)")

    print(f"\nAll done. Results saved to {RESULTS_FILE}")