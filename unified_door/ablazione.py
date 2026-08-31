#!/usr/bin/env python3

"""
Uso:
    export PROGETTI_ORIGINALI="$(cd .. && pwd)"
    python3 ablazione.py --task close --episodes 50
    python3 ablazione.py --task open  --episodes 50 --seeds 42,101,7
"""

import os
import sys
import math
import copy
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_unified import UnifiedConfig
from fsm_unified    import Fase

import train_unified as T

VARIANTI = {
    "baseline":                {},
    "senza_riporto_leva":      {"task.riporto_leva": False},
    "senza_fuga":              {"task.escape": False},
    "senza_normale_orientata": {"task.orienta_normale": False},
    "consegna_alla_soglia":    {"task.leva_consegna": 0.15},
    "riporto_lento":           {"thr.riporto_mag_max": 0.6},
    "senza_blocco_hold":       {"thr.stall_guard_steps": 10_000},
    "senza_morsa":             {"thr.grip_lock_margin": -10.0},
    "senza_override":          {"task.riporto_leva": False, "task.escape": False},
}

def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def applica(cfg, modifiche: dict) -> None:
    for chiave, valore in modifiche.items():
        dove, campo = chiave.split(".", 1)
        oggetto     = cfg.task if dove == "task" else cfg.thr
        if not hasattr(oggetto, campo):
            raise KeyError(f"parametro inesistente: {chiave}")
        setattr(oggetto, campo, valore)


def valuta(task: str, semi, n_ep: int, modifiche: dict) -> dict:
    class A:
        pass
    a = A()
    a.task, a.seed, a.run_dir = task, semi[0], f"runs/unified_{task}"
    cfg = UnifiedConfig.per(task)

    cfg.task = copy.deepcopy(cfg.task)
    cfg.thr  = copy.deepcopy(cfg.thr)
    applica(cfg, modifiche)

    env, model = T.carica(cfg, a)
    grezzo     = env.venv.envs[0] if hasattr(env, "venv") else env.envs[0]
    while not hasattr(grezzo, "_eef") and hasattr(grezzo, "env"):
        grezzo = grezzo.env

    visti, righe = set(), []
    for seme in semi:
        for i in range(n_ep):
            if seme + i in visti:
                continue
            visti.add(seme + i)
            np.random.seed(seme + i)
            obs  = env.reset()
            dist = 0.0
            while True:
                act = model.predict(obs, deterministic=True)[0]
                obs, r, d, infos = env.step(act)
                info = infos[0]
                if int(info["fsm_phase"]) == Fase.RELEASE:
                    dist = float(np.linalg.norm(
                        np.asarray(grezzo._eef, float) - np.asarray(grezzo._handle, float)))
                if d[0]:
                    break
            righe.append((bool(info["is_success"]), bool(info["successo_pulito"]), dist))
    env.close()

    n  = len(righe)
    k  = sum(1 for x in righe if x[0])
    kc = sum(1 for x in righe if x[1])
    lo, hi = wilson(k, n)
    return {"n": n, "true": k, "clean": kc, "true_rate": k / n, "clean_rate": kc / n,
            "ic95": [round(lo, 3), round(hi, 3)],
            "dist_maniglia": float(np.mean([x[2] for x in righe]))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task",     choices=["close", "open"], required=True)

    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--seeds",    type=str, default="42")
    ap.add_argument("--varianti", type=str, default=None, help="sottoinsieme separato da virgola; se assente, tutte")
    ap.add_argument("--out",      type=str, default=None)
    args = ap.parse_args()

    semi = [int(s) for s in args.seeds.split(",") if s.strip()]
    nomi = ([v for v in args.varianti.split(",") if v.strip()]
            if args.varianti else list(VARIANTI))

    print(f"\nablazione · {args.task} · {args.episodes} episodi × semi {semi}\n")
    print(f"{'variante':26s} {'true':>11} {'clean':>11} {'IC 95 %':>16} {'maniglia':>10}   Δ true")
    esiti, base = {}, None
    for nome in nomi:
        e = valuta(args.task, semi, args.episodes, VARIANTI[nome])
        esiti[nome] = e
        if nome == "baseline":
            base = e["true_rate"]
        delta = "" if base is None else f"{e['true_rate'] - base:+.3f}"
        print(f"{nome:26s} {e['true']:>4}/{e['n']:<6} {e['clean']:>4}/{e['n']:<6} "
              f"[{e['ic95'][0]:.3f}, {e['ic95'][1]:.3f}] {e['dist_maniglia']:>10.3f}   {delta}")

    out = args.out or f"ablazione_{args.task}.json"
    json.dump({"task": args.task, "episodi": args.episodes, "semi": semi,
               "esiti": esiti}, open(out, "w"), indent=1)
    print(f"\nsalvato in {out}")


if __name__ == "__main__":
    main()