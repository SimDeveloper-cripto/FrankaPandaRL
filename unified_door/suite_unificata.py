#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
suite_unificata.py — le batterie di verifica della reward machine unificata.

Rispecchia le suite dei due progetti separati (`scratch/test_*_task_v2/`), che
hanno sei batterie: functional, physics, evaluate, phase, robustness, ablation.
Qui ci sono le tre che mancavano — `physics`, `phase`, `robustness` — perche' le
altre tre esistono gia':

    functional -> tests/test_unified.py     (244 controlli)
    evaluate   -> train_unified.py --eval   (200 episodi × 3 semi)
    ablation   -> ablazione.py              (un override per volta)

Non tocca l'implementazione: legge, esegue e misura.

Uso:
    export PROGETTI_ORIGINALI="$(cd .. && pwd)"
    python3 suite_unificata.py physics    --task close
    python3 suite_unificata.py phase      --task close --episodes 100
    python3 suite_unificata.py robustness --task close --episodes 300
"""
import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_unified import UnifiedConfig            # noqa: E402
from fsm_unified import Fase                        # noqa: E402
import env_unified as EU                            # noqa: E402
import train_unified as T                           # noqa: E402

FASI = ("REACH", "MOVE", "HOLD", "RELEASE", "FINE")


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def carica(task):
    class A:
        pass
    a = A()
    a.task, a.seed, a.run_dir = task, 42, f"runs/unified_{task}"
    cfg = UnifiedConfig.per(task)
    env, model = T.carica(cfg, a)
    grezzo = env.venv.envs[0] if hasattr(env, "venv") else env.envs[0]
    while not hasattr(grezzo, "_eef") and hasattr(grezzo, "env"):
        grezzo = grezzo.env
    return cfg, env, model, grezzo


# ═════════════════════════════════════════════════════════════════════════════
# PHYSICS — proprieta' del modello MuJoCo, senza politica
# ═════════════════════════════════════════════════════════════════════════════
def physics(task, _episodi):
    cfg = UnifiedConfig.per(task)
    e = EU.UnifiedDoorEnv(cfg)
    e.reset()
    # NB: `reset()` ricostruisce il simulatore, quindi il riferimento va ripreso
    # DOPO ogni reset, altrimenti si scrive su un oggetto orfano.
    ad = e._rs.handle_qpos_addr
    sim = e._rs.sim
    esiti = []

    def prova(nome, ok, dettaglio):
        esiti.append({"prova": nome, "esito": bool(ok), "dettaglio": dettaglio})
        print(f"  [{'ok ' if ok else 'NO '}] {nome:44s} {dettaglio}")

    # P1 — la molla riporta la leva a riposo, da entrambi i lati
    for segno in (+1.0, -1.0):
        e.reset()
        sim = e._rs.sim
        sim.data.qpos[ad] = segno * 1.5
        sim.data.qvel[:] = 0.0
        sim.forward()
        a = np.zeros(7, dtype=np.float32)
        a[-1] = -1.0
        for _ in range(200):
            e._rs.step(a)
        fin = float(e._rs.sim.data.qpos[ad])
        prova(f"P1 molla: leva {segno:+.1f} torna a riposo",
              abs(fin) <= cfg.thr.latch_term_tol, f"|leva| finale {abs(fin):.3f} ≤ {cfg.thr.latch_term_tol}")

    # P2 — il verso che ABBASSA la maniglia e' il positivo
    sim = e._rs.sim
    gid = [g for g in range(sim.model.ngeom) if sim.model.geom_id2name(g) == "Door_handle"][0]
    z = {}
    for q in (-1.5, 0.0, +1.5):
        sim.data.qpos[ad] = q
        sim.forward()
        z[q] = float(sim.data.geom_xpos[gid][2])
    prova("P2 leva positiva = maniglia abbassata",
          z[+1.5] < z[0.0] < z[-1.5],
          f"z: {z[-1.5]:.3f} (su) · {z[0.0]:.3f} (riposo) · {z[+1.5]:.3f} (giu)")
    prova("P2 il verso dichiarato coincide", cfg.thr.verso_leva == +1.0,
          f"verso_leva = {cfg.thr.verso_leva:+.1f}")

    # P3 — il catenaccio blocca la porta a leva ferma
    e.reset()
    sim = e._rs.sim
    sim.data.qpos[ad] = 0.0
    theta0 = 0.30 if task == "close" else 0.0
    sim.data.qpos[e._rs.hinge_qpos_addr] = theta0
    sim.data.qvel[:] = 0.0
    sim.forward()
    a = np.zeros(7, dtype=np.float32)
    a[-1] = -1.0
    for _ in range(300):
        e._rs.step(a)
    theta = float(e._rs.sim.data.qpos[e._rs.hinge_qpos_addr])
    prova("P3 a leva ferma il catenaccio trattiene la porta",
          abs(theta - theta0) < 0.05,
          f"da {theta0:+.3f} resta a {theta:+.4f} rad: non completa la corsa")

    # P4 — i range di randomizzazione e il campionamento del bersaglio
    raggi, attriti, rigid, bers, pose = [], [], [], [], []
    for i in range(60):
        np.random.seed(9000 + i)
        e.reset()
        raggi.append(e._rand.current_handle_radius)
        attriti.append(e._rand.current_handle_friction)
        rigid.append(e._rand.current_latch_stiffness / (e._rand.base_latch_stiffness or 1.0))
        amp = e.corsa_max - e.corsa_min
        bers.append((e.door.theta_star - e.corsa_min) / amp)
        pose.append((e.door.theta_zero - e.corsa_min) / amp)
    lo_b, hi_b = cfg.task.theta_star_frac
    lo_p, hi_p = cfg.task.theta_zero_frac
    prova("P4 bersaglio dentro la frazione dichiarata",
          lo_b - 1e-6 <= min(bers) and max(bers) <= hi_b + 1e-6,
          f"θ* in [{min(bers):.3f}, {max(bers):.3f}] · dichiarato [{lo_b}, {hi_b}]")
    prova("P4 posa iniziale dentro la frazione dichiarata",
          lo_p - 1e-6 <= min(pose) and max(pose) <= hi_p + 1e-6,
          f"θ₀ in [{min(pose):.3f}, {max(pose):.3f}] · dichiarato [{lo_p}, {hi_p}]")
    prova("P4 la randomizzazione varia davvero",
          np.std(raggi) > 0 and np.std(attriti) > 0 and np.std(rigid) > 0,
          f"raggio [{min(raggi):.4f}, {max(raggi):.4f}] · attrito [{min(attriti):.2f}, "
          f"{max(attriti):.2f}] · cricchetto [×{min(rigid):.2f}, ×{max(rigid):.2f}]")
    e.close()
    return {"prove": esiti, "superate": sum(x["esito"] for x in esiti), "totale": len(esiti)}


# ═════════════════════════════════════════════════════════════════════════════
# PHASE — diagnostica per fase su N episodi
# ═════════════════════════════════════════════════════════════════════════════
def phase(task, n_ep):
    cfg, env, model, g = carica(task)
    per_fase = {f: {"n": 0, "R": 0.0, "ep": 0} for f in FASI}
    raggiunta = {f: 0 for f in FASI}
    regressi, ritorni, passi_tot, veri = 0, [], [], 0
    hold_norm, wrist_rel, spost = [], [], []
    for i in range(n_ep):
        np.random.seed(50_000 + i)
        obs = env.reset()
        conta = {f: 0 for f in FASI}
        somma = {f: 0.0 for f in FASI}
        tot, k, vista_hold, err_dopo = 0.0, 0, False, []
        while True:
            act = model.predict(obs, deterministic=True)[0]
            obs, r, d, infos = env.step(act)
            info = infos[0]
            f = FASI[min(int(info["fsm_phase"]), 4)]
            conta[f] += 1
            somma[f] += float(r[0])
            tot += float(r[0])
            k += 1
            a = np.clip(act[0], -1, 1)
            if f == "HOLD":
                hold_norm.append(float(np.linalg.norm(a[:-1])))
                vista_hold = True
            if f == "RELEASE":
                wrist_rel.append(float(np.linalg.norm(a[3:6])))
            if vista_hold:
                err_dopo.append(abs(float(info["door_error"])))
            if d[0]:
                break
        for f in FASI:
            if conta[f]:
                per_fase[f]["n"] += conta[f]
                per_fase[f]["R"] += somma[f]
                per_fase[f]["ep"] += 1
                raggiunta[f] += 1
        ritorni.append(tot)
        passi_tot.append(k)
        veri += int(info["is_success"])
        spost.append(float(info.get("ritiro_spostato", 0.0)))
        if err_dopo and max(err_dopo) > cfg.task.tol + 0.02:
            regressi += 1
    env.close()

    print(f"\n  {'fase':<9} {'episodi':>8} {'passi medi':>11} {'R medio':>10} {'R/passo':>9}")
    righe = {}
    for f in FASI:
        d = per_fase[f]
        if not d["ep"]:
            continue
        righe[f] = {"episodi": d["ep"], "passi_medi": d["n"] / d["ep"],
                    "R_medio": d["R"] / d["ep"], "R_per_passo": d["R"] / max(d["n"], 1)}
        print(f"  {f:<9} {d['ep']:>5}/{n_ep:<3} {d['n']/d['ep']:>11.1f} "
              f"{d['R']/d['ep']:>10.1f} {d['R']/max(d['n'],1):>9.2f}")
    lo, hi = wilson(veri, n_ep)
    print(f"\n  true success {veri}/{n_ep} · IC95 [{lo:.3f}, {hi:.3f}]")
    print(f"  norma dell'azione in HOLD (braccio bloccato): {np.mean(hold_norm):.4f}")
    print(f"  rotazione del polso in RELEASE, per passo   : {np.mean(wrist_rel):.3f}")
    print(f"  spostamento del ritiro                      : {np.mean(spost):.3f} m")
    print(f"  episodi con regressione della porta dopo HOLD: {regressi}/{n_ep}")
    return {"n_episodi": n_ep, "fasi": righe, "true": veri, "ic95": [round(lo, 3), round(hi, 3)],
            "hold_action_norm": float(np.mean(hold_norm)),
            "wrist_release": float(np.mean(wrist_rel)),
            "ritiro_spostato": float(np.mean(spost)),
            "regressioni": regressi, "ritorno_medio": float(np.mean(ritorni)),
            "passi_medi": float(np.mean(passi_tot))}


# ═════════════════════════════════════════════════════════════════════════════
# ROBUSTNESS — esito contro parametro REALIZZATO, a randomizzazione naturale
# ═════════════════════════════════════════════════════════════════════════════
ASSI = {"raggio_maniglia": "raggio della maniglia (m)",
        "attrito_maniglia": "attrito della maniglia",
        "rigidita_cricchetto": "rigidità del cricchetto (×base)",
        "bersaglio_frazione": "bersaglio (frazione della corsa)"}


def robustness(task, n_ep):
    cfg, env, model, g = carica(task)
    rec = []
    for i in range(n_ep):
        np.random.seed(60_000 + i)
        obs = env.reset()
        base = g._rand.base_latch_stiffness or 1.0
        amp = g.corsa_max - g.corsa_min
        p = {"raggio_maniglia": g._rand.current_handle_radius,
             "attrito_maniglia": g._rand.current_handle_friction,
             "rigidita_cricchetto": g._rand.current_latch_stiffness / base,
             "bersaglio_frazione": (g.door.theta_star - g.corsa_min) / amp}
        while True:
            act = model.predict(obs, deterministic=True)[0]
            obs, r, d, infos = env.step(act)
            info = infos[0]
            if d[0]:
                break
        p["true"] = bool(info["is_success"])
        p["clean"] = bool(info["successo_pulito"])
        rec.append(p)
    env.close()

    n = len(rec)
    kt = sum(r["true"] for r in rec)
    kc = sum(r["clean"] for r in rec)
    lo, hi = wilson(kt, n)
    print(f"\n  true  {kt}/{n} · IC95 [{lo:.3f}, {hi:.3f}]")
    lo2, hi2 = wilson(kc, n)
    print(f"  clean {kc}/{n} · IC95 [{lo2:.3f}, {hi2:.3f}]")
    out = {"n_episodi": n, "true": kt, "clean": kc, "ic95": [round(lo, 3), round(hi, 3)], "assi": {}}
    for chiave, etichetta in ASSI.items():
        v = np.array([r[chiave] for r in rec], float)
        if v.std() < 1e-9:
            continue
        bordi = np.quantile(v, [0.0, 0.25, 0.5, 0.75, 1.0])
        bordi[-1] += 1e-9
        print(f"\n  {etichetta}")
        bins = []
        for a, b in zip(bordi[:-1], bordi[1:]):
            m = (v >= a) & (v < b)
            nb = int(m.sum())
            if nb == 0:
                continue
            kb = int(sum(r["true"] for r, mm in zip(rec, m) if mm))
            l, h = wilson(kb, nb)
            bins.append({"da": float(a), "a": float(b), "n": nb, "true": kb,
                         "rate": kb / nb, "ic95": [round(l, 3), round(h, 3)]})
            print(f"     [{a:7.4f}, {b:7.4f})  n={nb:>4}  true {kb:>4}/{nb:<4} = {kb/nb:.3f}  [{l:.3f}, {h:.3f}]")
        out["assi"][chiave] = bins
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("batteria", choices=["physics", "phase", "robustness"])
    ap.add_argument("--task", choices=["close", "open"], required=True)
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    print(f"\n{args.batteria} · {args.task} · {args.episodes} episodi\n")
    fn = {"physics": physics, "phase": phase, "robustness": robustness}[args.batteria]
    ris = fn(args.task, args.episodes)
    out = args.out or f"suite_{args.batteria}_{args.task}.json"
    json.dump(ris, open(out, "w"), indent=1)
    print(f"\nsalvato in {out}")


if __name__ == "__main__":
    main()
