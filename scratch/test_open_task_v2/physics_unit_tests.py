#!/usr/bin/env python3
# scratch/test_open_task_v2/physics_unit_tests.py
"""
physics_unit_tests — Test di proprietà FISICHE dell'ambiente di APERTURA (deterministici).

Non coinvolgono la policy: verificano che il TASK sia fisicamente ben posto. Se uno di
questi fallisce, nessun risultato di RL sopra è interpretabile (Henderson et al. 2018:
gran parte della varianza attribuita agli algoritmi è in realtà dell'ambiente).

  T1 — MOLLA DEL LATCH. La leva, rilasciata a fondo corsa, torna a neutro in tempo
       finito. È la precondizione dell'uscita PULITA del RETREAT: reward_v2 termina
       l'episodio con bonus solo se |latch| < retreat_latch_term_tol. Identico alla
       chiusura (il modulo fisico è lo stesso), ma qui è ancora più critico perché
       nell'apertura la leva arriva a fondo corsa (~1.57 rad) e deve scaricarsi.

  T2 — RITENZIONE DELLA PORTA APERTA (specchio del test di rimbalzo della chiusura).
       Nella chiusura si misura il rimbalzo dopo l'urto contro il telaio; nell'apertura
       la proprietà simmetrica è che la porta, portata al goal e LASCIATA LIBERA, ci
       RESTI. Se derivasse verso la chiusura da sola, HOLD_OPEN e il true-success
       sarebbero irraggiungibili per costruzione, non per colpa della policy.
       Si misura anche la risposta a un piccolo impulso di richiusura (il "RIMBALZO"
       documentato nelle tracce di RETREAT): riportato come diagnostica.

  T7 — RANDOMIZATION ESTESA (§3.4): raggio, frizione, rigidità latch, smorzamento
       cerniera, massa porta restano nei range dichiarati (Tobin et al. 2017;
       Zhao et al. 2020). Serve perché la FSM tara le sue soglie su questi valori
       (fsm_v2 §3.1: grip_thresh sulla frizione, hold_steps sulla rigidità del latch).

  T8 — CAMPIONAMENTO DEL GOAL (specifico dell'apertura). `goal_angle` è ricampionato a
       ogni reset in [goal_frac_min, goal_frac_max] × range effettivo: si verifica che
       stia nel range e non sia degenere. Senza questo, la "generalizzazione al goal"
       non sarebbe testata affatto.

Compatibile con `pytest`. Output in results/physics/.
"""

from __future__ import annotations

import os
import json
import numpy as np

from _common import (find_repo_root, make_raw_env, results_dir, setup_matplotlib,
                     safe_hist, json_default, resolve_run_dir)

find_repo_root()
import robosuite as suite  # noqa: E402

LATCH_NEUTRAL = 0.15            # = cfg.retreat_latch_term_tol (soglia di uscita pulita)
LATCH_RETURN_MAX_STEPS = 90     # 3 s a 30 Hz
RETENTION_MAX_DRIFT = 0.05      # = cfg.open_tol_rad
RETENTION_STEPS = 60            # 2 s a 30 Hz
DEFAULT_PHYSICS_SEED = 12_345   # rende riproducibile il placement randomizzato
RETENTION_MAX_ATTEMPTS = 8      # tentativi per trovare una posa senza contatto


def _make_rs_door():
    return suite.make("Door", robots="Panda", has_renderer=False,
                      has_offscreen_renderer=False, use_camera_obs=False,
                      reward_shaping=True, control_freq=30, ignore_done=True)


def _find_joint(sim, must, forbid=()):
    """Trova un giunto per pattern sul nome (robusto ai nomi del modello)."""
    for name in sim.model.joint_names:
        low = name.lower()
        if all(m in low for m in must) and not any(f in low for f in forbid):
            return name
    return None


def _joint_addrs(sim, name):
    jid = sim.model.joint_name2id(name)
    return (jid, int(sim.model.jnt_qposadr[jid]), int(sim.model.jnt_dofadr[jid]))


# ── T1 — molla del latch ─────────────────────────────────────────────────────
def measure_latch_spring(start_qpos=1.4, max_steps=300, seed=DEFAULT_PHYSICS_SEED):
    np.random.seed(seed)                        # riproducibilità del placement
    env = _make_rs_door(); env.reset(); sim = env.sim
    name = _find_joint(sim, ["latch"]) or "Door_latch_joint"
    jid, addr, dof = _joint_addrs(sim, name)
    stiffness = float(sim.model.jnt_stiffness[jid]); damping = float(sim.model.dof_damping[dof])
    sim.data.qpos[addr] = start_qpos; sim.data.qvel[dof] = 0.0; sim.forward()
    traj, return_steps = [], None
    for step in range(max_steps):
        a = np.zeros(env.action_dim); a[-1] = -1.0     # gripper aperto: non tocca nulla
        env.step(a)
        q = float(sim.data.qpos[addr]); traj.append(q)
        if abs(q) < LATCH_NEUTRAL and return_steps is None:
            return_steps = step + 1
    env.close()
    return dict(joint=name, stiffness=stiffness, damping=damping,
                return_steps=return_steps, trajectory=traj, final=traj[-1])


def test_latch_spring_returns_to_neutral():
    r = measure_latch_spring()
    assert r["return_steps"] is not None, "Il latch non torna sotto soglia in 300 step"
    assert r["return_steps"] <= LATCH_RETURN_MAX_STEPS
    return r


# ── T2 — ritenzione della porta aperta ───────────────────────────────────────
#
# ATTENZIONE ALL'ARTEFATTO (corretto qui). Il test porta il cardine a ~0.36 rad
# scrivendo direttamente `qpos`: così facendo il PANNELLO ruota, e se la posa del
# braccio — o il posizionamento randomizzato della porta — lo lascia nel volume
# spazzato, MuJoCo risolve la compenetrazione RESPINGENDO la porta. La misura
# registrerebbe allora un "richiudersi" che è in realtà il contatto col robot.
# Sintomo osservato: due esecuzioni dello STESSO commit hanno dato deriva 0.0000 rad
# (nessun contatto) e 0.1814 rad (contatto), cioè PASS e FAIL sulla stessa proprietà.
# Rimedio: (1) reset SEEDATO → riproducibile; (2) rilevamento esplicito dei contatti
# robot↔porta; (3) più tentativi con seed diversi per trovare una configurazione
# pulita; (4) se nessuna è pulita la misura è marcata NON VALIDA invece di produrre
# un verdetto falso.
_DOOR_TOKENS = ("door", "latch", "handle", "frame")
_ROBOT_TOKENS = ("robot", "gripper", "finger", "hand", "link", "panda")


def _geom_groups(sim):
    """Insiemi di geom-id appartenenti alla porta e al robot, per nome."""
    door, robot = set(), set()
    try:
        names = list(sim.model.geom_names)
    except Exception:
        return door, robot
    for i, n in enumerate(names):
        if not n:
            continue
        low = str(n).lower()
        if any(t in low for t in _DOOR_TOKENS):
            door.add(i)
        if any(t in low for t in _ROBOT_TOKENS):
            robot.add(i)
    return door, robot


def _robot_door_contact(sim, door_geoms, robot_geoms) -> bool:
    """True se al passo corrente esiste un contatto attivo fra robot e porta."""
    if not door_geoms or not robot_geoms:
        return False
    try:
        for i in range(int(sim.data.ncon)):
            c = sim.data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if (g1 in door_geoms and g2 in robot_geoms) or \
               (g2 in door_geoms and g1 in robot_geoms):
                return True
    except Exception:
        return False
    return False


def measure_door_retention(open_frac=0.9, impulse=-0.3, max_steps=RETENTION_STEPS,
                           seed=DEFAULT_PHYSICS_SEED, max_attempts=RETENTION_MAX_ATTEMPTS):
    last = None
    for attempt in range(max_attempts):
        np.random.seed(seed + attempt)          # riproducibilità del placement
        env = _make_rs_door()
        env.reset()
        sim = env.sim
        door_geoms, robot_geoms = _geom_groups(sim)
        name = _find_joint(sim, ["hinge"], forbid=("latch",)) or "Door_hinge"
        jid, addr, dof = _joint_addrs(sim, name)
        jmin, jmax = (float(x) for x in sim.model.jnt_range[jid])
        stiffness = float(sim.model.jnt_stiffness[jid])
        damping = float(sim.model.dof_damping[dof])
        # bersaglio realistico: cap di apertura del task (0.40 rad) o il limite del giunto
        target = float(jmin + open_frac * min(jmax - jmin, 0.40))

        def _run(vel0):
            sim.data.qpos[addr] = target
            sim.data.qvel[dof] = vel0
            sim.forward()
            touched = _robot_door_contact(sim, door_geoms, robot_geoms)
            series = []
            for step in range(max_steps):
                env.step(np.zeros(env.action_dim))
                touched = touched or _robot_door_contact(sim, door_geoms, robot_geoms)
                series.append((step, float(sim.data.qpos[addr]), float(sim.data.qvel[dof])))
            return series, touched

        free, touched_free = _run(0.0)
        kicked, touched_kick = _run(impulse)
        env.close()

        res = dict(
            joint=name, target=target, stiffness=stiffness, damping=damping,
            series_free=free, series_kicked=kicked,
            drift_free=float(target - min(q for _, q, _ in free)),
            drift_kicked=float(target - min(q for _, q, _ in kicked)),
            final_free=free[-1][1], final_kicked=kicked[-1][1],
            seed_used=seed + attempt, attempts=attempt + 1,
            robot_contact_free=bool(touched_free),
            robot_contact_kicked=bool(touched_kick),
            valid=(not touched_free),
            contact_detection=bool(door_geoms and robot_geoms),
        )
        last = res
        if res["valid"]:
            return res
    return last


def test_door_stays_open():
    r = measure_door_retention()
    assert r["valid"], (
        f"Misura NON VALIDA dopo {r['attempts']} tentativi: il braccio tocca la porta "
        f"quando la si porta a {r['target']:.3f} rad, quindi la deriva osservata "
        f"({r['drift_free']:.4f} rad) è un artefatto di contatto, non dinamica del cardine.")
    assert r["drift_free"] <= RETENTION_MAX_DRIFT, (
        f"La porta lasciata al goal si richiude di {r['drift_free']:.4f} rad "
        f"(> tolleranza {RETENTION_MAX_DRIFT}): il task di apertura non sarebbe ben posto")
    return r


# ── T7 — randomization estesa ────────────────────────────────────────────────
def measure_randomization_ranges(n_resets=200, run_dir=None, curriculum=1.0):
    env, cfg = make_raw_env(curriculum_level=curriculum, run_dir=resolve_run_dir(run_dir))
    dr = env._domain_rand
    base_s = getattr(dr, "base_latch_stiffness", None) or 1.0
    base_d = getattr(dr, "base_hinge_damping", None) or 0.1
    base_m = getattr(dr, "base_door_mass", None) or 1.0
    keys = ["radius", "friction", "latch_ratio", "damp_ratio", "mass_ratio", "goal_angle"]
    out = {k: [] for k in keys}
    for i in range(n_resets):
        np.random.seed(20_000 + i)
        env.reset(seed=20_000 + i)
        out["radius"].append(float(dr.current_handle_radius))
        out["friction"].append(float(dr.current_handle_friction))
        out["latch_ratio"].append(float(dr.current_latch_stiffness / base_s))
        out["damp_ratio"].append(float(dr.current_hinge_damping / base_d))
        out["mass_ratio"].append(float(dr.current_door_mass / base_m))
        out["goal_angle"].append(float(env._goal_angle))
    meta = dict(door_min=float(env._door_min), effective_max=float(env._effective_max),
                goal_frac_min=float(cfg.goal_frac_min), goal_frac_max=float(cfg.goal_frac_max))
    try:
        env.close()
    except Exception:
        pass
    return {k: np.asarray(v, float) for k, v in out.items()}, meta


def _ranges_ok(r):
    return (0.013 <= r["radius"].min() and r["radius"].max() <= 0.029
            and 0.05 <= r["friction"].min() and r["friction"].max() <= 2.0
            and 0.49 <= r["latch_ratio"].min() and r["latch_ratio"].max() <= 2.05
            and 0.29 <= r["damp_ratio"].min() and r["damp_ratio"].max() <= 1.55
            and 0.49 <= r["mass_ratio"].min() and r["mass_ratio"].max() <= 2.05
            and all(r[k].std() > 1e-5 for k in
                    ["radius", "friction", "latch_ratio", "damp_ratio", "mass_ratio"]))


def test_randomization_ranges():
    r, _meta = measure_randomization_ranges()
    assert _ranges_ok(r)
    return r


# ── T8 — campionamento del goal ──────────────────────────────────────────────
def _goal_ok(r, meta):
    span = meta["effective_max"] - meta["door_min"]
    lo = meta["door_min"] + meta["goal_frac_min"] * span - 1e-6
    hi = meta["door_min"] + meta["goal_frac_max"] * span + 1e-6
    g = r["goal_angle"]
    return bool(g.min() >= lo and g.max() <= hi and g.std() > 1e-6)


def test_goal_sampling():
    r, meta = measure_randomization_ranges()
    assert _goal_ok(r, meta), "goal_angle fuori dal range dichiarato o degenere"
    return r


# ── runner ───────────────────────────────────────────────────────────────────
def run(make_plots=True, run_dir=None):
    print("=" * 76)
    print("PHYSICS UNIT TESTS — APERTURA v2 (deterministici, senza modello)")
    print("=" * 76)
    out = {}; checks = []

    try:
        r1 = measure_latch_spring()
        p1 = (r1["return_steps"] is not None) and (r1["return_steps"] <= LATCH_RETURN_MAX_STEPS)
        out["latch_spring"] = r1
        print(f"  T1 molla latch    : ritorno in {r1['return_steps']} step "
              f"(giunto={r1['joint']}, k={r1['stiffness']:.3f}, c={r1['damping']:.3f}) "
              f"→ {'PASS' if p1 else 'FAIL'}")
        checks.append(("T1 molla del latch", "PASS" if p1 else "FAIL"))
    except Exception as e:
        print(f"  T1 molla latch    : ERROR {e}"); checks.append(("T1 molla del latch", "FAIL"))

    try:
        r2 = measure_door_retention()
        out["door_retention"] = r2
        if not r2["valid"]:
            p2 = "NON VALIDO"
            print(f"  T2 ritenzione     : NON VALIDO — contatto braccio↔porta in tutti i "
                  f"{r2['attempts']} tentativi; la deriva misurata "
                  f"({r2['drift_free']:.4f} rad) è un artefatto di contatto")
        else:
            p2 = "PASS" if r2["drift_free"] <= RETENTION_MAX_DRIFT else "FAIL"
            print(f"  T2 ritenzione     : deriva libera={r2['drift_free']:.4f} rad "
                  f"(tol {RETENTION_MAX_DRIFT}), con impulso={r2['drift_kicked']:.4f} rad "
                  f"[seed {r2['seed_used']}, {r2['attempts']} tentativi] → {p2}")
        checks.append(("T2 ritenzione porta aperta", p2))
    except Exception as e:
        print(f"  T2 ritenzione     : ERROR {e}"); checks.append(("T2 ritenzione porta aperta", "FAIL"))

    try:
        r7, meta = measure_randomization_ranges(run_dir=run_dir)
        p7 = _ranges_ok(r7)
        p8 = _goal_ok(r7, meta)
        out["randomization"] = {k: dict(min=float(v.min()), max=float(v.max()),
                                        mean=float(v.mean()), std=float(v.std()),
                                        values=v.tolist()) for k, v in r7.items()}
        out["randomization_meta"] = meta
        print(f"  T7 domain rand    : raggio∈[{r7['radius'].min():.4f},{r7['radius'].max():.4f}] "
              f"frizione∈[{r7['friction'].min():.3f},{r7['friction'].max():.3f}] "
              f"latch×∈[{r7['latch_ratio'].min():.2f},{r7['latch_ratio'].max():.2f}] "
              f"damp×∈[{r7['damp_ratio'].min():.2f},{r7['damp_ratio'].max():.2f}] "
              f"massa×∈[{r7['mass_ratio'].min():.2f},{r7['mass_ratio'].max():.2f}] "
              f"→ {'PASS' if p7 else 'FAIL'}")
        print(f"  T8 goal sampling  : goal∈[{r7['goal_angle'].min():.4f},{r7['goal_angle'].max():.4f}] "
              f"atteso⊆[{meta['door_min']+meta['goal_frac_min']*(meta['effective_max']-meta['door_min']):.4f},"
              f"{meta['door_min']+meta['goal_frac_max']*(meta['effective_max']-meta['door_min']):.4f}] "
              f"→ {'PASS' if p8 else 'FAIL'}")
        checks.append(("T7 domain randomization", "PASS" if p7 else "FAIL"))
        checks.append(("T8 campionamento del goal", "PASS" if p8 else "FAIL"))
    except Exception as e:
        print(f"  T7/T8             : ERROR {e}")
        checks.append(("T7 domain randomization", "FAIL"))
        checks.append(("T8 campionamento del goal", "FAIL"))

    n_pass = sum(1 for _, p in checks if p == "PASS")
    n_inv = sum(1 for _, p in checks if p == "NON VALIDO")
    tail = f" ({n_inv} non valido/i)" if n_inv else ""
    print("-" * 76); print(f"  ESITO: {n_pass}/{len(checks)} PASS{tail}"); print("=" * 76)
    out["summary"] = dict(passed=n_pass, total=len(checks), invalid=n_inv,
                          checks={n: p for n, p in checks})

    outdir = results_dir("physics")
    with open(os.path.join(outdir, "physics_results.json"), "w") as f:
        json.dump(out, f, indent=2, default=json_default)

    if make_plots:
        plt = setup_matplotlib()
        if "latch_spring" in out:
            fig, ax = plt.subplots(figsize=(8, 5))
            traj = out["latch_spring"]["trajectory"]; t = np.arange(len(traj)) / 30.0
            ax.plot(t, traj, color="#1f77b4")
            ax.axhline(LATCH_NEUTRAL, ls="--", color="k"); ax.axhline(-LATCH_NEUTRAL, ls="--", color="k")
            ax.set_xlabel("tempo (s)"); ax.set_ylabel("latch_qpos (rad)")
            ax.set_title("T1 — ritorno della molla del latch a neutro")
            fig.tight_layout(); fig.savefig(os.path.join(outdir, "plot_latch_spring.png"), dpi=130); plt.close(fig)
        if "door_retention" in out:
            fig, ax = plt.subplots(figsize=(8, 5))
            r2 = out["door_retention"]
            ax.plot([s for s, _, _ in r2["series_free"]], [q for _, q, _ in r2["series_free"]],
                    color="#2ca02c", label="rilasciata ferma")
            ax.plot([s for s, _, _ in r2["series_kicked"]], [q for _, q, _ in r2["series_kicked"]],
                    color="#d62728", label="con impulso di richiusura")
            ax.axhline(r2["target"], ls="--", color="k", label="goal")
            ax.axhline(r2["target"] - RETENTION_MAX_DRIFT, ls=":", color="k", label="tolleranza")
            ax.set_xlabel("step"); ax.set_ylabel("door_angle (rad)")
            ax.set_title("T2 — la porta aperta resta aperta?"); ax.legend()
            fig.tight_layout(); fig.savefig(os.path.join(outdir, "plot_door_retention.png"), dpi=130); plt.close(fig)
        if "randomization" in out:
            keys = ["friction", "radius", "latch_ratio", "damp_ratio", "mass_ratio", "goal_angle"]
            titles = ["frizione", "raggio (m)", "rigidità latch ×base",
                      "smorzamento cerniera ×base", "massa porta ×base", "goal_angle (rad)"]
            fig, axes = plt.subplots(1, 6, figsize=(23, 3.6))
            for ax, k, ttl in zip(axes, keys, titles):
                safe_hist(ax, out["randomization"][k]["values"], 25, color="#2ca02c")
                ax.set_title(f"{'T8' if k == 'goal_angle' else 'T7'} — {ttl}", fontsize=10)
            fig.tight_layout(); fig.savefig(os.path.join(outdir, "plot_randomization.png"), dpi=130); plt.close(fig)
        print(f"  Grafici in {outdir}")
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Physics unit tests — apertura v2")
    ap.add_argument("--run-dir", type=str, default=None)
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()
    run(make_plots=not args.no_plots, run_dir=args.run_dir)
