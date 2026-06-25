#!/usr/bin/env python3
# scratch/test_close_task_v1/physics_unit_tests.py
"""
physics_unit_tests — Test di proprietà FISICHE dell'ambiente (task di chiusura v1).

Rifattorizza T1 (latch spring) e T2 (hinge damping) di `diag_phase34.py` in
*property test* deterministici, con tolleranze esplicite ed esito PASS/FAIL, più
due nuovi controlli sulla domain randomization (Tobin et al. 2017; Mehta et al. 2020).

Sono test sull'AMBIENTE, non sulla policy: non richiedono il modello addestrato e sono
deterministici (Henderson et al. 2018 — un test fisico deve dare sempre lo stesso esito).
Compatibili con `pytest` (funzioni `test_*` con `assert`) e lanciabili standalone.

Output (in results/physics/): physics_results.json, plot_latch_spring.png,
plot_hinge_bounce.png, plot_friction_hist.png
"""

from __future__ import annotations

import os
import json
import numpy as np

from _common import (find_repo_root, make_cfg, GeneralizedDoorEnv,
                     results_dir, setup_matplotlib, json_default)

find_repo_root()
import robosuite as suite  # noqa: E402

# Tolleranze (soglie usate anche dalla FSM dell'env)
LATCH_NEUTRAL = 0.15          # |latch_qpos| sotto cui il latch è "neutro" (env)
LATCH_RETURN_MAX_STEPS = 90   # 3 s a 30 Hz: oltre, la condizione potrebbe bloccare la FSM
BOUNCE_VEL_MAX = 0.5          # rad/s: bounce d'urto plausibile da contrastare


def _make_rs_door():
    return suite.make("Door", robots="Panda", has_renderer=False,
                      has_offscreen_renderer=False, use_camera_obs=False,
                      reward_shaping=True, control_freq=30)


# ─────────────────────────────────────────────────────────────────────────────
# T1 — La molla del latch riporta il giunto a neutro da sola?
# ─────────────────────────────────────────────────────────────────────────────
def measure_latch_spring(start_qpos: float = 1.2, max_steps: int = 300):
    env = _make_rs_door(); env.reset(); sim = env.sim
    jid = sim.model.joint_name2id("Door_latch_joint")
    dof = sim.model.jnt_dofadr[jid]
    addr = sim.model.jnt_qposadr[jid]
    stiffness = float(sim.model.jnt_stiffness[jid])
    damping = float(sim.model.dof_damping[dof])

    sim.data.qpos[addr] = start_qpos
    sim.data.qvel[dof] = 0.0
    sim.forward()

    traj, return_steps = [], None
    for step in range(max_steps):
        a = np.zeros(env.action_dim); a[-1] = -1.0  # gripper aperto, nessun moto del braccio
        env.step(a)
        q = float(sim.data.qpos[addr]); traj.append(q)
        if abs(q) < LATCH_NEUTRAL and return_steps is None:
            return_steps = step + 1
    env.close()
    return dict(stiffness=stiffness, damping=damping, return_steps=return_steps,
                trajectory=traj, final=traj[-1])


def test_latch_spring_returns_to_neutral():
    r = measure_latch_spring()
    assert r["return_steps"] is not None, (
        f"Latch NON torna sotto {LATCH_NEUTRAL} rad in 300 step (finale {r['final']:.3f}). "
        "La condizione 'latch neutro alla transizione' bloccherebbe la FSM all'infinito."
    )
    assert r["return_steps"] <= LATCH_RETURN_MAX_STEPS, (
        f"Latch torna a neutro in {r['return_steps']} step (>{LATCH_RETURN_MAX_STEPS}): "
        "rischio di prolungare HOLD oltre il budget."
    )
    return r


# ─────────────────────────────────────────────────────────────────────────────
# T2 — Damping della cerniera: il bounce dopo la chiusura è contenuto?
# ─────────────────────────────────────────────────────────────────────────────
def measure_hinge_bounce(impact_vel: float = -0.5, max_steps: int = 60):
    env = _make_rs_door(); env.reset(); sim = env.sim
    jid = sim.model.joint_name2id("Door_hinge")
    dof = sim.model.jnt_dofadr[jid]
    addr = sim.model.jnt_qposadr[jid]
    stiffness = float(sim.model.jnt_stiffness[jid])
    damping = float(sim.model.dof_damping[dof])

    sim.data.qpos[addr] = 0.01
    sim.data.qvel[dof] = impact_vel
    sim.forward()

    series = []
    for step in range(max_steps):
        env.step(np.zeros(env.action_dim))
        series.append((step, float(sim.data.qpos[addr]), float(sim.data.qvel[dof])))
    env.close()
    pos_vels = [v for _, _, v in series if v > 0]
    max_bounce = max(pos_vels) if pos_vels else 0.0
    return dict(stiffness=stiffness, damping=damping, series=series, max_bounce_vel=max_bounce)


def test_hinge_bounce_bounded():
    r = measure_hinge_bounce()
    assert r["max_bounce_vel"] <= BOUNCE_VEL_MAX, (
        f"Bounce della cerniera {r['max_bounce_vel']:.3f} rad/s > {BOUNCE_VEL_MAX}: "
        "il termine hold_veldamp potrebbe non bastare a stabilizzare la porta."
    )
    return r


# ─────────────────────────────────────────────────────────────────────────────
# T7 — Domain randomization: i parametri campionati restano nei range dichiarati?
#      (Tobin 2017: la generalizzazione vive solo se il range è quello atteso.)
# ─────────────────────────────────────────────────────────────────────────────
def measure_randomization_ranges(n_resets: int = 200, run_dir: str = "runs/close_gen"):
    cfg = make_cfg(run_dir=run_dir)
    env = GeneralizedDoorEnv(cfg); env.set_curriculum_level(1.0)
    radii, frictions, xs = [], [], []
    for i in range(n_resets):
        np.random.seed(20_000 + i)
        env.reset()
        radii.append(float(env._current_handle_radius))
        frictions.append(float(env._current_handle_friction))
        xs.append(float(env._rs_env.sim.model.body_pos[env.door_body_id][0]))
    try:
        env.close()
    except Exception:
        pass
    arr = lambda v: np.asarray(v, float)
    return dict(radius=arr(radii), friction=arr(frictions), door_x=arr(xs))


def test_randomization_ranges():
    r = measure_randomization_ranges()
    # raggio: base 0.02 × U(0.7, 1.4) → [0.014, 0.028]
    assert 0.013 <= r["radius"].min() and r["radius"].max() <= 0.029, \
        f"radius fuori range: [{r['radius'].min():.4f}, {r['radius'].max():.4f}]"
    # frizione: clip(base×U(0.3,1.2), 0.05, 2.0); con base 0.8 → ~[0.24, 0.96]
    assert 0.05 <= r["friction"].min() and r["friction"].max() <= 2.0, \
        f"friction fuori clip: [{r['friction'].min():.4f}, {r['friction'].max():.4f}]"
    # varietà effettiva (non degenere)
    assert r["radius"].std() > 1e-4 and r["friction"].std() > 1e-3, \
        "domain randomization degenere (std troppo bassa)"
    return r


# ─────────────────────────────────────────────────────────────────────────────
# Runner standalone con report + plot
# ─────────────────────────────────────────────────────────────────────────────
def run(make_plots: bool = True, run_dir: str = "runs/close_gen") -> dict:
    print("=" * 72)
    print("PHYSICS UNIT TESTS (deterministici, no modello)")
    print("=" * 72)
    out = {}

    checks = []
    # T1
    try:
        r1 = measure_latch_spring()
        p1 = (r1["return_steps"] is not None) and (r1["return_steps"] <= LATCH_RETURN_MAX_STEPS)
        out["latch_spring"] = {k: v for k, v in r1.items() if k != "trajectory"}
        out["latch_spring"]["trajectory"] = r1["trajectory"]
        print(f"  T1 latch spring : return_steps={r1['return_steps']} "
              f"(stiffness={r1['stiffness']:.3f}, damping={r1['damping']:.3f}) "
              f"→ {'PASS' if p1 else 'FAIL'}")
        checks.append(("T1 latch spring", p1))
    except Exception as e:
        print(f"  T1 latch spring : ERROR {e}"); checks.append(("T1 latch spring", False))

    # T2
    try:
        r2 = measure_hinge_bounce()
        p2 = r2["max_bounce_vel"] <= BOUNCE_VEL_MAX
        out["hinge_bounce"] = {k: v for k, v in r2.items() if k != "series"}
        out["hinge_bounce"]["series"] = r2["series"]
        print(f"  T2 hinge bounce : max_bounce_vel={r2['max_bounce_vel']:.4f} rad/s "
              f"(damping={r2['damping']:.3f}) → {'PASS' if p2 else 'FAIL'}")
        checks.append(("T2 hinge bounce", p2))
    except Exception as e:
        print(f"  T2 hinge bounce : ERROR {e}"); checks.append(("T2 hinge bounce", False))

    # T7
    try:
        r7 = measure_randomization_ranges(run_dir=run_dir)
        p7 = (0.013 <= r7["radius"].min() and r7["radius"].max() <= 0.029
              and 0.05 <= r7["friction"].min() and r7["friction"].max() <= 2.0
              and r7["radius"].std() > 1e-4 and r7["friction"].std() > 1e-3)
        out["randomization"] = {
            k: dict(min=float(v.min()), max=float(v.max()),
                    mean=float(v.mean()), std=float(v.std()),
                    values=v.tolist())
            for k, v in r7.items()
        }
        print(f"  T7 domain rand  : radius∈[{r7['radius'].min():.4f},{r7['radius'].max():.4f}] "
              f"friction∈[{r7['friction'].min():.3f},{r7['friction'].max():.3f}] "
              f"→ {'PASS' if p7 else 'FAIL'}")
        checks.append(("T7 domain randomization", p7))
    except Exception as e:
        print(f"  T7 domain rand  : ERROR {e}"); checks.append(("T7 domain randomization", False))

    n_pass = sum(p for _, p in checks)
    print("-" * 72)
    print(f"  ESITO: {n_pass}/{len(checks)} PASS")
    print("=" * 72)
    out["summary"] = dict(passed=n_pass, total=len(checks),
                          checks={name: bool(p) for name, p in checks})

    outdir = results_dir("physics")
    with open(os.path.join(outdir, "physics_results.json"), "w") as f:
        json.dump(out, f, indent=2, default=json_default)

    if make_plots:
        plt = setup_matplotlib()
        if "latch_spring" in out:
            fig, ax = plt.subplots(figsize=(8, 5))
            traj = out["latch_spring"]["trajectory"]
            t = np.arange(len(traj)) / 30.0
            ax.plot(t, traj, color="#1f77b4")
            ax.axhline(LATCH_NEUTRAL, ls="--", color="k")
            ax.axhline(-LATCH_NEUTRAL, ls="--", color="k")
            ax.set_xlabel("tempo (s)"); ax.set_ylabel("latch_qpos (rad)")
            ax.set_title("T1 — Ritorno della molla del latch a neutro")
            fig.tight_layout(); fig.savefig(os.path.join(outdir, "plot_latch_spring.png"), dpi=130); plt.close(fig)
        if "hinge_bounce" in out:
            fig, ax = plt.subplots(figsize=(8, 5))
            series = out["hinge_bounce"]["series"]
            steps = [s for s, _, _ in series]; vel = [v for _, _, v in series]
            ax.plot(steps, vel, color="#d62728")
            ax.axhline(0, color="k", lw=0.6)
            ax.set_xlabel("step"); ax.set_ylabel("door_qvel (rad/s)")
            ax.set_title("T2 — Velocità della cerniera dopo l'urto di chiusura")
            fig.tight_layout(); fig.savefig(os.path.join(outdir, "plot_hinge_bounce.png"), dpi=130); plt.close(fig)
        if "randomization" in out:
            fig, axes = plt.subplots(1, 2, figsize=(11, 4))
            axes[0].hist(out["randomization"]["friction"]["values"], bins=30, color="#ff7f0e")
            axes[0].set_title("T7 — Frizione maniglia campionata"); axes[0].set_xlabel("friction")
            axes[1].hist(out["randomization"]["radius"]["values"], bins=30, color="#2ca02c")
            axes[1].set_title("T7 — Raggio maniglia campionato"); axes[1].set_xlabel("radius (m)")
            fig.tight_layout(); fig.savefig(os.path.join(outdir, "plot_friction_hist.png"), dpi=130); plt.close(fig)
        print(f"  Grafici in {outdir}")
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Physics unit tests (deterministici)")
    ap.add_argument("--run-dir", type=str, default="runs/close_gen")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()
    run(make_plots=not args.no_plots, run_dir=args.run_dir)
