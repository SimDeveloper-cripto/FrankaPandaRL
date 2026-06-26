#!/usr/bin/env python3
# scratch/test_close_task_v2/physics_unit_tests.py

"""
physics_unit_tests — Test di proprietà FISICHE dell'ambiente v2 (deterministici).

T1 (molla del latch) e T2 (smorzamento cerniera) come nella suite v1 — proprietà
dell'ambiente base, indipendenti dal modello e dal curriculum.

T7 è ESTESO alla domain randomization v2 (domain_rand_v2 §3.4): oltre a raggio e
frizione, verifica che rigidità del latch, smorzamento della cerniera e massa della
porta restino nei range dichiarati (Tobin et al. 2017; Zhao et al. 2020). È rilevante
in v2 perché la FSM adatta la soglia di HOLD alla rigidità del latch (fsm_v2 §3.1): se
la randomizzazione uscisse dal range, la soglia adattiva sarebbe mal calibrata.

Compatibile con `pytest`. Output in results/physics/.
"""

from __future__ import annotations

import os
import json
import numpy as np

from _common import (find_repo_root, make_raw_env, results_dir, setup_matplotlib, json_default)

find_repo_root()
import robosuite as suite  # noqa: E402

LATCH_NEUTRAL = 0.15
LATCH_RETURN_MAX_STEPS = 90
BOUNCE_VEL_MAX = 0.5


def _make_rs_door():
    return suite.make("Door", robots="Panda", has_renderer=False,
                      has_offscreen_renderer=False, use_camera_obs=False,
                      reward_shaping=True, control_freq=30)


# ── T1 — molla del latch ─────────────────────────────────────────────────────
def measure_latch_spring(start_qpos=1.2, max_steps=300):
    env = _make_rs_door(); env.reset(); sim = env.sim
    jid = sim.model.joint_name2id("Door_latch_joint")
    dof = sim.model.jnt_dofadr[jid]; addr = sim.model.jnt_qposadr[jid]
    stiffness = float(sim.model.jnt_stiffness[jid]); damping = float(sim.model.dof_damping[dof])
    sim.data.qpos[addr] = start_qpos; sim.data.qvel[dof] = 0.0; sim.forward()
    traj, return_steps = [], None
    for step in range(max_steps):
        a = np.zeros(env.action_dim); a[-1] = -1.0
        env.step(a)
        q = float(sim.data.qpos[addr]); traj.append(q)
        if abs(q) < LATCH_NEUTRAL and return_steps is None:
            return_steps = step + 1
    env.close()
    return dict(stiffness=stiffness, damping=damping, return_steps=return_steps,
                trajectory=traj, final=traj[-1])


def test_latch_spring_returns_to_neutral():
    r = measure_latch_spring()
    assert r["return_steps"] is not None, "Latch non torna sotto la soglia in 300 step"
    assert r["return_steps"] <= LATCH_RETURN_MAX_STEPS
    return r


# ── T2 — bounce della cerniera ───────────────────────────────────────────────
def measure_hinge_bounce(impact_vel=-0.5, max_steps=60):
    env = _make_rs_door(); env.reset(); sim = env.sim
    jid = sim.model.joint_name2id("Door_hinge")
    dof = sim.model.jnt_dofadr[jid]; addr = sim.model.jnt_qposadr[jid]
    stiffness = float(sim.model.jnt_stiffness[jid]); damping = float(sim.model.dof_damping[dof])
    sim.data.qpos[addr] = 0.01; sim.data.qvel[dof] = impact_vel; sim.forward()
    series = []
    for step in range(max_steps):
        env.step(np.zeros(env.action_dim))
        series.append((step, float(sim.data.qpos[addr]), float(sim.data.qvel[dof])))
    env.close()
    pos_vels = [v for _, _, v in series if v > 0]
    return dict(stiffness=stiffness, damping=damping, series=series,
                max_bounce_vel=max(pos_vels) if pos_vels else 0.0)


def test_hinge_bounce_bounded():
    r = measure_hinge_bounce()
    assert r["max_bounce_vel"] <= BOUNCE_VEL_MAX
    return r


# ── T7 — randomization v2 estesa ─────────────────────────────────────────────
def measure_randomization_ranges(n_resets=200, run_dir=None, curriculum=1.0):
    from _common import RUN_DIR_C1
    env, _cfg = make_raw_env(curriculum_level=curriculum, run_dir=run_dir or RUN_DIR_C1)
    dr = env._domain_rand
    base_s = getattr(dr, "base_latch_stiffness", None) or 1.0
    base_d = getattr(dr, "base_hinge_damping", None) or 0.1
    base_m = getattr(dr, "base_door_mass", None) or 1.0
    out = {k: [] for k in ["radius", "friction", "latch_ratio", "damp_ratio", "mass_ratio"]}
    for i in range(n_resets):
        np.random.seed(20_000 + i)
        env.reset(seed=20_000 + i)
        out["radius"].append(float(dr.current_handle_radius))
        out["friction"].append(float(dr.current_handle_friction))
        out["latch_ratio"].append(float(dr.current_latch_stiffness / base_s))
        out["damp_ratio"].append(float(dr.current_hinge_damping / base_d))
        out["mass_ratio"].append(float(dr.current_door_mass / base_m))
    return {k: np.asarray(v, float) for k, v in out.items()}


def test_randomization_ranges():
    r = measure_randomization_ranges()
    assert 0.013 <= r["radius"].min() and r["radius"].max() <= 0.029
    assert 0.05 <= r["friction"].min() and r["friction"].max() <= 2.0
    assert 0.49 <= r["latch_ratio"].min() and r["latch_ratio"].max() <= 2.05
    assert 0.29 <= r["damp_ratio"].min() and r["damp_ratio"].max() <= 1.55
    assert 0.49 <= r["mass_ratio"].min() and r["mass_ratio"].max() <= 2.05
    for k in r:
        assert r[k].std() > 1e-5, f"randomization degenere su {k}"
    return r


# ── runner ───────────────────────────────────────────────────────────────────
def run(make_plots=True, run_dir=None):
    print("=" * 72)
    print("PHYSICS UNIT TESTS v2 (deterministici, no modello)")
    print("=" * 72)
    out = {}; checks = []

    try:
        r1 = measure_latch_spring()
        p1 = (r1["return_steps"] is not None) and (r1["return_steps"] <= LATCH_RETURN_MAX_STEPS)
        out["latch_spring"] = r1
        print(f"  T1 latch spring : return_steps={r1['return_steps']} "
              f"(stiff={r1['stiffness']:.3f}, damp={r1['damping']:.3f}) → {'PASS' if p1 else 'FAIL'}")
        checks.append(("T1 latch spring", p1))
    except Exception as e:
        print(f"  T1 latch spring : ERROR {e}"); checks.append(("T1 latch spring", False))

    try:
        r2 = measure_hinge_bounce()
        p2 = r2["max_bounce_vel"] <= BOUNCE_VEL_MAX
        out["hinge_bounce"] = r2
        print(f"  T2 hinge bounce : max_bounce_vel={r2['max_bounce_vel']:.4f} rad/s "
              f"(damp={r2['damping']:.3f}) → {'PASS' if p2 else 'FAIL'}")
        checks.append(("T2 hinge bounce", p2))
    except Exception as e:
        print(f"  T2 hinge bounce : ERROR {e}"); checks.append(("T2 hinge bounce", False))

    try:
        r7 = measure_randomization_ranges(run_dir=run_dir)
        p7 = (0.013 <= r7["radius"].min() and r7["radius"].max() <= 0.029
              and 0.05 <= r7["friction"].min() and r7["friction"].max() <= 2.0
              and 0.49 <= r7["latch_ratio"].min() and r7["latch_ratio"].max() <= 2.05
              and 0.29 <= r7["damp_ratio"].min() and r7["damp_ratio"].max() <= 1.55
              and 0.49 <= r7["mass_ratio"].min() and r7["mass_ratio"].max() <= 2.05
              and all(r7[k].std() > 1e-5 for k in r7))
        out["randomization"] = {k: dict(min=float(v.min()), max=float(v.max()),
                                        mean=float(v.mean()), std=float(v.std()),
                                        values=v.tolist()) for k, v in r7.items()}
        print(f"  T7 domain rand  : radius∈[{r7['radius'].min():.4f},{r7['radius'].max():.4f}] "
              f"fric∈[{r7['friction'].min():.3f},{r7['friction'].max():.3f}] "
              f"latch×∈[{r7['latch_ratio'].min():.2f},{r7['latch_ratio'].max():.2f}] "
              f"damp×∈[{r7['damp_ratio'].min():.2f},{r7['damp_ratio'].max():.2f}] "
              f"mass×∈[{r7['mass_ratio'].min():.2f},{r7['mass_ratio'].max():.2f}] "
              f"→ {'PASS' if p7 else 'FAIL'}")
        checks.append(("T7 domain randomization", p7))
    except Exception as e:
        print(f"  T7 domain rand  : ERROR {e}"); checks.append(("T7 domain randomization", False))

    n_pass = sum(p for _, p in checks)
    print("-" * 72); print(f"  ESITO: {n_pass}/{len(checks)} PASS"); print("=" * 72)
    out["summary"] = dict(passed=n_pass, total=len(checks),
                          checks={n: bool(p) for n, p in checks})

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
        if "hinge_bounce" in out:
            fig, ax = plt.subplots(figsize=(8, 5))
            series = out["hinge_bounce"]["series"]
            ax.plot([s for s, _, _ in series], [v for _, _, v in series], color="#d62728")
            ax.axhline(0, color="k", lw=0.6); ax.set_xlabel("step"); ax.set_ylabel("door_qvel (rad/s)")
            ax.set_title("T2 — velocità cerniera dopo l'urto di chiusura")
            fig.tight_layout(); fig.savefig(os.path.join(outdir, "plot_hinge_bounce.png"), dpi=130); plt.close(fig)
        if "randomization" in out:
            keys = ["friction", "radius", "latch_ratio", "damp_ratio", "mass_ratio"]
            titles = ["frizione", "raggio (m)", "rigidità latch ×base",
                      "smorzamento cerniera ×base", "massa porta ×base"]
            fig, axes = plt.subplots(1, 5, figsize=(20, 3.6))
            for ax, k, ttl in zip(axes, keys, titles):
                ax.hist(out["randomization"][k]["values"], bins=25, color="#2ca02c")
                ax.set_title(f"T7 — {ttl}")
            fig.tight_layout(); fig.savefig(os.path.join(outdir, "plot_randomization.png"), dpi=130); plt.close(fig)
        print(f"  Grafici in {outdir}")
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description = "Physics unit tests v2")
    ap.add_argument("--run-dir",  type = str, default = None)
    ap.add_argument("--no-plots", action = "store_true")
    args = ap.parse_args()
    run(make_plots = not args.no_plots, run_dir = args.run_dir)