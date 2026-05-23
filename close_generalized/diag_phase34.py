"""
Diagnostic per Phase 3 e Phase 4

Test inclusi:
  T1 — Latch spring: il latch_qpos torna a 0 da solo (con molla stiffness = 1) ?
       E in quanto tempo (step) ?

  T2 — Hinge damping: velocità del bounce dopo contatto con il frame ?

  T3 — hold_act conflict: quanto spesso action_norm > 0.05 durante HOLD
       (ovvero: il robot è davvero fermo o sta ancora spingendo) ?

  T4 — ret_rot: distribuzione di ||action[3:6]|| durante RETREAT
       (quanto conta la penalità -10.0) ?

  T5 — latch_qpos al momento della transizione HOLD → RETREAT
       (il latch è già al neutro o ancora ruotato) ?

  T6 — Bounce check: quante volte door_qvel > 0.1 in Phase 3 ?
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from config.train_close_config import TrainConfig
from close_generalized.env_gen import GeneralizedDoorEnv

# ─────────────────────────────────────────────────────────────────────────────
# Configurazione
# ─────────────────────────────────────────────────────────────────────────────
RUNS_DIR    = "runs/close_gen"
N_EPISODES  = 30
VERBOSE     = True

def find_latest_model(runs_dir):
    best = os.path.join(runs_dir, "best_model.zip")
    if os.path.exists(best):
        return best
    candidates = sorted(
        [f for f in os.listdir(runs_dir) if f.endswith(".zip")],
        key=lambda f: os.path.getmtime(os.path.join(runs_dir, f))
    )
    if candidates:
        return os.path.join(runs_dir, candidates[-1])
    raise FileNotFoundError(f"Nessun modello trovato in {runs_dir}")

def make_env():
    cfg = TrainConfig()
    env = GeneralizedDoorEnv(cfg)

    env.curriculum_level = 1
    return env

# ─────────────────────────────────────────────────────────────────────────────
# T1 — Latch spring test: rilascia il gripper da latch_qpos noto, misura ritorno
# ─────────────────────────────────────────────────────────────────────────────
def test_latch_spring():
    print("\n" + "="*70)
    print("T1 — LATCH SPRING: il latch_qpos torna a 0 senza gripper ?")
    print("="*70)

    import robosuite as suite
    env_rs = suite.make(
        'Door',
        robots='Panda',
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=30
    )
    obs = env_rs.reset()
    sim = env_rs.sim

    # Trova il DOF del latch joint
    latch_jnt_id    = sim.model.joint_name2id("Door_latch_joint")
    latch_dof       = sim.model.jnt_dofadr[latch_jnt_id]
    latch_qpos_addr = sim.model.jnt_qposadr[latch_jnt_id]

    # Leggi fisica del giunto
    stiffness = sim.model.jnt_stiffness[latch_jnt_id]
    damping   = sim.model.dof_damping[latch_dof]
    springref = sim.model.qpos_spring[latch_qpos_addr]

    print(f"  Latch joint: stiffness={stiffness:.4f}, damping={damping:.4f}, springref={springref:.4f}")
    print(f"  Con stiffness={stiffness} la molla tira verso springref={springref}")

    # Forza latch_qpos a 1.2 rad (valore tipico del failure mode B)
    sim.data.qpos[latch_qpos_addr] = 1.2
    sim.data.qvel[latch_dof]       = 0.0
    sim.forward()

    # Lascia girare la simulazione senza azioni del robot (gripper aperto)
    # per vedere se il latch torna autonomamente a 0
    return_steps = None
    trajectory   = []
    for step in range(300):
        action          = np.zeros(env_rs.action_dim)
        action[-1]      = -1.0
        obs, _, done, _ = env_rs.step(action)
        qpos            = sim.data.qpos[latch_qpos_addr]
        trajectory.append(float(qpos))
        if abs(qpos) < 0.1 and return_steps is None:
            return_steps = step + 1
        if done:
            break

    env_rs.close()

    if return_steps:
        print(f"  ✅ Latch torna a <0.1 rad in {return_steps} step ({return_steps/30:.2f}s)")
    else:
        print(f"  ❌ Latch NON torna a <0.1 rad in 300 step!")
        print(f"     Valore finale: {trajectory[-1]:.4f} rad")
        print(f"     Min raggiunto: {min(trajectory):.4f} rad")

    print(f"  Traiettoria (step 0,10,20,50,100,150,200): "
          f"{[f'{trajectory[min(i,len(trajectory)-1)]:.3f}' for i in [0, 10, 20, 50, 100, 150, 200]]}")

    return return_steps, trajectory


# ─────────────────────────────────────────────────────────────────────────────
# T2 — Hinge damping: misura velocità bounce dopo chiusura porta
# ─────────────────────────────────────────────────────────────────────────────
def test_hinge_damping():
    print("\n" + "="*70)
    print("T2 — HINGE DAMPING: velocità bounce del giunto porta")
    print("="*70)

    import robosuite as suite
    env_rs = suite.make(
        'Door',
        robots                 = 'Panda',
        has_renderer           = False,
        has_offscreen_renderer = False,
        use_camera_obs         = False,
        reward_shaping         = True,
        control_freq           = 30
    )
    obs = env_rs.reset()
    sim = env_rs.sim

    hinge_id        = sim.model.joint_name2id("Door_hinge")
    hinge_dof       = sim.model.jnt_dofadr[hinge_id]
    hinge_qpos_addr = sim.model.jnt_qposadr[hinge_id]

    stiffness = sim.model.jnt_stiffness[hinge_id]
    damping   = sim.model.dof_damping[hinge_dof]
    print(f"  Hinge joint: stiffness={stiffness:.4f}, damping={damping:.4f}")

    # Simula una porta che arriva a 0.00 con velocità -0.5 rad/s (chiusura rapida)
    sim.data.qpos[hinge_qpos_addr] = 0.01
    sim.data.qvel[hinge_dof]       = -0.5  # velocità verso chiusura
    sim.forward()

    velocities = []
    for step in range(60):
        action          = np.zeros(env_rs.action_dim)
        obs, _, done, _ = env_rs.step(action)

        vel = float(sim.data.qvel[hinge_dof])
        pos = float(sim.data.qpos[hinge_qpos_addr])
        velocities.append((step, pos, vel))
        if done:
            break

    env_rs.close()

    print(f"  Step | door_qpos | door_qvel | hold_veldamp (-15*|vel|)")
    for step, pos, vel in velocities[:20]:
        veldamp = -15.0 * abs(vel)
        marker  = " ← BOUNCE" if vel > 0.05 else ""
        print(f"  {step:4d} | {pos:+.4f}  | {vel:+.4f}  | {veldamp:+.3f}{marker}")

    max_bounce_vel = max(v for _, _, v in velocities if v > 0)
    print(f"\n  Velocità max di bounce: {max_bounce_vel:.4f} rad/s")
    print(f"  Penalità hold_veldamp massima: {-15.0*max_bounce_vel:.4f}")

    return velocities


# ─────────────────────────────────────────────────────────────────────────────
# T3-T6 — Test con il modello addestrato
# ─────────────────────────────────────────────────────────────────────────────
def test_with_model():
    print("\n" + "="*70)
    print("T3-T6 — TEST CON MODELLO (episodi reali)")
    print("="*70)

    model_path = find_latest_model(RUNS_DIR)
    print(f"  Modello: {model_path}")

    vn_path = os.path.join(RUNS_DIR, "vecnorm.pkl")
    if not os.path.exists(vn_path):
        vn_path = os.path.join(RUNS_DIR, "vecnormalize.pkl")

    raw_env = DummyVecEnv([make_env])
    if os.path.exists(vn_path):
        vec_env             = VecNormalize.load(vn_path, raw_env)
        vec_env.training    = False
        vec_env.norm_reward = False
    else:
        vec_env = raw_env

    model = SAC.load(model_path, env=vec_env)

    # Contatori T3-T6
    t3_action_norms_hold   = []    # hold: action_norm durante HOLD attivo
    t4_wrist_rots_retreat  = []    # retreat: ||action[3:6]||
    t5_latch_at_transition = []    # latch_qpos al momento della transizione
    t6_bounce_events       = []    # door_qvel durante Phase 3

    episodes_completed   = 0
    episodes_success     = 0
    episodes_latch_fail  = 0    # failure B: latch non neutro al retreat

    obs = vec_env.reset()

    for ep in range(N_EPISODES):
        done               = False
        step               = 0
        ep_hold_norms      = []
        ep_wrist_rots      = []
        ep_bounces         = []
        prev_ready_retreat = False

        while not done:
            action, _                   = model.predict(obs, deterministic=True)
            obs, reward, done_arr, info = vec_env.step(action)
            done                        = done_arr[0]
            step += 1

            if hasattr(vec_env, 'envs'):
                inner_env = vec_env.envs[0]
            elif hasattr(vec_env, 'venv') and hasattr(vec_env.venv, 'envs'):
                inner_env = vec_env.venv.envs[0]
            else:
                inner_env = vec_env.venv.venv.envs[0]

            sim           = inner_env._rs_env.sim
            grasp         = getattr(inner_env, '_grasp_phase', False)
            latched       = getattr(inner_env, '_success_latched', False)
            ready_retreat = getattr(inner_env, '_ready_to_retreat', False)

            # Leggi latch_qpos
            latch_jnt_id    = sim.model.joint_name2id("Door_latch_joint")
            latch_qpos_addr = sim.model.jnt_qposadr[latch_jnt_id]
            latch_qpos      = float(sim.data.qpos[latch_qpos_addr])

            # Hinge vel
            hinge_id        = sim.model.joint_name2id("Door_hinge")
            hinge_dof       = sim.model.jnt_dofadr[hinge_id]
            hinge_qpos_addr = sim.model.jnt_qposadr[hinge_id]
            door_qvel       = float(sim.data.qvel[hinge_dof])
            door_qpos       = float(sim.data.qpos[hinge_qpos_addr])

            # T3 — action_norm durante HOLD (Phase 3, not ready_retreat)
            if latched and not ready_retreat:
                raw_action = action[0]
                a_norm     = float(np.linalg.norm(raw_action[:-1]))
                ep_hold_norms.append(a_norm)

                # T6 — bounce durante HOLD
                if abs(door_qvel) > 0.05:
                    ep_bounces.append((step, door_qpos, door_qvel))

            # T4 — wrist rotation durante RETREAT
            if latched and ready_retreat:
                raw_action = action[0]
                wrist_rot  = float(np.linalg.norm(raw_action[3:6]))
                ep_wrist_rots.append(wrist_rot)

            # T5 — latch_qpos al momento della transizione HOLD→RETREAT
            if latched and ready_retreat and not prev_ready_retreat:
                t5_latch_at_transition.append(latch_qpos)
                if abs(latch_qpos) > 0.15:
                    episodes_latch_fail += 1
                    if VERBOSE:
                        print(f"  Ep {ep+1:3d}: TRANSIZIONE RETREAT con latch_qpos={latch_qpos:.3f} rad ← FAILURE MODE B")
                else:
                    if VERBOSE:
                        print(f"  Ep {ep+1:3d}: TRANSIZIONE RETREAT con latch_qpos={latch_qpos:.3f} rad ✅")

            prev_ready_retreat = ready_retreat

        episodes_completed += 1
        ep_info = info[0] if info else {}
        success = ep_info.get('is_success', False)
        if success:
            episodes_success += 1

        if ep_hold_norms:
            t3_action_norms_hold.extend(ep_hold_norms)
        if ep_wrist_rots:
            t4_wrist_rots_retreat.extend(ep_wrist_rots)
        if ep_bounces:
            t6_bounce_events.extend(ep_bounces)

    vec_env.close()

    # ── Risultati T3 ──────────────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print(f"T3 — ACTION NORM durante HOLD (Phase 3):")
    if t3_action_norms_hold:
        arr        = np.array(t3_action_norms_hold)
        frac_small = (arr < 0.05).mean()
        frac_large = (arr > 0.3).mean()
        print(f"  Steps analizzati: {len(arr)}")
        print(f"  Media: {arr.mean():.4f} | Std: {arr.std():.4f} | Max: {arr.max():.4f}")
        print(f"  % action_norm < 0.05 (reward +1): {frac_small*100:.1f}%")
        print(f"  % action_norm > 0.30 (forte penalità): {frac_large*100:.1f}%")
        print(f"  Penalità media hold_act: {(-5.0 * arr).mean():.4f}/step")
        if frac_small > 0.5:
            print(f"  → Il robot è prevalentemente fermo (+1 per step)")
        else:
            print(f"  → Il robot è prevalentemente in movimento → hold_act penalizza!")
    else:
        print("  Nessun dato (nessun episodio ha raggiunto Phase 3)")

    # ── Risultati T4 ──────────────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print(f"T4 — WRIST ROTATION durante RETREAT (Phase 4):")
    if t4_wrist_rots_retreat:
        arr = np.array(t4_wrist_rots_retreat)
        print(f"  Steps analizzati: {len(arr)}")
        print(f"  Media: {arr.mean():.4f} | Std: {arr.std():.4f} | Max: {arr.max():.4f}")
        print(f"  Penalità media ret_rot (-10): {(-10.0 * arr).mean():.4f}/step")
        print(f"  Penalità media ret_rot (-3): {(-3.0 * arr).mean():.4f}/step  ← proposta Fix 4")
        pct_high = (arr > 0.1).mean()
        print(f"  % step con wrist_rot > 0.1: {pct_high*100:.1f}%")
    else:
        print("  Nessun dato (nessun episodio ha raggiunto Phase 4)")

    # ── Risultati T5 ──────────────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print(f"T5 — LATCH_QPOS AL MOMENTO DELLA TRANSIZIONE HOLD→RETREAT:")
    if t5_latch_at_transition:
        arr = np.array(t5_latch_at_transition)
        print(f"  Transizioni analizzate: {len(arr)}")
        print(f"  Media latch_qpos: {arr.mean():.4f} rad")
        print(f"  Std:  {arr.std():.4f} rad | Max: {arr.max():.4f} rad | Min: {arr.min():.4f} rad")
        frac_bad = (arr > 0.15).mean()
        print(f"  % transizioni con latch_qpos > 0.15 (FAILURE MODE B): {frac_bad*100:.1f}%")
        print(f"  % transizioni con latch_qpos < 0.15 (latch OK): {(1-frac_bad)*100:.1f}%")
        print(f"  Distribuzione: {np.histogram(arr, bins=[0, 0.1, 0.2, 0.5, 1.0, 2.0])[0]}")
        if episodes_latch_fail > 0:
            print(f"  ⚠️  {episodes_latch_fail}/{len(arr)} transizioni con latch non al neutro")
    else:
        print("  Nessuna transizione HOLD→RETREAT osservata")

    # ── Risultati T6 ──────────────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print(f"T6 — BOUNCE EVENTS in Phase 3 (door_qvel > 0.05 rad/s):")
    if t6_bounce_events:
        vels = [abs(v) for _, _, v in t6_bounce_events]
        print(f"  Bounce events: {len(t6_bounce_events)}")
        print(f"  Velocità media bounce: {np.mean(vels):.4f} rad/s")
        print(f"  Velocità max bounce:   {np.max(vels):.4f} rad/s")
        print(f"  Penalità veldamp media (weight=-15): {np.mean([-15*v for v in vels]):.4f}/step")
        severe = sum(1 for v in vels if v > 0.15)
        print(f"  Bounce severi (>0.15 rad/s): {severe}/{len(vels)}")
    else:
        print("  Nessun bounce event osservato! Porta stabile in Phase 3.")

    # ── Sommario finale ───────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"SOMMARIO EPISODI:")
    print(f"  Totale:  {episodes_completed}")
    print(f"  Success: {episodes_success} ({100 * episodes_success / max(1, episodes_completed):.1f}%)")
    print(f"  Failure Mode B (latch non neutro): {episodes_latch_fail}")
    print(f"{'='*70}")

    return {
        "success_rate"               : episodes_success / max(1, episodes_completed),
        "latch_fail_rate"            : episodes_latch_fail / max(1, len(t5_latch_at_transition)),
        "t3_mean_action_norm"        : np.mean(t3_action_norms_hold) if t3_action_norms_hold else None,
        "t4_mean_wrist_rot"          : np.mean(t4_wrist_rots_retreat) if t4_wrist_rots_retreat else None,
        "t5_mean_latch_at_transition": np.mean(t5_latch_at_transition) if t5_latch_at_transition else None,
        "t6_bounce_count"            : len(t6_bounce_events),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║       DIAGNOSTIC Phase 3 & 4 — FrankaPandaRL                        ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")

    # T1: Fisica della molla del latch
    latch_return_steps, latch_traj = test_latch_spring()

    # T2: Fisica del bounce del giunto porta
    bounce_data = test_hinge_damping()

    # T3-T6: Test con il modello addestrato
    results = test_with_model()

    print("\n╔══════════════════════════════════════════════════════════════════════╗")
    print("║                    CONCLUSIONI PER I FIX                            ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")

    # F1: Latch spring
    if latch_return_steps and latch_return_steps < 90:
        print(f"  F1 (latch_qpos < 0.15 su transizione): ✅ SICURO")
        print(f"     La molla riporta il latch a neutro in {latch_return_steps} step ({latch_return_steps/30:.1f}s)")
        print(f"     → La condizione non può bloccare il robot all'infinito")
    elif latch_return_steps:
        print(f"  F1 (latch_qpos < 0.15 su transizione): ⚠️  LENTO")
        print(f"     Latch torna a neutro in {latch_return_steps} step ({latch_return_steps/30:.1f}s)")
        print(f"     → Potrebbe prolungare Phase 3 di {latch_return_steps} step (≈ {latch_return_steps/30:.1f}s)")
    else:
        print(f"  F1 (latch_qpos < 0.15 su transizione): ❌ NON SICURO")
        print(f"     La molla NON riporta il latch a neutro → Fix F1 bloccherebbe il training!")

    # F2: hold_act
    if results["t3_mean_action_norm"] is not None:
        norm = results["t3_mean_action_norm"]
        if norm < 0.05:
            print(f"  F2 (hold_act ridotto): ✅ Poco impatto (action_norm medio = {norm:.3f})")
        else:
            print(f"  F2 (hold_act: -5 → -2): ✅ UTILE (action_norm medio = {norm:.3f} → penalità media da {-5*norm:.2f} a {-2*norm:.2f})")

    # F4: ret_rot
    if results["t4_mean_wrist_rot"] is not None:
        wr = results["t4_mean_wrist_rot"]
        print(f"  F4 (ret_rot: -10 → -3): ✅ UTILE")
        print(f"     Penalità media da {-10*wr:.2f} → {-3*wr:.2f} per step")

    # F1 rate
    if results["t5_mean_latch_at_transition"] is not None:
        lq = results["t5_mean_latch_at_transition"]
        lr = results["latch_fail_rate"]
        print(f"  T5: latch_qpos medio alla transizione = {lq:.3f} rad ({lr*100:.1f}% sopra 0.15)")