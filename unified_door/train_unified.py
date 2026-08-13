#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_unified.py — addestramento, play e valutazione. Un file per i due compiti.

Documento di riferimento: tabella_reward_machine_unificata.md §7 «Come si
verifica che funzioni». L'esperimento è A/B a UNA variabile: cambia solo
`--task`, tutto il resto è identico.

    export PROGETTI_ORIGINALI=/percorso/che/contiene/close_generalized_v2

    # addestramento
    python3 train_unified.py --task close --total-steps 1000000
    python3 train_unified.py --task open  --total-steps 1000000

    # play: guarda il robot (su macOS serve mjpython)
    mjpython train_unified.py --task open --play

    # valutazione: i due numeri del §7, da riportare INSIEME
    python3 train_unified.py --task close --eval --episodes 20

§7, avvertenza: il gate bilaterale e la guardia di stallo sono introdotti
INSIEME e non hanno un interruttore separato — è deliberato, perché il primo
senza la seconda peggiora la situazione.
"""
import argparse
import os
import time

import numpy as np

from config_unified import UnifiedConfig


def crea_env(cfg, render_mode=None, seed=None):
    from env_unified import UnifiedDoorEnv
    env = UnifiedDoorEnv(cfg, render_mode=render_mode)
    if seed is not None:
        np.random.seed(seed)          # semina anche la randomizzazione di dominio
    return env


def crea_vecenv(cfg, n_envs, seed):
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from stable_baselines3.common.monitor import Monitor
    venv = DummyVecEnv([(lambda i=i: Monitor(crea_env(cfg, seed=seed + i))) for i in range(n_envs)])
    return VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=10.0)


def crea_modello(cfg, env, tb=None):
    """SAC con gli iperparametri del §6, invariati rispetto ai due progetti."""
    from stable_baselines3 import SAC
    h = cfg.sac
    return SAC(
        "MlpPolicy", env,
        learning_rate=h.learning_rate, buffer_size=h.buffer_size, batch_size=h.batch_size,
        gamma=h.gamma, tau=h.tau, gradient_steps=h.gradient_steps,
        learning_starts=h.learning_starts, ent_coef=h.ent_coef,
        target_entropy=h.target_entropy, target_update_interval=h.target_update_interval,
        use_sde=h.use_sde, policy_kwargs=dict(net_arch=list(h.net_arch)),
        tensorboard_log=tb, verbose=1,
    )


def intestazione(cfg):
    t, h, w = cfg.task, cfg.sac, cfg.w
    print(f"\ncompito: {t.nome}")
    print(f"  θ* = {t.theta_star_frac} della corsa   ·   θ₀ = {t.theta_zero_frac}")
    print(f"  tol = {t.tol} rad   ·   T_hold = {t.t_hold_s} s")
    print(f"  d_ret = {t.d_ret} m · z_ret = {t.z_ret} m · "
          f"normale {'ruotata' if t.orienta_normale else 'diretta'}")
    print(f"  controllori: riporto leva = {t.riporto_leva} · escape = {t.escape}")
    print(f"  SAC invariato (§6): rete {h.net_arch}, lr {h.learning_rate}, γ {h.gamma}, "
          f"target entropy {h.target_entropy}")
    print(f"  budget di progresso: {w.budget_progress} su tutta l'escursione\n")


class DiarioRewardMachine:
    """Rende esplicita la reward machine DURANTE l'addestramento.

    Ogni `ogni` passi stampa, per ciascuna fase FSM: quanti passi vi sono stati
    spesi, il bilancio, e il totale di ogni termine. Serve a diagnosticare un
    addestramento senza doverlo rieseguire: se una fase paga positivo e non si
    chiude mai, si vede qui.
    """

    def __init__(self, ogni=20_000):
        self.ogni = ogni
        self.prossima = ogni
        self.reset_finestra()

    def reset_finestra(self):
        self.per_fase = {}
        self.transizioni = {}
        self.episodi = 0
        self.riusciti = 0

    def osserva(self, infos, rewards):
        for info, r in zip(infos, rewards):
            fase = FASI[min(int(info.get("fsm_phase", 0)), 4)]
            b = self.per_fase.setdefault(fase, {"n": 0, "R": 0.0, "t": {}})
            b["n"] += 1; b["R"] += float(r)
            for k, v in (info.get("reward_terms") or {}).items():
                b["t"][k] = b["t"].get(k, 0.0) + float(v)

    def stampa(self, passi):
        tot = sum(b["n"] for b in self.per_fase.values()) or 1
        print(f"\n  ┌─ reward machine · ultimi {tot} passi (totale {passi}) "
              + "─" * 22)
        print(f"  │ {'fase':<9}{'% passi':>9}{'R/passo':>10}   termini principali")
        for nome in FASI:
            b = self.per_fase.get(nome)
            if not b:
                continue
            top = sorted(b["t"].items(), key=lambda kv: -abs(kv[1]))[:6]
            print(f"  │ {nome:<9}{100*b['n']/tot:>8.1f}%{b['R']/b['n']:>10.2f}   "
                  + " · ".join(f"{k} {v/b['n']:+.2f}" for k, v in top))
        mai = [f for f in FASI[:4] if f not in self.per_fase]
        if mai:
            print(f"  │ MAI RAGGIUNTE: {', '.join(mai)}")
        print("  └" + "─" * 74)
        self.reset_finestra()


def _crea_callback_diario(ogni):
    from stable_baselines3.common.callbacks import BaseCallback

    class Diario(BaseCallback):
        def __init__(self):
            super().__init__()
            self.d = DiarioRewardMachine(ogni)

        def _on_step(self) -> bool:
            self.d.osserva(self.locals.get("infos", []),
                           self.locals.get("rewards", []))
            if self.num_timesteps >= self.d.prossima:
                self.d.stampa(self.num_timesteps)
                self.d.prossima += self.d.ogni
            return True

    return Diario()


# ═════════════════════════════════════════════════════════════════════════
def addestra(cfg, args):
    from stable_baselines3.common.callbacks import CheckpointCallback
    env = crea_vecenv(cfg, cfg.sac.n_envs, args.seed)
    tb = os.path.join(args.run_dir, "tb")
    try:                       # tensorboard è opzionale
        import tensorboard  # noqa: F401
    except ImportError:
        tb = None
    model = crea_modello(cfg, env, tb=tb)
    os.makedirs(args.run_dir, exist_ok=True)
    cb = [CheckpointCallback(save_freq=max(1, 50_000 // cfg.sac.n_envs),
                             save_path=args.run_dir, name_prefix="ckpt"),
          _crea_callback_diario(args.diario_ogni)]
    model.learn(total_timesteps=args.total_steps, callback=cb, progress_bar=False)
    model.save(os.path.join(args.run_dir, "best_model"))
    env.save(os.path.join(args.run_dir, "vecnormalize.pkl"))
    print(f"\nsalvato in {args.run_dir}")


def carica(cfg, args, render_mode=None):
    from stable_baselines3 import SAC
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    env = DummyVecEnv([lambda: crea_env(cfg, render_mode=render_mode, seed=args.seed)])
    vn = os.path.join(args.run_dir, "vecnormalize.pkl")
    if os.path.exists(vn):
        env = VecNormalize.load(vn, env)
        env.training, env.norm_reward = False, False
    mp = os.path.join(args.run_dir, "best_model.zip")
    model = SAC.load(mp, env=env) if os.path.exists(mp) else None
    if model is None:
        print(f"[avviso] nessun modello in {mp}: la politica sarà casuale")
    return env, model


FASI = ("REACH", "MOVE", "HOLD", "RELEASE", "FINE")
CONTROLLORI = ("-", "C0 riporto leva", "C1 escape")


def episodio(env, model, render=False, slow=0.0, verboso=False, hud_ogni=0):
    """Esegue un episodio. Con `verboso` stampa transizioni e reward machine."""
    obs = env.reset()
    tot, passi, info = 0.0, 0, {}
    fase_prec, accum, per_fase = None, {}, {}
    while True:
        a = model.predict(obs, deterministic=True)[0] if model else \
            np.array([env.action_space.sample()])
        obs, r, done, infos = env.step(a)
        tot += float(r[0]); passi += 1; info = infos[0]
        fase = int(info.get("fsm_phase", 0))
        termini = info.get("reward_terms", {}) or {}
        for k, v in termini.items():
            accum[k] = accum.get(k, 0.0) + float(v)
        b = per_fase.setdefault(FASI[min(fase, 4)], {"n": 0, "R": 0.0, "t": {}})
        b["n"] += 1; b["R"] += float(r[0])
        for k, v in termini.items():
            b["t"][k] = b["t"].get(k, 0.0) + float(v)

        if verboso:
            if fase_prec is not None and fase != fase_prec:
                # TRANSIZIONE DI FASE: la si stampa sempre, con il bilancio
                # della fase che si sta chiudendo e i suoi termini principali.
                pb = per_fase[FASI[min(fase_prec, 4)]]
                top = sorted(pb["t"].items(), key=lambda kv: -abs(kv[1]))[:5]
                print(f"  >>> step {passi:>3}  {FASI[fase_prec]} -> {FASI[min(fase,4)]}"
                      f"   | fase chiusa: {pb['n']} step, R {pb['R']:+.1f}"
                      f"   | {' · '.join(f'{k} {v:+.1f}' for k, v in top)}")
            if hud_ogni and passi % hud_ogni == 0:
                att = sorted(termini.items(), key=lambda kv: -abs(kv[1]))[:6]
                masch = info.get("mascherati") or []
                print(f"      step {passi:>3} {FASI[min(fase,4)]:<8} r {float(r[0]):+7.2f}"
                      f" | e {info.get('door_error', 0):+.4f} A {int(bool(info.get('A')))}"
                      f" | {CONTROLLORI[int(info.get('controllore', 0))]}"
                      + (f" | mascherati: {','.join(masch)}" if masch else "")
                      + f"\n           " + " · ".join(f"{k} {v:+.2f}" for k, v in att))
        fase_prec = fase
        if render:
            env.envs[0].render() if hasattr(env, "envs") else None
            if slow: time.sleep(slow / 30.0)
        if done[0]:
            break

    if verboso:
        print(f"\n  --- reward machine dell'episodio ({passi} step, ritorno {tot:+.1f}) ---")
        print(f"  {'fase':<9}{'step':>6}{'R':>10}{'R/step':>9}   termini principali")
        for nome, b in per_fase.items():
            top = sorted(b["t"].items(), key=lambda kv: -abs(kv[1]))[:5]
            print(f"  {nome:<9}{b['n']:>6}{b['R']:>10.1f}{b['R']/max(b['n'],1):>9.2f}   "
                  + " · ".join(f"{k} {v:+.1f}" for k, v in top))
        print(f"  {'-'*72}")
        print("  totale per termine: " + " · ".join(
            f"{k} {v:+.1f}" for k, v in sorted(accum.items(), key=lambda kv: -abs(kv[1]))))
    return dict(ritorno=tot, passi=passi,
                is_success=bool(info.get("is_success", False)),
                door_error=float(info.get("door_error", 0.0)),
                fase=int(info.get("fsm_phase", 0)), termini=accum)


def metriche(eps):
    """§7 — i due numeri che vanno riportati INSIEME.

    Il risultato interessante non è «l'apertura è migliorata», è «la stessa
    macchina serve due compiti senza peggiorarne nessuno».
    """
    err = np.array([e["door_error"] for e in eps], dtype=float)
    return {
        "success_rate": float(np.mean([e["is_success"] for e in eps])),
        "errore_medio_con_segno": float(err.mean()),      # criterio 2: deve smettere
        "errore_medio_assoluto": float(np.abs(err).mean()),  # di essere sistematico
        "ritorno_medio": float(np.mean([e["ritorno"] for e in eps])),
        "passi_medi": float(np.mean([e["passi"] for e in eps])),
        "n_episodi": len(eps),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=["close", "open"], required=True,
                    help="l'UNICA differenza fra i due esperimenti (§6)")
    ap.add_argument("--play", action="store_true", help="mostra il robot")
    ap.add_argument("--eval", action="store_true", help="valuta e stampa le metriche del §7")
    ap.add_argument("--total-steps", type=int, default=1_000_000)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--slow", type=float, default=1.0, help="rallenta il play")
    ap.add_argument("--verbose", action="store_true",
                    help="stampa transizioni di fase e reward machine anche in --eval")
    ap.add_argument("--diario-ogni", type=int, default=20_000,
                    help="in addestramento stampa la reward machine ogni N passi (0 = mai)")
    ap.add_argument("--hud-every", type=int, default=0,
                    help="stampa lo stato ogni N step (0 = solo le transizioni)")
    args = ap.parse_args()
    args.run_dir = args.run_dir or f"runs/unified_{args.task}"

    cfg = UnifiedConfig.per(args.task)
    intestazione(cfg)

    if args.play or args.eval:
        env, model = carica(cfg, args, render_mode="human" if args.play else None)
        eps = []
        for i in range(1 if args.play else args.episodes):
            np.random.seed(args.seed + i)
            e = episodio(env, model, render=args.play, slow=args.slow,
                         verboso=args.play or args.verbose, hud_ogni=args.hud_every)
            eps.append(e)
            print(f"  ep {i}: {e['passi']:>3} step · ritorno {e['ritorno']:+9.1f} · "
                  f"errore {e['door_error']:+.4f} · {'RIUSCITO' if e['is_success'] else 'no'}")
        print("\nmetriche §7:")
        for k, v in metriche(eps).items():
            print(f"  {k:<26} {v}")
        env.close()
    else:
        addestra(cfg, args)


if __name__ == "__main__":
    main()
