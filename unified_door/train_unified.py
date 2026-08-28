#!/usr/bin/env python3

import os
import time
import argparse
import math
import numpy as np

from config_unified import UnifiedConfig

def crea_env(cfg, render_mode = None, seed = None):
    from env_unified import UnifiedDoorEnv
    env = UnifiedDoorEnv(cfg, render_mode=render_mode)
    if seed is not None:
        np.random.seed(seed)
    return env


def crea_vecenv(cfg, n_envs, seed):
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from stable_baselines3.common.monitor import Monitor

    venv = DummyVecEnv([(lambda i=i: Monitor(crea_env(cfg, seed=seed + i))) for i in range(n_envs)])
    return VecNormalize(venv, norm_obs = True, norm_reward = False, clip_obs = 10.0)


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
def _crea_callback_migliore(cfg, args, env_addestramento):
    """Salva l'ISTANTANEA MIGLIORE, non l'ultima.

    I due progetti originali consegnano due file — il modello finale e quello
    con la valutazione piu' alta — e per l'apertura il migliore e' a 650 000
    passi su un budget di 1 500 000, cioe' il finale e' PEGGIORE. Salvare solo
    l'ultimo butta via il picco. Qui la valutazione e' deterministica, su un
    ambiente separato, e la statistica di VecNormalize viene salvata INSIEME al
    modello: altrimenti in valutazione si caricherebbero pesi di un istante e
    normalizzazioni di un altro.
    """
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import sync_envs_normalization

    class Migliore(BaseCallback):
        def __init__(self):
            super().__init__()
            self.ogni = max(1, args.eval_ogni // cfg.sac.n_envs)
            self.migliore = -np.inf
            self.env_eval = None

        def _on_step(self) -> bool:
            if self.n_calls % self.ogni:
                return True
            if self.env_eval is None:
                self.env_eval = crea_vecenv(cfg, 1, args.seed + 10_000)
                self.env_eval.training = False
                self.env_eval.norm_reward = False
            sync_envs_normalization(env_addestramento, self.env_eval)
            eps = [episodio(self.env_eval, self.model) for _ in range(args.eval_episodi)]
            m = metriche(eps)
            # criterio: prima il successo, poi il ritorno a parita' di successo
            punteggio = m["success_rate"] * 1e6 + m["ritorno_medio"]
            if punteggio > self.migliore:
                self.migliore = punteggio
                self.model.save(os.path.join(args.run_dir, "best_model"))
                env_addestramento.save(os.path.join(args.run_dir, "vecnormalize.pkl"))
                print(f"  [migliore] {self.num_timesteps} passi · "
                      f"success {m['success_rate']:.2f} · ritorno {m['ritorno_medio']:+.0f}"
                      f" · salvato")
            return True

    return Migliore()


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
          _crea_callback_diario(args.diario_ogni),
          _crea_callback_migliore(cfg, args, env)]
    model.learn(total_timesteps=args.total_steps, callback=cb, progress_bar=False)
    model.save(os.path.join(args.run_dir, "final_model"))
    env.save(os.path.join(args.run_dir, "vecnormalize_final.pkl"))
    print(f"\nsalvato in {args.run_dir}: best_model.zip (istantanea migliore) "
          f"e final_model.zip (fine budget)")


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


def scena(env) -> str:
    """§9 — la scena estratta dopo il reset: posa della porta e fisica campionate.

    Serve al `--play` su piu' episodi: senza dichiararle, uno spettatore non ha
    modo di sapere che porta e attrito cambiano a ogni episodio, e il robot
    sembra ripetere sempre la stessa cosa. Sono gli stessi valori che entrano
    nell'osservazione (raggio e attrito) e nel timer di HOLD (cricchetto).
    """
    g = env.envs[0] if hasattr(env, "envs") else None
    if g is None or not hasattr(g, "door"):
        return ""
    amp = max(g.corsa_max - g.corsa_min, 1e-9)
    f = lambda t: (t - g.corsa_min) / amp
    r = g._rand
    base = r.base_latch_stiffness or 1.0
    return (f"  ── porta a {g.door.theta_zero:+.3f} rad ({100*f(g.door.theta_zero):.0f} % della corsa)"
            f"  →  bersaglio {g.door.theta_star:+.3f} rad ({100*f(g.door.theta_star):.0f} %)\n"
            f"     maniglia: raggio {r.current_handle_radius:.4f} m · attrito "
            f"{r.current_handle_friction:.2f} · cricchetto ×{r.current_latch_stiffness/base:.2f}")


def episodio(env, model, render=False, slow=0.0, verboso=False, hud_ogni=0,
             mostra_scena=False):
    """Esegue un episodio. Con `verboso` stampa transizioni e reward machine."""
    obs = env.reset()
    if mostra_scena:
        s = scena(env)
        if s:
            print(s)
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
        if render and not done[0]:
            # §9 — NON si disegna sull'ultimo passo. Il VecEnv ha gia' fatto
            # auto-reset, quindi quel fotogramma mostrerebbe l'episodio
            # SUCCESSIVO al posto della posa finale. I fotogrammi veri della
            # fine li disegna `_assestamento`, dentro `step`, prima del reset.
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
                permissivo=bool(info.get("successo_permissivo", False)),
                pulito=bool(info.get("successo_pulito", False)),
                spostato=float(info.get("ritiro_spostato", 0.0)),
                latch=float(info.get("latch_finale", 0.0)),
                door_error=float(info.get("door_error", 0.0)),
                fase=int(info.get("fsm_phase", 0)), termini=accum)


def wilson(k: int, n: int, z: float = 1.96):
    """§8 — intervallo di confidenza di Wilson al 95 %.

    E' lo stesso stimatore usato dalla suite dei due progetti separati
    (`scratch/test_*_task_v2/stats_utils.py`), quindi i numeri sono
    confrontabili con quelli della baseline senza conversioni.
    """
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1.0 + z * z / n
    centro = (p + z * z / (2 * n)) / d
    meta = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, centro - meta), min(1.0, centro + meta)


def metriche(eps):
    """§8 — i tre livelli di successo, con l'intervallo di confidenza.

    Il risultato interessante non è «l'apertura è migliorata», è «la stessa
    macchina serve due compiti senza peggiorarne nessuno».
    """
    err = np.array([e["door_error"] for e in eps], dtype=float)
    n = len(eps)
    k_true = int(sum(e["is_success"] for e in eps))
    k_pul = int(sum(e["pulito"] for e in eps))
    lo_t, hi_t = wilson(k_true, n)
    lo_p, hi_p = wilson(k_pul, n)
    return {
        # i tre livelli della tesi (§6.3.4): permissivo / true / clean
        "successo_permissivo": float(np.mean([e["permissivo"] for e in eps])),
        "success_rate": float(np.mean([e["is_success"] for e in eps])),
        "successo_pulito": float(np.mean([e["pulito"] for e in eps])),
        "true_IC95": f"[{lo_t:.3f}, {hi_t:.3f}]",
        "pulito_IC95": f"[{lo_p:.3f}, {hi_p:.3f}]",
        # §7 quanto la mano si e' allontanata dalla posa di rilascio [m]
        "ritiro_spostamento_medio": float(np.mean([e["spostato"] for e in eps])),
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
    ap.add_argument("--total-steps", type=int, default=1_500_000,
                    help="budget: 1.5·10⁶, quello dichiarato dal progetto dell'apertura")
    ap.add_argument("--eval-ogni", type=int, default=25_000,
                    help="ogni N passi valuta e, se è la migliore, salva l'istantanea")
    ap.add_argument("--eval-episodi", type=int, default=10,
                    help="episodi deterministici per ogni valutazione periodica")
    ap.add_argument("--episodes", type=int, default=None,
                    help="episodi da eseguire: 1 con --play, 20 con --eval, se non indicato")
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--eval-seeds", type=str, default=None,
                    help="§8 semi di valutazione separati da virgola, es. 42,101,7")
    ap.add_argument("--eval-offset", type=int, default=0,
                    help="§8 scarto dei semi di valutazione (0 = compatibile con le misure precedenti)")
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

        # §9 — la coda del play va disegnata allo STESSO ritmo dell'episodio,
        # altrimenti i suoi 90 fotogrammi passano in un decimo di secondo e non
        # si vede niente. `--slow` vale per tutti e due.
        if args.play and hasattr(env, "envs"):
            env.envs[0].ritmo_play = max(float(args.slow), 0.0) / 30.0

        # §8 — VALUTAZIONE SU PIU' SEMI. Un solo seme fissa quali porte vengono
        # provate: ripeterla su semi diversi mostra se il risultato dipende dal
        # campione o dalla politica. I semi sono indipendenti fra loro e l'unione
        # degli episodi da' l'intervallo di confidenza piu' stretto.
        semi = ([int(s) for s in args.eval_seeds.split(",") if s.strip()]
                if args.eval_seeds else [args.seed])
        # §9 — `--episodes` vale per il play come per la valutazione. Il default
        # resta uno solo a schermo e venti in valutazione, ma con `--episodes N`
        # il play mostra N porte diverse: e' il modo di far vedere che fisica e
        # posa cambiano a ogni episodio e la politica regge lo stesso.
        n_ep = args.episodes if args.episodes is not None else (1 if args.play else 20)
        tutti = []

        for seme in semi:
            eps = []
            if len(semi) > 1:
                print(f"\n─── seme di valutazione {seme} · {n_ep} episodi ───")
            for i in range(n_ep):
                # `eval_offset` tiene gli episodi di valutazione lontani da
                # quelli visti in addestramento (che usa `seed + 0..n_envs-1`).
                np.random.seed(seme + args.eval_offset + i)
                if args.play and n_ep > 1:
                    print(f"\n═══ episodio {i + 1} di {n_ep} ═══")
                e = episodio(env, model, render=args.play, slow=args.slow,
                             verboso=args.play or args.verbose, hud_ogni=args.hud_every,
                             mostra_scena=args.play)
                eps.append(e)
                print(f"  ep {i}: {e['passi']:>3} step · ritorno {e['ritorno']:+9.1f} · "
                      f"errore {e['door_error']:+.4f} · leva {e['latch']:+.3f} · "
                      f"{'RIUSCITO' if e['is_success'] else ('quasi' if e['permissivo'] else 'no')}")
            print(f"\nmetriche §8" + (f" · seme {seme}:" if len(semi) > 1 else ":"))
            for k, v in metriche(eps).items():
                print(f"  {k:<26} {v}")
            tutti.extend(eps)

        if len(semi) > 1:
            print(f"\n═══ TUTTI I SEMI UNITI · {len(tutti)} episodi ═══")
            for k, v in metriche(tutti).items():
                print(f"  {k:<26} {v}")
        env.close()
    else:
        addestra(cfg, args)


if __name__ == "__main__":
    main()
