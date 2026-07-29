#!/usr/bin/env python3
# scratch/test_open_task_v2/phase_diagnostics.py
"""
phase_diagnostics — Diagnostica delle fasi HOLD_OPEN e RETREAT (task APERTURA v2).

Porta `diagnose_phase.py` (stampe qualitative su ~5 episodi) a statistica a intervallo
su rollout seedati. Le grandezze sono scelte per discriminare le cause documentate nei
commenti del sorgente, non a caso:

  T3 — ‖azione braccio‖ in HOLD_OPEN. Misura quanto la policy TENTA di muoversi mentre
       dovrebbe tenere la porta ferma al goal. Valori alti = il braccio lotta contro la
       porta (candidato per il rimbalzo).
  T4 — ‖rotazione del polso‖ (action[3:6]) in RETREAT. Nel riporto §1.49 la rotazione è
       imposta dall'env; dopo il rilascio dovrebbe essere piccola. Torsioni residue
       tengono il dito nel piano della leva.
  T5 — STATO ALLA TRANSIZIONE HOLD_OPEN→RETREAT: open_error (la porta è davvero al goal
       quando si inizia il ritiro?) e latch_qpos (la leva è a fondo corsa, ~1.57 rad:
       è il carico che il riporto §1.46 deve scaricare).
  T6 — REGRESSI in HOLD_OPEN: step in cui open_error supera la tolleranza, cioè la porta
       cala sotto il goal (specchio esatto del rimbalzo misurato nella chiusura), con la
       velocità di cardine associata.
  T7 — RITIRO EFFETTIVO (§1.55): distribuzione dell'allontanamento massimo del braccio in
       RETREAT e frazione di episodi "fermi sulla maniglia" (< 6 cm). È la metrica che
       distingue un ritiro riuscito da una terminazione esogena mascherata da successo.

Rif.: Agarwal et al. 2021 (IQM + bootstrap CI); Chan et al. 2020 (dispersione e rischio);
Wilson 1927 (CI per le frazioni).

Output in results/phase/.
"""

from __future__ import annotations

import os
import json
import argparse
import numpy as np

from _common import (make_raw_env, load_obs_rms, load_model, rollout_episode,
                     results_dir, setup_matplotlib, safe_hist, json_default, resolve_run_dir,
                     CURRICULUM, STUCK_MOVE_THRESH)
import stats_utils as S


def run(n_episodes, deterministic, curriculum, run_dir, base_seed=30_000, tag=None):
    tag = tag or "curr1_posa_variabile"
    run_dir = resolve_run_dir(run_dir)
    env, cfg = make_raw_env(curriculum_level=curriculum, run_dir=run_dir)
    obs_rms = load_obs_rms(run_dir=run_dir)
    model = load_model(run_dir=run_dir)
    open_tol = float(getattr(cfg, "open_tol_rad", 0.05))

    hold_norms, wrist_rots, regress = [], [], []
    oe_trans, latch_trans, moved_max = [], [], []
    dev_per_ep = []          # (n_oltre, n_sotto) per episodio → statistica episodio-pesata
    moved_ret = []           # allontanamento SOLO degli episodi che raggiungono RETREAT
    succ = 0
    n_ret = 0
    for i in range(n_episodes):
        rec = rollout_episode(env, model, obs_rms, deterministic=deterministic,
                              seed=base_seed + i, collect_trace=True)
        succ += int(rec.true_success)
        hold_norms += rec.hold_action_norms
        wrist_rots += rec.retreat_wrist_rots
        regress += rec.regress_events
        moved_max.append(rec.retreat_moved_max)
        dev_per_ep.append((sum(1 for _, e, _ in rec.regress_events if e > 0),
                           sum(1 for _, e, _ in rec.regress_events if e < 0)))
        if rec.max_phase == "RETREAT":
            n_ret += 1
            moved_ret.append(rec.retreat_moved_max)
        if rec.open_error_at_transition is not None:
            oe_trans.append(rec.open_error_at_transition)
        if rec.latch_at_transition is not None:
            latch_trans.append(rec.latch_at_transition)
        ot = "n/a" if rec.open_error_at_transition is None else f"{rec.open_error_at_transition:.4f}"
        lt = "n/a" if rec.latch_at_transition is None else f"{rec.latch_at_transition:+.3f}"
        print(f"  ep {i+1:>3} | true={rec.true_success} | open_err@transiz={ot} "
              f"latch@transiz={lt} | mosso={rec.retreat_moved_max:.3f}m | "
              f"HOLD_n={len(rec.hold_action_norms)} RET_n={len(rec.retreat_wrist_rots)} | "
              f"{rec.termination_type}")
    try:
        env.close()
    except Exception:
        pass

    out = dict(mode=("det" if deterministic else "sto"), curriculum=curriculum,
               n_episodes=n_episodes,
               true_success_rate=S.wilson_ci(succ, n_episodes).as_dict())

    print("\n" + "=" * 76)
    print(f"DIAGNOSTICA FASI APERTURA — {'det' if deterministic else 'sto'} "
          f"({tag}, {n_episodes} episodi)")
    print("=" * 76)

    if hold_norms:
        a = np.asarray(hold_norms)
        out["T3_hold_action_norm"] = dict(n=len(a), iqm=S.bootstrap_ci(a, "iqm").as_dict(),
                                          frac_below_005=float((a < 0.05).mean()),
                                          frac_above_03=float((a > 0.3).mean()), values=a.tolist())
        print(f"  T3 HOLD_OPEN ‖a‖   : IQM {S.bootstrap_ci(a,'iqm')} | "
              f"%<0.05={(a<0.05).mean()*100:.1f}%  %>0.30={(a>0.3).mean()*100:.1f}%")
    else:
        out["T3_hold_action_norm"] = None
        print("  T3 HOLD_OPEN ‖a‖   : nessun dato (mai entrato in HOLD_OPEN)")

    if wrist_rots:
        a = np.asarray(wrist_rots)
        out["T4_retreat_wrist_rot"] = dict(n=len(a), iqm=S.bootstrap_ci(a, "iqm").as_dict(),
                                           frac_above_01=float((a > 0.1).mean()), values=a.tolist())
        print(f"  T4 RETREAT polso   : IQM {S.bootstrap_ci(a,'iqm')} | %>0.1={(a>0.1).mean()*100:.1f}%")
    else:
        out["T4_retreat_wrist_rot"] = None
        print("  T4 RETREAT polso   : nessun dato")

    if oe_trans:
        a = np.asarray(oe_trans); bad = int((a > open_tol).sum())
        out["T5_open_error_at_transition"] = dict(
            n=len(a), mean=float(a.mean()), std=float(a.std()),
            min=float(a.min()), max=float(a.max()),
            frac_above_tol=float(bad / len(a)),
            frac_above_tol_ci=S.wilson_ci(bad, len(a)).as_dict(), values=a.tolist())
        print(f"  T5a open_err@trans : media={a.mean():.4f} rad | "
              f"% oltre tol({open_tol}): {S.wilson_ci(bad,len(a))}")
    else:
        out["T5_open_error_at_transition"] = None
        print("  T5a open_err@trans : nessuna transizione HOLD_OPEN→RETREAT")

    if latch_trans:
        a = np.asarray(latch_trans)
        out["T5_latch_at_transition"] = dict(n=len(a), mean=float(a.mean()), std=float(a.std()),
                                             min=float(a.min()), max=float(a.max()),
                                             values=a.tolist())
        print(f"  T5b latch@trans    : media={a.mean():+.3f} rad "
              f"[{a.min():+.3f},{a.max():+.3f}] (carico che il riporto §1.46 deve scaricare)")
    else:
        out["T5_latch_at_transition"] = None
        print("  T5b latch@trans    : n/d")

    n_over = int(sum(1 for _, e, _ in regress if e > 0))
    # ATTENZIONE METODOLOGICA. Il conteggio per STEP non è episodio-pesato: un episodio in
    # stallo resta centinaia di step fuori tolleranza e da solo può ribaltare la
    # proporzione. (Osservato: con 30 episodi senza stalli il conteggio dava il 91% di
    # scostamenti "oltre"; con 100 episodi, 4 stalli hanno portato gli "oltre" al 12%.)
    # Si riporta quindi anche la statistica EPISODIO-pesata, che è quella da citare, e un
    # indice di dominanza che segnala quando pochi episodi governano il totale.
    ep_over = int(sum(1 for o, _ in dev_per_ep if o > 0))
    ep_under = int(sum(1 for _, u in dev_per_ep if u > 0))
    tot_per_ep = sorted((o + u for o, u in dev_per_ep), reverse=True)
    dominance = (float(sum(tot_per_ep[:3]) / len(regress)) if regress else None)
    out["T6_deviation_events"] = dict(
        n=len(regress), n_overshoot=n_over, n_regress=len(regress) - n_over,
        frac_overshoot=(float(n_over / len(regress)) if regress else None),
        # ── statistica episodio-pesata (da preferire) ──
        n_episodes=len(dev_per_ep),
        episodes_with_overshoot=ep_over, episodes_with_regress=ep_under,
        frac_episodes_overshoot=float(ep_over / max(1, len(dev_per_ep))),
        frac_episodes_regress=float(ep_under / max(1, len(dev_per_ep))),
        episodes_overshoot_ci=S.wilson_ci(ep_over, max(1, len(dev_per_ep))).as_dict(),
        episodes_regress_ci=S.wilson_ci(ep_under, max(1, len(dev_per_ep))).as_dict(),
        top3_episode_share=dominance,
        max_abs_error=float(max((abs(e) for _, e, _ in regress), default=0.0)),
        max_qvel=float(max((abs(v) for _, _, v in regress), default=0.0)),
        severe=int(sum(1 for _, e, _ in regress if abs(e) > 2 * open_tol)),
        events=[(int(s), float(e), float(v)) for s, e, v in regress])
    print(f"  T6 scostamenti HOLD:")
    print(f"      per EPISODIO (da citare): {ep_over}/{len(dev_per_ep)} episodi con almeno "
          f"uno scostamento OLTRE il goal, {ep_under}/{len(dev_per_ep)} con almeno uno SOTTO")
    print(f"      per STEP (non episodio-pesato): {len(regress)} step "
          f"({n_over} oltre, {len(regress)-n_over} sotto)"
          + (f" — i 3 episodi peggiori pesano il {dominance*100:.0f}% del totale"
             if dominance is not None else ""))
    print(f"      |err| max {out['T6_deviation_events']['max_abs_error']:.4f} rad, "
          f"severi(>2·tol) {out['T6_deviation_events']['severe']}")

    # "Fermo sulla maniglia" ha senso SOLO per gli episodi che il RETREAT lo raggiungono:
    # chi si blocca prima ha allontanamento 0 per definizione, non per incastro. Il
    # denominatore corretto è quindi n_ret (come già fa la batteria evaluate).
    a = np.asarray(moved_max) if moved_max else np.zeros(0)
    ar = np.asarray(moved_ret) if moved_ret else np.zeros(0)
    n_stuck = int((ar < STUCK_MOVE_THRESH).sum()) if ar.size else 0
    out["T7_retreat_moved"] = dict(
        n=int(a.size), n_retreat_episodes=int(ar.size),
        iqm=(S.bootstrap_ci(ar, "iqm").as_dict() if ar.size else None),
        cvar_worst10=(S.cvar(ar, 0.1, lower_tail=True) if ar.size else None),
        n_stuck=n_stuck,
        stuck_ci=S.wilson_ci(n_stuck, max(1, int(ar.size))).as_dict(),
        n_never_retreat=int(a.size - ar.size),
        values=ar.tolist(), values_all=a.tolist())
    if ar.size:
        print(f"  T7 allontanamento  : IQM {S.bootstrap_ci(ar,'iqm')} m | "
              f"CVaR peggior 10% = {S.cvar(ar, 0.1, lower_tail=True):.4f} m | "
              f"fermi sulla maniglia (<{STUCK_MOVE_THRESH} m): "
              f"{S.wilson_ci(n_stuck, int(ar.size))} sui {int(ar.size)} episodi che "
              f"raggiungono il RETREAT ({int(a.size - ar.size)} non lo raggiungono)")
    print("=" * 76 + "\n")

    plt = setup_matplotlib(); outdir = results_dir("phase")
    m = "det" if deterministic else "sto"
    if hold_norms:
        fig, ax = plt.subplots(figsize=(8, 5)); safe_hist(ax, hold_norms, 40, color="#1f77b4")
        ax.axvline(0.05, ls="--", color="k")
        ax.set_title(f"T3 — ‖action[:-1]‖ in HOLD_OPEN ({tag})")
        ax.set_xlabel("norma azione braccio richiesta dalla policy")
        fig.tight_layout(); fig.savefig(os.path.join(outdir, f"plot_T3_hold_norm_{m}_{tag}.png"), dpi=130); plt.close(fig)
    if wrist_rots:
        fig, ax = plt.subplots(figsize=(8, 5)); safe_hist(ax, wrist_rots, 40, color="#ff7f0e")
        ax.set_title(f"T4 — torsione del polso in RETREAT ({tag})"); ax.set_xlabel("norma rotazione")
        fig.tight_layout(); fig.savefig(os.path.join(outdir, f"plot_T4_wrist_{m}_{tag}.png"), dpi=130); plt.close(fig)
    if oe_trans or latch_trans:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        if oe_trans:
            safe_hist(axes[0], oe_trans, 30, color="#2ca02c")
            axes[0].axvline(open_tol, ls="--", color="k")
            axes[0].set_title("T5a — open_error alla transizione"); axes[0].set_xlabel("rad")
        if latch_trans:
            safe_hist(axes[1], latch_trans, 30, color="#9467bd")
            axes[1].set_title("T5b — latch_qpos alla transizione"); axes[1].set_xlabel("rad")
        fig.suptitle(f"Stato all'ingresso in RETREAT ({tag})")
        fig.tight_layout(); fig.savefig(os.path.join(outdir, f"plot_T5_transition_{m}_{tag}.png"), dpi=130); plt.close(fig)
    if regress:
        fig, ax = plt.subplots(figsize=(8, 5))
        safe_hist(ax, [e for _, e, _ in regress], 30, color="#d62728")
        ax.axvline(open_tol, ls="--", color="k"); ax.axvline(-open_tol, ls="--", color="k")
        ax.axvline(0.0, color="k", lw=0.8)
        _t6 = out["T6_deviation_events"]
        ax.set_title(f"T6 — scostamenti in HOLD_OPEN ({tag})\n"
                     f"destra = aperta OLTRE il goal, sinistra = ricaduta sotto  ·  "
                     f"per episodio: {_t6['episodes_with_overshoot']} oltre / "
                     f"{_t6['episodes_with_regress']} sotto su {_t6['n_episodes']}",
                     fontsize=10)
        ax.set_xlabel("errore con segno (angolo − goal) [rad]")
        fig.tight_layout(); fig.savefig(os.path.join(outdir, f"plot_T6_regress_{m}_{tag}.png"), dpi=130); plt.close(fig)
    if ar.size:
        fig, ax = plt.subplots(figsize=(8, 5)); safe_hist(ax, ar, 30, color="#17becf")
        ax.axvline(STUCK_MOVE_THRESH, ls="--", color="k")
        ax.set_title(f"T7 — allontanamento massimo del braccio in RETREAT ({tag})\n"
                     f"solo i {int(ar.size)} episodi che raggiungono il RETREAT "
                     f"({int(a.size - ar.size)} esclusi: bloccati prima)")
        ax.set_xlabel("m (a sinistra della linea = fermo sulla maniglia)")
        fig.tight_layout(); fig.savefig(os.path.join(outdir, f"plot_T7_moved_{m}_{tag}.png"), dpi=130); plt.close(fig)

    with open(os.path.join(outdir, f"phase_diag_{m}_{tag}.json"), "w") as f:
        json.dump(out, f, indent=2, default=json_default)
    print(f"Risultati in {outdir}")
    return out


def main():
    ap = argparse.ArgumentParser(description="Diagnostica fasi HOLD_OPEN/RETREAT — apertura v2")
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--curriculum", type=float, default=CURRICULUM)
    ap.add_argument("--run-dir", type=str, default=None)
    ap.add_argument("--stochastic", action="store_true")
    args = ap.parse_args()
    run(args.episodes, not args.stochastic, args.curriculum, args.run_dir)


if __name__ == "__main__":
    main()
