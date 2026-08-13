#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_unified.py — verifica che il codice faccia ESATTAMENTE quello che dice
il documento tabella_reward_machine_unificata.md.

Ogni test cita la sezione che verifica. Si esegue con:
    python3 tests/test_unified.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np

from config_unified import UnifiedConfig, Weights
from state_unified import DoorState, USI_DI_A
from fsm_unified import UnifiedFSM, Fase, Controllore, MASCHERA
from reward_unified import TERMINI, ASSORBE, SOPPRESSI, UnifiedReward, Potenziale

OK = KO = 0
def check(cond, msg):
    global OK, KO
    if cond: OK += 1
    else:
        KO += 1
        print(f"   FALLITO · {msg}")


# ══ §4 la chiusura è l'apertura con bersaglio zero ═════════════════════
def test_due_compiti_una_macchina():
    cc, co = UnifiedConfig.per("close"), UnifiedConfig.per("open")
    # §4 la chiusura è l'apertura con bersaglio zero: stessa struttura,
    # frazioni speculari (parte aperta e punta a 0, o viceversa)
    check(cc.task.theta_star_frac == (0.015, 0.015), "§4 bersaglio della chiusura ~0")
    check(cc.task.theta_zero_frac == (0.70, 1.00), "§4 la chiusura parte APERTA")
    check(co.task.theta_zero_frac == (0.0, 0.0), "§4 l'apertura parte CHIUSA")
    check(co.task.theta_star_frac == (0.85, 1.00), "§6 goal campionato all'85-100 %")
    # verso del compito opposto, escursione confrontabile
    dc = DoorState(0, 0, 0.03); dc.reset(0.36, 0.006, 0.03)
    do = DoorState(0, 0, 0.05); do.reset(0.0, 0.384, 0.05)
    check(dc.sigma == -1.0 and do.sigma == +1.0, "§4 σ opposto nei due compiti")
    check(abs(dc.escursione - do.escursione) < 0.05, "§4 escursioni confrontabili")
    # la differenza fra i due compiti è SOLO nei dieci parametri del §6
    dieci = {"theta_star_frac", "theta_zero_frac", "tol", "t_hold_s",
             "d_ret", "z_ret", "orienta_normale", "riporto_leva",
             "leva_rilascio", "escape"}
    diversi = {k for k in vars(cc.task) if k != "nome"
               and getattr(cc.task, k) != getattr(co.task, k)}
    check(diversi <= dieci, f"§6 differenze fuori dai dieci parametri: {diversi - dieci}")
    check(cc.task.leva_rilascio == 0.0, "§6 chiusura: nessun blocco di chiavistello")
    check(co.task.leva_rilascio == 1.03, "§6 apertura: rilascio a |leva| = 1.03 rad")
    # i pesi sono gli stessi oggetti per i due compiti
    check(vars(cc.w) == vars(co.w), "§1 i pesi non dipendono dal compito")
    # iperparametri invariati (§6)
    for k in ("net_arch", "learning_rate", "batch_size", "gamma", "tau",
              "buffer_size", "learning_starts", "control_freq", "horizon"):
        check(getattr(cc.sac, k) == getattr(co.sac, k), f"§6 {k} deve essere invariato")
    check(cc.sac.net_arch == (512, 512), "§6 rete MLP [512, 512]")


# ══ §4 le quattro grandezze derivate ═══════════════════════════════════
def test_grandezze_derivate():
    d = DoorState(theta_star=0.0, theta_zero=0.35, tol=0.03)
    d.reset(0.35, 0.0, 0.03)
    check(d.sigma == -1.0, "§4 σ = −1 nella chiusura")
    d2 = DoorState(theta_star=0.37, theta_zero=0.0, tol=0.05); d2.reset(0.0, 0.37, 0.05)
    check(d2.sigma == +1.0, "§4 σ = +1 nell'apertura")
    d2.step(0.37)
    check(abs(d2.e) < 1e-9, "§4 e = θ − θ* è nullo al bersaglio")
    check(d2.A, "§4 A è vera al bersaglio")
    d2.step(0.30)
    check(not d2.A, "§4 A è falsa a 0.07 dal bersaglio (tol 0.05)")


def _avanza(d, theta, leva=0.0):
    """Un passo di MOVE con la presa comandata sopra soglia: aggiorna e incassa."""
    d.step(theta, leva); d.ancora(); return d.incassa()


def test_cricchetto_e_saturazione():
    """§7 — il progresso non è ripagabile ed è saturato al bersaglio."""
    d = DoorState(theta_star=0.37, theta_zero=0.0, tol=0.05); d.reset(0.0, 0.37, 0.05)
    _avanza(d, 0.20)
    indietro = _avanza(d, 0.10)
    check(indietro == 0.0, "§7 tornare indietro non paga (cricchetto)")
    riavanti = _avanza(d, 0.20)
    check(riavanti == 0.0, "§7 rifare lo stesso tratto non paga")
    oltre = _avanza(d, 0.50)                   # sovra-apertura
    check(abs(d.p - 0.37) < 1e-9, "§7 p è SATURATO al bersaglio")
    check(abs((0.37 - 0.20) - oltre) < 1e-9, "§7 oltre il goal non si paga")
    # nella chiusura la saturazione era automatica: deve valere lo stesso
    c = DoorState(theta_star=0.0, theta_zero=0.35, tol=0.03); c.reset(0.35, 0.0, 0.03)
    _avanza(c, 0.0); _avanza(c, -0.05)
    check(abs(c.p - 0.35) < 1e-9, "§7 saturazione anche nella chiusura")


def test_cricchetto_ancorato_e_non_distruttivo():
    """§7 — le due righe che i sorgenti hanno e questa macchina aveva perso.

    1. Il riferimento si posa ENTRANDO in MOVE: il tratto che la porta percorre
       da sola durante REACH non consuma il budget.
    2. Il riferimento avanza SOLO quando si incassa: il tratto percorso con la
       presa comandata sotto soglia non viene distrutto, resta da incassare.
    """
    # 1 — la porta cade da sola durante REACH, poi si afferra
    d = DoorState(theta_star=0.0, theta_zero=0.35, tol=0.03); d.reset(0.35, 0.0, 0.03)
    d.step(0.20)                                   # REACH: solo stato, nessun incasso
    check(d.p == 0.0, "§7 in REACH il cricchetto non si muove")
    d.ancora()                                     # ingresso in MOVE
    check(abs(d.p - 0.15) < 1e-9, "§7 il riferimento si posa dove si è afferrato")
    check(d.incassa() == 0.0, "§7 posare il riferimento non paga la gravità")
    pagato = _avanza(d, 0.10)
    check(abs(pagato - 0.10) < 1e-9, "§7 si paga solo il tratto fatto dopo la presa")
    # 2 — la porta avanza mentre la presa è sotto soglia: non si distrugge nulla
    d2 = DoorState(theta_star=0.0, theta_zero=0.35, tol=0.03); d2.reset(0.35, 0.0, 0.03)
    d2.ancora()
    d2.step(0.25)                                  # presa sotto soglia: non si incassa
    d2.step(0.15)
    check(abs(_avanza(d2, 0.15) - 0.20) < 1e-9,
          "§7 il tratto non incassato resta da incassare, non sparisce")


def test_leva_come_seconda_porta():
    """§8 — dove il chiavistello blocca, la leva è il primo tratto della corsa.

    Due proprietà, entrambe dichiarate nel documento:
      · con `leva_rilascio` = 0 (chiusura) NULLA cambia rispetto a prima;
      · con `leva_rilascio` > 0 (apertura) girare la leva produce `progress`
        e alza `avanzamento` anche a cerniera perfettamente ferma.
    """
    # chiusura: comportamento identico a leva assente
    c = DoorState(theta_star=0.0, theta_zero=0.35, tol=0.03)
    c.reset(0.35, 0.0, 0.03, leva_rilascio=0.0)
    check(abs(c.escursione - 0.35) < 1e-9, "§8 chiusura: escursione = |θ* − θ₀|")
    d1 = _avanza(c, 0.20, leva=0.0)
    c2 = DoorState(theta_star=0.0, theta_zero=0.35, tol=0.03)
    c2.reset(0.35, 0.0, 0.03, leva_rilascio=0.0)
    d2 = _avanza(c2, 0.20, leva=-1.30)
    check(d1 == d2, "§8 con leva_rilascio = 0 la leva non entra nel progresso")

    # apertura: la corsa totale è leva + cerniera
    a = DoorState(theta_star=0.37, theta_zero=0.0, tol=0.05)
    a.reset(0.0, 0.37, 0.05, leva_rilascio=1.03)
    check(abs(a.escursione - 1.40) < 1e-9, "§8 apertura: escursione = leva + cerniera")
    check(a.avanzamento == 0.0, "§8 a riposo l'avanzamento è nullo")
    a.ancora()                                 # ingresso in MOVE, leva ancora a riposo
    dl = _avanza(a, 0.0, leva=-0.515)          # mezza leva, porta ferma
    check(dl > 0.0, "§8 girare la leva PAGA anche a cerniera ferma")
    check(abs(a.avanzamento - 0.5 * 1.03 / 1.40) < 1e-6, "§8 avanzamento = frazione di corsa")
    _avanza(a, 0.0, leva=-1.45)                # leva oltre il rilascio
    check(abs(a.avanzamento - 1.03 / 1.40) < 1e-6, "§8 oltre il rilascio la leva non paga più")
    _avanza(a, 0.37, leva=-1.45)               # porta al bersaglio
    check(abs(a.avanzamento - 1.0) < 1e-9, "§8 leva + cerniera = corsa completa")
    # il cricchetto vale sul lavoro composto
    check(_avanza(a, 0.20, leva=0.0) == 0.0, "§8 tornare indietro non ripaga (cricchetto)")


def test_target_rampa():
    """§1 — target = clip(1 − |e|/tol, 0, 1): rampa, non premio piatto."""
    d = DoorState(theta_star=0.37, theta_zero=0.0, tol=0.05); d.reset(0.0, 0.37, 0.05)
    d.step(0.37);        check(abs(d.target_ramp - 1.0) < 1e-9, "§1 vale 1 al bersaglio")
    d.step(0.37 + 0.025); check(abs(d.target_ramp - 0.5) < 1e-9, "§1 vale 0.5 a metà tolleranza")
    d.step(0.37 + 0.05);  check(abs(d.target_ramp) < 1e-9, "§1 vale 0 al bordo")
    d.step(0.37 + 0.20);  check(d.target_ramp == 0.0, "§1 non diventa negativo")
    # i due valori misurati e citati nel documento (§1)
    d.step(0.37 + 0.0027)
    check(abs(d.target_ramp - 0.946) < 0.002, "§1 ≈0.945 con errore +0.0027 rad")
    d.step(0.37 + 0.0401)
    check(abs(d.target_ramp - 0.198) < 0.002, "§1 ≈0.198 con errore +0.0401 rad")


# ══ §3 le quattro fasi e le due correzioni ═════════════════════════════
def _fsm(nome="open"):
    cfg = UnifiedConfig.per(nome)
    return cfg, UnifiedFSM(cfg)


def _passo(f, d, **kw):
    """fsm.step con i valori «presa salda, porta che avanza»: i test che non
    parlano di presa non devono farsi disturbare da isteresi e guardia."""
    arg = dict(presa_ok=True, contatto=True, dist_mano=0.02, latch=0.0,
               hold_target=30, grip_cmd=1.0, soglia=0.75, door_qvel=0.0)
    arg.update(kw)
    return f.step(door=d, **arg)


def test_gate_bilaterale():
    """§3 correzione 1 — MOVE→HOLD è bilaterale: la sovra-apertura NON passa."""
    cfg, f = _fsm("open")
    d = DoorState(theta_star=0.37, theta_zero=0.0, tol=0.05); d.reset(0.0, 0.37, 0.05)
    f.s.fase = Fase.MOVE
    d.step(0.50)          # ben oltre il goal: il vecchio gate θ ≥ goal−tol passava
    _passo(f, d)
    check(f.s.fase == Fase.MOVE, "§3 la sovra-apertura non deve entrare in HOLD")
    d.step(0.37)          # al bersaglio
    _passo(f, d)
    check(f.s.fase == Fase.HOLD, "§3 al bersaglio entra in HOLD")


def test_guardia_di_stallo():
    """§3 correzione 2 — senza guardia l'episodio in overshoot resta bloccato."""
    cfg, f = _fsm("open")
    d = DoorState(theta_star=0.37, theta_zero=0.0, tol=0.05); d.reset(0.0, 0.37, 0.05)
    f.s.fase = Fase.HOLD
    d.step(0.60)                                  # fuori tolleranza, non rientra
    for _ in range(cfg.thr.stall_guard_steps):
        _passo(f, d)
    check(f.s.fase == Fase.RELEASE, "§3 la guardia forza l'uscita da HOLD")
    check(cfg.thr.stall_guard_steps > 0, "§3 la guardia è configurata")


def test_timer_sale_e_scende():
    """§4 uso 2 — il timer sale se A, scende se non A."""
    cfg, f = _fsm("open")
    d = DoorState(theta_star=0.37, theta_zero=0.0, tol=0.05); d.reset(0.0, 0.37, 0.05)
    f.s.fase = Fase.HOLD
    d.step(0.37)
    _passo(f, d, hold_target=99)
    check(f.s.timer_hold == 1, "§4 il timer sale quando A è vera")
    d.step(0.50)
    _passo(f, d, hold_target=99)
    check(f.s.timer_hold == 0, "§4 il timer scende quando A è falsa")


def test_ritorno_di_fase():
    cfg, f = _fsm("close")
    d = DoorState(theta_star=0.0, theta_zero=0.35, tol=0.03); d.reset(0.35, 0.0, 0.03)
    f.s.fase = Fase.MOVE
    for _ in range(cfg.thr.grasp_lose_steps):
        _passo(f, d, presa_ok=False, contatto=False, dist_mano=0.2, hold_target=60)
    check(f.s.fase == Fase.REACH, "§3 presa persa per 3 frame → ritorno a REACH")


def test_isteresi_della_presa():
    """§3 correzione 3 — si afferra stretto (0.035 m), si perde largo (0.05–0.12 m).

    Senza isteresi la stessa soglia serve a confermare e a perdere: muovere il
    braccio — cioe' fare il compito — rischia di far cadere MOVE. La condotta
    ottima diventa afferrare e restare fermi, ed e' quella misurata nei training.
    """
    cfg, f = _fsm("close")
    d = DoorState(theta_star=0.0, theta_zero=0.35, tol=0.03); d.reset(0.35, 0.0, 0.03)
    d.step(0.30)                                     # lontano dal bersaglio
    soglia, thr = 0.75, cfg.thr
    # la soglia di CONFERMA e' 1.5·0.02 + 0.005 = 0.035 m
    check(f.soglia_distanza(0.02) < thr.lose_dist_base,
          "§3 si afferra piu' vicino di quanto si perda")
    check(not f.presa_persa(d, True, 0.045, 1.0, soglia, 0.0),
          "§3 a 0.045 m la presa NON e' persa")
    check(f.presa_persa(d, True, 0.13, 1.0, soglia, 0.0),
          "§3 a 0.13 m la presa e' persa")
    # porta in movimento: la tolleranza cresce fino al tetto
    check(not f.presa_persa(d, True, 0.11, 1.0, soglia, 1.0),
          "§3 se la porta si muove la mano puo' restare piu' indietro")
    # comando del gripper: margine di rilascio 0.20
    check(not f.presa_persa(d, True, 0.02, soglia - 0.15, soglia, 0.0),
          "§3 sotto soglia ma dentro il margine: presa mantenuta")
    check(f.presa_persa(d, True, 0.02, soglia - 0.25, soglia, 0.0),
          "§3 oltre il margine di rilascio: presa persa")
    # dita aperte: perde, TRANNE al bersaglio (`near_latch` della chiusura)
    check(f.presa_persa(d, False, 0.02, 1.0, soglia, 0.0),
          "§3 dita aperte lontano dal bersaglio: presa persa")
    d.step(0.02)                                     # |e| = 0.02 ≤ 0.05
    check(not f.presa_persa(d, False, 0.02, 1.0, soglia, 0.0),
          "§3 al bersaglio si accompagna la porta a mano aperta")


def test_nessuna_guardia_di_stallo_in_move():
    """§3 — la guardia di stallo esiste SOLO in HOLD.

    Provata anche in MOVE, misurata e tolta: produceva un ping-pong
    REACH<->MOVE al periodo della guardia, con Φ che scendeva di 4.6 a ogni
    espulsione (`phi` −0.82 per passo in REACH, termine dominante della fase) e
    `ep_rew_mean` da −400 a −464. Con la presa salda MOVE non si abbandona.
    """
    cfg, f = _fsm("close")
    d = DoorState(theta_star=0.0, theta_zero=0.35, tol=0.03); d.reset(0.35, 0.0, 0.03)
    d.step(0.20)
    f.s.fase = Fase.MOVE
    for _ in range(3 * cfg.thr.stall_guard_steps):
        _passo(f, d)
    check(f.s.fase == Fase.MOVE, "§3 con la presa salda MOVE non viene abbandonata")


def test_contact_non_paga_a_porta_ferma():
    """§1 termine 9 — `contact` premia il contatto DURANTE IL MOTO."""
    cfg = UnifiedConfig.per("close")
    rw, fsm = UnifiedReward(cfg), UnifiedFSM(cfg)
    d = DoorState(0, 0, 0.03); d.reset(0.35, 0.0, 0.03); d.step(0.20)
    fsm.s.fase = Fase.MOVE
    a = np.zeros(7); a[-1] = 1.0
    comune = dict(fsm=fsm, door=d, action=a, dist_mano=0.02, dist_xy=0.01, dz=0.0,
                  allineamento=1.0, polso_piatto=0.0, grip_cmd=1.0, soglia_presa=0.75,
                  contatto=True, door_qvel=0.0, latch=0.0, vel_giunti=0.0, raggio=0.02,
                  dist_ritiro=0.3, dir_ritiro=None, hold_target=30, terminato=False)
    fermo = rw.compute(delta_p=0.0, **comune)
    check("contact" not in fermo, "§1 termine 9: a porta ferma `contact` non paga")
    moto = rw.compute(delta_p=1e-4, **comune)
    check(moto.get("contact", 0.0) > 0.0, "§1 termine 9: con la porta in moto `contact` paga")


# ══ §8 i controllori e la maschera ═════════════════════════════════════
def test_maschera_completa():
    """§8 — la maschera è dichiarata in un punto solo ed è COMPLETA.

    Nel progetto originale copriva ret_grip, ret_dir, ret_perp ma NON ret_rot,
    che restava attivo e costava −14.44 su 27 step.
    """
    m = MASCHERA[Controllore.C0_RIPORTO_LEVA]
    check("release" in m, "§8 `release` è mascherato per intero")
    check("retreat" in m, "§8 `retreat` è mascherato per intero (incluso ret_rot)")
    check(MASCHERA[Controllore.C1_ESCAPE] == frozenset(), "§8 C1 non maschera nulla")
    # solo l'apertura accende i controllori
    check(UnifiedConfig.per("close").task.riporto_leva is False, "§6 chiusura: riporto off")
    check(UnifiedConfig.per("open").task.riporto_leva is True, "§6 apertura: riporto on")


def test_maschera_applicata():
    """I termini mascherati non devono comparire nel dizionario del reward."""
    cfg = UnifiedConfig.per("open")
    f, rw = UnifiedFSM(cfg), UnifiedReward(cfg)
    d = DoorState(theta_star=0.37, theta_zero=0.0, tol=0.05); d.reset(0.0, 0.37, 0.05)
    d.step(0.37)
    f.s.fase, f.s.controllore = Fase.RELEASE, Controllore.C0_RIPORTO_LEVA
    t = rw.compute(fsm=f, door=d, action=np.zeros(7), delta_p=0.0, dist_mano=0.02,
                   dist_xy=0.01, dz=0.0, allineamento=1.0, polso_piatto=0.0,
                   grip_cmd=1.0, soglia_presa=0.7, contatto=True,
                   door_qvel=0.0, latch=0.5, vel_giunti=0.0, raggio=0.02,
                   dist_ritiro=0.2, dir_ritiro=np.array([0., 1., 0.]),
                   hold_target=30, terminato=False)
    check("release" not in t, "§8 `release` non pagato durante il riporto")
    check("retreat" not in t, "§8 `retreat` non pagato durante il riporto")
    check("latch_home" in t, "§8 `latch_home` resta attivo: non è mascherato")


# ══ §1 i sedici termini ════════════════════════════════════════════════
def test_sedici_termini():
    check(len(TERMINI) == 16, f"§1 devono essere 16 termini, sono {len(TERMINI)}")
    check(set(ASSORBE) == set(TERMINI), "§5 ogni termine ha la sua mappatura")
    # copertura: 40 definizioni della chiusura + 25 dell'apertura, nessuna orfana
    sc = {k for k, v in SOPPRESSI.items() if v[0] == "chiusura"}
    so = {k for k, v in SOPPRESSI.items() if v[0] == "apertura"}
    c = {k for v in ASSORBE.values() for k in v[0]} | sc
    o = {k for v in ASSORBE.values() for k in v[1]} | so
    check(len(c) == 40, f"§5 devono essere 40 definizioni della chiusura, sono {len(c)}")
    check(len(o) == 25, f"§5 devono essere 25 definizioni dell'apertura, sono {len(o)}")
    check(len(c | o) == 41, f"§5 unione = 41, è {len(c | o)}")
    check(set(SOPPRESSI) == {"door_regress", "hold_veldamp"},
          "§5 due definizioni soppresse, entrambe dichiarate")


def test_mappatura_contro_i_sorgenti():
    """§5 — la mappatura va confrontata con i FILE ORIGINALI, non con sé stessa.

    Imposta PROGETTI_ORIGINALI alla cartella che contiene close_generalized_v2/
    e open_generalized_v2/. Senza quella variabile il test si salta.
    """
    import re
    base = os.environ.get("PROGETTI_ORIGINALI", "")
    fc = os.path.join(base, "close_generalized_v2", "reward_v2.py")
    fo = os.path.join(base, "open_generalized_v2", "reward_v2.py")
    if not (base and os.path.exists(fc) and os.path.exists(fo)):
        print("   (test_mappatura_contro_i_sorgenti saltato: PROGETTI_ORIGINALI non impostata)")
        return
    def kk(f, v):
        pat = re.compile(v + r'\["(\w+)"\]\s*=')
        return {m.group(1) for l in open(f) for m in [pat.search(l)] if m}
    CS, OS = kk(fc, "rew_info"), kk(fo, "rew")
    mc = ({k for v in ASSORBE.values() for k in v[0]}
          | {k for k, v in SOPPRESSI.items() if v[0] == "chiusura"})
    mo = ({k for v in ASSORBE.values() for k in v[1]}
          | {k for k, v in SOPPRESSI.items() if v[0] == "apertura"})
    check(not (CS - mc), f"§5 definizioni della chiusura non mappate: {sorted(CS - mc)}")
    check(not (mc - CS), f"§5 definizioni inventate per la chiusura: {sorted(mc - CS)}")
    check(not (OS - mo), f"§5 definizioni dell'apertura non mappate: {sorted(OS - mo)}")
    check(not (mo - OS), f"§5 definizioni inventate per l'apertura: {sorted(mo - OS)}")


def test_termini_per_fase():
    """§1 — ogni termine compare solo nelle fasi dichiarate dalla tabella."""
    cfg = UnifiedConfig.per("close")
    f, rw = UnifiedFSM(cfg), UnifiedReward(cfg)
    d = DoorState(theta_star=0.0, theta_zero=0.35, tol=0.03); d.reset(0.35, 0.0, 0.03)
    d.step(0.0)
    attesi = {
        Fase.REACH:   {"time", "smooth", "phi", "approach", "approach_geom", "wrist", "grip"},
        # `success` compare anche in MOVE: e' il bonus di transizione una tantum
        # (`phase_trans` degli originali), che il §1 fa assorbire da `success`.
        Fase.MOVE:    {"time", "smooth", "phi", "approach", "grip", "progress",
                       "contact", "success"},
        Fase.HOLD:    {"time", "smooth", "phi", "wrist", "target", "still", "hold_grip"},
        # `damp` NON c'e': vive solo in HOLD, come nel sorgente della chiusura
        Fase.RELEASE: {"time", "smooth", "phi", "target", "still", "release",
                       "retreat", "latch_home"},
    }
    for fase, ammessi in attesi.items():
        rw.reset(); f.s.fase, f.s.controllore = fase, Controllore.NESSUNO
        t = rw.compute(fsm=f, door=d, action=np.ones(7) * 0.3, delta_p=0.01, dist_mano=0.02,
                       dist_xy=0.01, dz=0.01, allineamento=0.9, polso_piatto=0.1,
                       grip_cmd=0.9, soglia_presa=0.7, contatto=True,
                       door_qvel=0.01, latch=0.02, vel_giunti=0.05, raggio=0.02,
                       dist_ritiro=0.1, dir_ritiro=np.array([1., 0., 0.]),
                       hold_target=60, terminato=False)
        extra = set(t) - ammessi
        check(not extra, f"§1 in {fase.name} compaiono termini non previsti: {extra}")


def test_success_una_volta_sola():
    """§1 — oggi il completamento è pagato DUE volte in entrambi i progetti."""
    cfg = UnifiedConfig.per("close")
    f, rw = UnifiedFSM(cfg), UnifiedReward(cfg)
    d = DoorState(theta_star=0.0, theta_zero=0.35, tol=0.03); d.reset(0.35, 0.0, 0.03)
    d.step(0.0); f.s.fase = Fase.RELEASE
    kw = dict(fsm=f, door=d, action=np.zeros(7), delta_p=0.0, dist_mano=0.02, dist_xy=0.0,
              dz=0.0, allineamento=1.0, polso_piatto=0.0, grip_cmd=-1.0, soglia_presa=0.7,
              contatto=False, door_qvel=0.0, latch=0.0, vel_giunti=0.0,
              raggio=0.02, dist_ritiro=0.02, dir_ritiro=np.array([1., 0., 0.]),
              hold_target=60, terminato=True)
    t1, t2 = rw.compute(**kw), rw.compute(**kw)
    check(t1.get("success") == cfg.w.success, "§1 il bonus è pagato alla terminazione")
    check("success" not in t2, "§1 e NON una seconda volta")


def test_progress_normalizzato():
    """§7 — percorrere l'escursione paga lo stesso budget nei due compiti."""
    w = Weights()
    for escursione in (0.35, 0.37):
        check(abs(w.w_progress(escursione) * escursione - w.budget_progress) < 1e-6,
              f"§7 budget costante per escursione {escursione}")
    # oggi i due budget divergono di 6x: 2000·0.35 = 700 contro 300·0.37 ≈ 111
    check(abs(2000 * 0.35 - w.budget_progress) < 1.0,
          "§7 il budget adottato è quello della chiusura (criterio §9)")


def test_damp_soppresso():
    """§7 — `hold_veldamp` non esiste piu': una porta che si muove non paga.

    Il sorgente dell'apertura lo ha respinto per iscritto (al goal la molla
    ritira la porta in modo inevitabile) e la misura gli da' ragione: −1.53 per
    passo in HOLD sull'apertura contro −0.05 sulla chiusura, dove e' inattivo
    perche' il bersaglio e' il punto di equilibrio.
    """
    cfg = UnifiedConfig.per("open")
    f, rw = UnifiedFSM(cfg), UnifiedReward(cfg)
    d = DoorState(theta_star=0.37, theta_zero=0.0, tol=0.05); d.reset(0.0, 0.37, 0.05)
    d.step(0.37); f.s.fase = Fase.HOLD
    t = rw.compute(fsm=f, door=d, action=np.zeros(7), delta_p=0.0, dist_mano=0.02,
                   dist_xy=0.0, dz=0.0, allineamento=1.0, polso_piatto=0.0, grip_cmd=0.9,
                   soglia_presa=0.7, contatto=True, door_qvel=0.2, latch=0.0,
                   vel_giunti=0.0, raggio=0.02, dist_ritiro=0.1, dir_ritiro=None,
                   hold_target=30, terminato=False)
    check("damp" not in t, "§7 nessun termine penalizza |dθ/dt|")
    check("damp" not in TERMINI and "hold_veldamp" in SOPPRESSI,
          "§7 `damp` non e' fra i termini e `hold_veldamp` e' dichiarato soppresso")


def test_formule_fedeli_al_documento():
    """§1 — le formule non devono avere fattori inventati.

    Nate da un audit: `approach` aveva un fattore di fase 0.5 che il documento
    non prevede, `still` un peso arbitrario, `damp` nessuna soglia.
    """
    import inspect
    from reward_unified import UnifiedReward as U
    src = inspect.getsource(U.compute)
    check("k = 1.0 if fase == Fase.REACH else 0.5" not in src,
          "§1 approach: nessun fattore di fase (il documento non lo prevede)")
    check("still_settle * 0.1" not in src, "§1 still: nessun peso arbitrario")
    w = Weights()
    check(hasattr(w, "still_release"), "§1 still: peso di RELEASE dichiarato")


def test_grandezze_lette_dalla_fisica():
    """Le grandezze devono venire dal simulatore, non da surrogati inventati."""
    import inspect
    from env_unified import UnifiedDoorEnv as E
    # polso piatto: componente della matrice di rotazione, non una distanza
    src = inspect.getsource(E._polso_piatto)
    check("robot0_eef_quat" in src, "polso_piatto viene dal quaternione del polso")
    check("self._handle" not in src, "polso_piatto NON e' una distanza dalla maniglia")
    # contatto: larghezza reale delle dita, non il comando del gripper
    src = inspect.getsource(E._leggi)
    check("robot0_gripper_qpos" in src, "contatto letto dalla larghezza reale del gripper")
    # controllore C0: tangente all'arco della leva da xaxis/xanchor
    src = inspect.getsource(E._direzione_leva)
    check("xaxis" in src and "xanchor" in src,
          "§8 C0: asse e ancora del giunto letti dal simulatore")
    check("np.cross" in src, "§8 C0: la tangente all'arco, non una direzione fissa")


def test_nessuna_fase_e_munta():
    """Nessuna fase ILLIMITATA deve pagare positivo restando fermi.

    Nato da un addestramento fallito: 780 000 step con ep_len 600, ritorno +500
    e successo 0 %. L'agente non stava sbagliando, stava massimizzando: restare
    fermi rendeva piu' che risolvere il compito [Krakovna et al. 2020].
    REACH e MOVE non hanno un tetto di durata, quindi il loro reward a regime
    deve essere <= 0. HOLD e RELEASE possono pagare, ma sono limitate dalla FSM.
    """
    cfg = UnifiedConfig.per("close")

    def regime(fase, passi=40, **over):
        f, rw = UnifiedFSM(cfg), UnifiedReward(cfg)
        d = DoorState(0, 0, 0.03); d.reset(0.362, 0.006, 0.03); d.step(0.006)
        f.s.fase = f.s.prev_fase = fase
        kw = dict(fsm=f, door=d, action=np.zeros(7), delta_p=0.0, dist_mano=0.02,
                  dist_xy=0.0, dz=0.0, allineamento=1.0, polso_piatto=0.0, grip_cmd=0.9,
                  soglia_presa=0.7, contatto=True, door_qvel=0.0,
                  latch=0.0, vel_giunti=0.0, raggio=0.02, dist_ritiro=0.1,
                  dir_ritiro=None, hold_target=60, terminato=False)
        kw.update(over)
        for _ in range(passi - 1):
            rw.compute(**kw)
        return sum(rw.compute(**kw).values())

    # MOVE non ha tetto di durata: deve costare, sempre.
    for g in (0.9, 0.65):
        v = regime(Fase.MOVE, grip_cmd=g)
        check(v <= 0.0, f"MOVE paga {v:+.2f}/step restando fermi: e' munta")
    # REACH e' illimitata SOLO per chi tiene il gripper sotto la soglia di presa:
    # chi lo chiude davvero transisce dopo 5 passi. Quel premio non si tocca —
    # e' l'unico segnale che insegna ad afferrare — ma la rendita che produce
    # deve restare sotto il pagamento del compito.
    sotto = regime(Fase.REACH, grip_cmd=0.65)
    onesto_ = cfg.w.budget_progress + cfg.w.transizione + cfg.w.success
    check(sotto * 600 < onesto_ * 2.0,
          f"REACH sotto soglia rende {sotto*600:+.0f} su 600 passi, troppo vicino a {onesto_:+.0f}")
    # le due fasi che possono pagare devono avere un tetto di durata
    check(cfg.thr.retreat_hard_cap > 0, "RELEASE deve avere un tetto di durata")
    check(cfg.thr.stall_guard_steps > 0, "HOLD deve avere la guardia di stallo")
    # e la rendita massima ottenibile non deve superare il percorso onesto
    rendita = regime(Fase.HOLD) * 60 + regime(Fase.RELEASE, grip_cmd=-1.0,
                                              contatto=False) * cfg.thr.retreat_hard_cap
    onesto = cfg.w.budget_progress + cfg.w.transizione + cfg.w.success
    check(rendita < onesto * 1.5,
          f"rendita massima {rendita:+.0f} troppo vicina al percorso onesto {onesto:+.0f}")


def test_still_in_release_e_penalita():
    """§8 — negli originali `ret_freeze` e' una penalita' PURA, mai un bonus.

    Un bonus di immobilita' nel ritiro e' l'incentivo ad accamparsi che il
    commento degli originali dichiara di voler evitare.
    """
    cfg = UnifiedConfig.per("close")
    f, rw = UnifiedFSM(cfg), UnifiedReward(cfg)
    d = DoorState(0, 0, 0.03); d.reset(0.362, 0.006, 0.03); d.step(0.006)
    f.s.fase = f.s.prev_fase = Fase.RELEASE
    kw = dict(fsm=f, door=d, action=np.zeros(7), delta_p=0.0, dist_mano=0.02, dist_xy=0.0,
              dz=0.0, allineamento=1.0, polso_piatto=0.0, grip_cmd=-1.0, soglia_presa=0.7,
              contatto=False, door_qvel=0.0, latch=0.0, vel_giunti=0.0,
              raggio=0.02, dist_ritiro=0.1, dir_ritiro=None, hold_target=60, terminato=False)
    rw.compute(**kw)
    check(rw.compute(**kw).get("still", 0.0) <= 0.0, "§8 `still` in RELEASE non deve mai essere positivo")


def test_bonus_di_transizione():
    """Il gradino che porta fuori da REACH: `phase_trans` degli originali, +10."""
    cfg = UnifiedConfig.per("close")
    check(cfg.w.transizione == 10.0, "il bonus di transizione vale 10, come negli originali")
    f, rw = UnifiedFSM(cfg), UnifiedReward(cfg)
    d = DoorState(0, 0, 0.03); d.reset(0.362, 0.006, 0.03); d.step(0.30)
    f.s.prev_fase, f.s.fase = Fase.REACH, Fase.MOVE
    kw = dict(fsm=f, door=d, action=np.zeros(7), delta_p=0.0, dist_mano=0.02, dist_xy=0.0,
              dz=0.0, allineamento=1.0, polso_piatto=0.0, grip_cmd=0.9, soglia_presa=0.7,
              contatto=True, door_qvel=0.0, latch=0.0, vel_giunti=0.0,
              raggio=0.02, dist_ritiro=0.1, dir_ritiro=None, hold_target=60, terminato=False)
    check(kw and rw.compute(**kw).get("success", 0.0) >= 10.0,
          "l'uscita da REACH deve pagare il bonus di transizione")


def test_scala_del_potenziale():
    """§2 — Φ vale ZERO in REACH: phi_reach e' il primo GRADINO, non una costante.

    Nei due sorgenti: `if phase == REACH: phi_now = 0.0`. Contando phi_reach
    anche in REACH la transizione REACH->MOVE valeva −0.15 (una penalita')
    invece di +2.85, e ogni fase avanzata veniva tassata piu' della precedente
    dalla deriva −(1−γ)Φ. E' la causa dei diari «REACH +0.35 · MOVE −0.91».
    """
    cfg = UnifiedConfig.per("close")
    d = DoorState(0, 0, 0.03); d.reset(0.362, 0.0, 0.03); d.step(0.362)
    pot = Potenziale(cfg)
    kw = dict(door=d, timer=0, hold_target=60, dist_ritiro=1.0,
              dist_mano=0.30, grip_cmd=-1.0, raggio=0.02)   # lontano, mano aperta
    check(pot._phi(Fase.REACH, **kw) == 0.0,
          "§2 Φ deve essere ZERO in REACH lontano dalla maniglia")
    # a presa gia' massima, il salto REACH -> MOVE e' esattamente phi_reach·k
    kwp = dict(kw, dist_mano=0.02, grip_cmd=1.0)
    salto = pot._phi(Fase.MOVE, **kwp) - pot._phi(Fase.REACH, **kwp)
    check(abs(salto - cfg.w.phi_reach * 1.5) < 1e-9,
          f"§2 uscendo da REACH si acquisisce il gradino phi_reach·k ({salto:+.3f})")
    # il gradino, misurato come lo vede l'agente
    pot.reset()
    pot.shaping(fase=Fase.REACH, **kw)          # primo passo: azzera prev
    pot.shaping(fase=Fase.REACH, **kw)
    salto = pot.shaping(fase=Fase.MOVE, **kw)
    check(salto > 2.5, f"§2 la transizione REACH->MOVE deve PAGARE, non costare ({salto:+.2f})")
    # la scala non scende mai passando alla fase successiva
    scala = [pot._phi(f, **kw) for f in (Fase.REACH, Fase.MOVE, Fase.HOLD, Fase.RELEASE)]
    check(all(b >= a for a, b in zip(scala, scala[1:])),
          f"§2 la scala del potenziale non deve scendere: {scala}")


def test_reach_non_puo_pagare_a_regime():
    """§8 — REACH non ha tetto di durata, quindi non puo' pagare positivo.

    Misura del training a 400-600 k passi: REACH 90 % dei passi, `grip` +2.1,
    HOLD mai raggiunta, ep_rew_mean in salita per il solo accamparsi. Causa: il
    premio si pagava alla sola VICINANZA, quindi l'agente teneva le dita sulla
    maniglia con il comando appena sotto la soglia — pagato, mai confermato.
    Ora il premio e' legato a `presa_ok`, la stessa condizione della conferma.
    """
    cfg = UnifiedConfig.per("close")

    def regime(**over):
        f, rw = UnifiedFSM(cfg), UnifiedReward(cfg)
        d = DoorState(0, 0, 0.03); d.reset(0.362, 0.006, 0.03); d.step(0.362)
        f.s.fase = f.s.prev_fase = Fase.REACH
        kw = dict(fsm=f, door=d, action=np.zeros(7), delta_p=0.0, dist_mano=0.02,
                  dist_xy=0.0, dz=0.0, allineamento=1.0, polso_piatto=0.0, grip_cmd=0.69,
                  soglia_presa=0.70, contatto=True, door_qvel=0.0,
                  latch=0.0, vel_giunti=0.0, raggio=0.02, dist_ritiro=1.0,
                  dir_ritiro=None, hold_target=60, terminato=False)
        kw.update(over)
        for _ in range(39):
            rw.compute(**kw)
        return sum(rw.compute(**kw).values())

    check(regime() < 0.0, "§8 dita sulla maniglia sotto soglia: deve costare")
    check(regime(contatto=False) < 0.0, "§8 dita chiuse sul vuoto: deve costare")
    check(regime(grip_cmd=0.9) < 0.0,
          "§8 nemmeno la presa confermata paga a REGIME: il premio e' un potenziale")

    # ...ma STRINGERE deve pagare, subito e una volta sola: e' il gradiente.
    f, rw = UnifiedFSM(cfg), UnifiedReward(cfg)
    d = DoorState(0, 0, 0.03); d.reset(0.362, 0.006, 0.03); d.step(0.362)
    f.s.fase = f.s.prev_fase = Fase.REACH
    base = dict(fsm=f, door=d, action=np.zeros(7), delta_p=0.0, dist_mano=0.02,
                dist_xy=0.0, dz=0.0, allineamento=1.0, polso_piatto=0.0,
                soglia_presa=0.70, contatto=True, door_qvel=0.0,
                latch=0.0, vel_giunti=0.0, raggio=0.02, dist_ritiro=1.0,
                dir_ritiro=None, hold_target=60, terminato=False)
    rw.compute(grip_cmd=-1.0, **base); rw.compute(grip_cmd=-1.0, **base)
    fermo = rw.compute(grip_cmd=-1.0, **base)["phi"]
    stringe = rw.compute(grip_cmd=0.9, **base)["phi"]
    check(stringe > 1.5, f"§8 stringere deve pagare subito ({stringe:+.2f})")
    check(fermo <= 0.0, "§8 tenere il comando fermo non deve pagare")


def test_potenziale_segue_la_porta():
    """§2 — Φ_move segue l'ANGOLO CORRENTE, non il cricchetto.

    Con il cricchetto dentro Φ il potenziale e' monotono: spingere e lasciar
    tornare indietro valgono uguale, e sparisce l'unico gradiente denso che dice
    al braccio in che verso spingere. Misurato nel training: la porta si chiude
    da sola per gravita' fino al 47 % (braccio fermo, gripper aperto), il
    cricchetto incassa quel tratto, e in MOVE `progress` non paga piu' nulla —
    l'agente resta aggrappato alla maniglia per 500 passi senza spingere.
    """
    cfg = UnifiedConfig.per("close")
    d = DoorState(0, 0, 0.03); d.reset(0.336, 0.006, 0.03)
    pot = Potenziale(cfg)
    kw = dict(timer=0, hold_target=60, dist_ritiro=1.0,
              dist_mano=0.02, grip_cmd=0.9, raggio=0.02)
    _avanza(d, 0.20); avanti = pot._phi(Fase.MOVE, door=d, **kw)
    _avanza(d, 0.15); piu_avanti = pot._phi(Fase.MOVE, door=d, **kw)
    _avanza(d, 0.28); indietro = pot._phi(Fase.MOVE, door=d, **kw)
    check(piu_avanti > avanti, "§2 avvicinare la porta al bersaglio deve alzare Φ")
    check(indietro < avanti, "§2 lasciarla tornare indietro deve ABBASSARE Φ")
    # il cricchetto invece resta, ed e' giusto: e' li' che vive `progress`
    check(abs(d.p - 0.186) < 1e-3, "§8 `p` resta a cricchetto")
    check(d.avanzamento < 0.25, "§2 `avanzamento` non e' a cricchetto")


def test_grip_lontano_non_e_mai_positivo():
    """§1 — la guardia `g > −0.85` degli originali, in ENTRAMBI i rami.

    Senza, a gripper spalancato (g = −1) il ramo lontano vale −w·(g+0.85) =
    +0.15/passo: una rendita per restare a mezz'aria con la mano aperta.
    """
    cfg = UnifiedConfig.per("close")
    for g in (-1.0, -0.9, -0.85, -0.5, 0.0, 1.0):
        f, rw = UnifiedFSM(cfg), UnifiedReward(cfg)
        d = DoorState(0, 0, 0.03); d.reset(0.362, 0.0, 0.03); d.step(0.362)
        f.s.fase = f.s.prev_fase = Fase.REACH
        t = rw.compute(fsm=f, door=d, action=np.zeros(7), delta_p=0.0, dist_mano=0.30,
                       dist_xy=0.2, dz=0.1, allineamento=1.0, polso_piatto=0.0,
                       grip_cmd=g, soglia_presa=0.7, contatto=False,
                       door_qvel=0.0, latch=0.0, vel_giunti=0.0, raggio=0.02,
                       dist_ritiro=1.0, dir_ritiro=None, hold_target=60, terminato=False)
        check(t.get("grip", 0.0) <= 0.0,
              f"§1 grip lontano non deve mai pagare (g={g}: {t.get('grip', 0.0):+.2f})")


def test_muovere_la_porta_conviene_piu_di_restare_in_reach():
    """§8 — l'invariante che conta: MOVE con la porta che si muove deve valere
    molto piu' di qualunque condotta stazionaria in REACH."""
    cfg = UnifiedConfig.per("close")

    def passo(fase, **over):
        f, rw = UnifiedFSM(cfg), UnifiedReward(cfg)
        d = DoorState(0, 0, 0.03); d.reset(0.362, 0.0, 0.03); d.step(0.362)
        f.s.fase = f.s.prev_fase = fase
        kw = dict(fsm=f, door=d, action=np.zeros(7), delta_p=0.0, dist_mano=0.02,
                  dist_xy=0.0, dz=0.0, allineamento=1.0, polso_piatto=0.0, grip_cmd=0.9,
                  soglia_presa=0.7, contatto=True, door_qvel=0.0,
                  latch=0.0, vel_giunti=0.0, raggio=0.02, dist_ritiro=1.0,
                  dir_ritiro=None, hold_target=60, terminato=False)
        kw.update(over)
        for _ in range(9):
            rw.compute(**kw)
        return sum(rw.compute(**kw).values())

    reach = passo(Fase.REACH)
    move_ferma = passo(Fase.MOVE)
    move_viva = passo(Fase.MOVE, delta_p=0.004)
    check(move_viva > reach,
          f"§8 muovere la porta deve pagare piu' di stare in REACH ({move_viva:+.2f} vs {reach:+.2f})")
    check(move_ferma < 0.0,
          f"§8 restare fermi in MOVE deve costare ({move_ferma:+.2f})")
    check(move_viva - move_ferma > 5.0,
          "§8 il progresso deve essere il termine dominante di MOVE")


def test_presa_richiede_contatto_fisico():
    """La conferma della presa ha TRE condizioni, come `grasp_cond` degli originali.

    Nato da un addestramento fallito: senza la chiusura fisica delle dita
    l'agente comandava il gripper a vuoto, si faceva confermare la presa,
    incassava il bonus di transizione e la riperdeva — un ciclo che pagava
    senza mai muovere la porta (ritorno +438, successo 0 %).
    """
    import inspect
    from env_unified import UnifiedDoorEnv as E
    src = inspect.getsource(E._leggi)
    riga = [l for l in src.split("\n") if "presa_ok =" in l]
    check(bool(riga), "la condizione di presa deve esistere")
    if riga:
        r = riga[0]
        check("vicino" in r, "presa: serve la vicinanza alla maniglia")
        check("soglia" in r, "presa: serve il comando sopra la soglia adattiva")
        check("contatto" in r, "presa: serve la chiusura FISICA delle dita")


def test_filtro_azione():
    """§6 (tesi §6.2.1) — media mobile esponenziale, alpha = 0.95."""
    check(UnifiedConfig.per("close").action_alpha == 0.95, "§6 filtro EMA alpha = 0.95")
    import inspect
    from env_unified import UnifiedDoorEnv as E
    check("action_alpha" in inspect.getsource(E.step), "§6 il filtro e' applicato nello step")


def test_premio_finale_non_tagliato():
    """§1 — il taglio ±100 non deve toccare il passo terminale.

    Gli originali scrivono `if not terminated: reward = clip(...)`. Tagliando
    anche la fine, i 600 che valgono il compito arrivano alla policy come 100:
    il bonus terminale smette di contare. Misurato nel diario: `success +600`
    fra i termini, ma la ricompensa effettiva del passo era 100.
    """
    cfg = UnifiedConfig.per("close")
    rw = UnifiedReward(cfg)
    t = {"time": -0.1, "success": cfg.w.success}
    check(abs(rw.total(t, terminato=True) - (cfg.w.success - 0.1)) < 1e-6,
          "§1 al passo terminale il premio arriva intero")
    check(abs(rw.total(t, terminato=False) - cfg.w.clip_step) < 1e-6,
          "§1 negli altri passi il taglio ±100 resta")


def test_controllori_di_fase():
    """§8 — blocco del braccio in HOLD e morsa in MOVE/HOLD.

    Entrambi presenti nei due originali (close env_v2 righe 175/180, open 451/318).
    Senza il blocco, in HOLD la porta continua a oscillare e `damp` (peso 25) la
    punisce: misurato −6…−12 per passo, HOLD a −14…−21 complessivi, cioe' una
    trappola in cui l'agente impara a non entrare.
    """
    import inspect
    from env_unified import UnifiedDoorEnv as E
    src = inspect.getsource(E._applica_controllori)
    check("Fase.HOLD" in src and "a[:-1] = 0.0" in src,
          "§8 in HOLD il braccio e' bloccato dall'ambiente")
    check("grip_lock_margin" in src, "§8 morsa sul gripper in MOVE e HOLD")
    check(UnifiedConfig.per("close").thr.grip_lock_margin > 0.0,
          "§8 il margine della morsa e' configurato")
    passo = inspect.getsource(E.step)
    check("action=eseguita" in passo,
          "§8 la ricompensa vede l'azione ESEGUITA, come negli originali")


def test_soglia_A_cinque_usi():
    check(len(USI_DI_A) == 5, f"§4 A è usata in cinque punti, dichiarati {len(USI_DI_A)}")


def test_osservazione_126():
    from env_unified import OBS_DIM, OBS_BLOCCHI
    check(OBS_DIM == 126, "§6 l'osservazione unificata ha 126 dimensioni")
    check(sum(n for _, n in OBS_BLOCCHI) == 126, "§6 i blocchi sommano a 126")


if __name__ == "__main__":
    print("Verifica dell'implementazione contro tabella_reward_machine_unificata.md\n")
    for nome, fn in sorted(list(globals().items())):
        if nome.startswith("test_"):
            fn()
    print(f"\n{OK + KO} controlli · {OK} superati · {KO} falliti")
    sys.exit(1 if KO else 0)
