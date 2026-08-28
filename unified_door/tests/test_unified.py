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
    # la differenza fra i due compiti è SOLO nei nove parametri del §6
    nove = {"theta_star_frac", "theta_zero_frac", "tol", "t_hold_s",
            "d_ret", "z_ret", "orienta_normale", "riporto_leva", "escape"}
    diversi = {k for k in vars(cc.task) if k != "nome"
               and getattr(cc.task, k) != getattr(co.task, k)}
    check(diversi <= nove, f"§6 differenze fuori dai nove parametri: {diversi - nove}")
    check(len(nove) == len(vars(cc.task)) - 1, "§6 TaskSpec ha esattamente nove campi")
    # la soglia di sblocco NON è un parametro di compito: è geometria della porta
    check(cc.thr.leva_rilascio == co.thr.leva_rilascio == 1.23,
          "§7 il chiavistello blocca in entrambi i versi: soglia unica, 1.23 rad")
    # §4 — anche il VERSO della maniglia è geometria della porta, non compito
    check(cc.thr.verso_leva == co.thr.verso_leva == +1.0,
          "§4 il verso che abbassa la maniglia è unico e positivo")
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


def test_leva_come_primo_tratto():
    """§7 — la leva è il primo tratto della corsa, nei DUE compiti, ma con un
    BUDGET SUO, non in concorrenza con la cerniera.

    Il chiavistello blocca la porta in entrambi i versi: con la leva a riposo la
    porta si ferma a 0.182 rad chiudendo e a 0.015 aprendo. Girarla è quindi
    lavoro utile in entrambi i casi, e va pagato in entrambi.

    Ma NON dallo stesso budget. Con `escursione = |θ*−θ₀| + leva_rilascio` la
    leva (1.23 rad) si prende l'80 % dei 700 punti e alla cerniera resta il
    20 %: il gradiente sull'angolo della porta scende a 458 per radiante e
    l'apertura, dove la cerniera È il compito, resta senza segnale. Togliendo la
    leva l'apertura riparte ma la chiusura si pianta contro il catenaccio —
    misurato, 18 valutazioni su 20 ferme a errore +0.1755, sempre lo stesso
    valore. `quota_leva` dà a ciascun tratto il suo budget: stessa formula per i
    due compiti, nessun ramo per task.
    """
    thr = UnifiedConfig.per("close").thr
    L, Q = thr.leva_rilascio, thr.quota_leva
    check(0.0 < Q < 0.5, "§7 la leva è un MEZZO: la sua quota è minore di 1/2")
    for nome, th0, ths, tol in (("chiusura", 0.35, 0.006, 0.03),
                                ("apertura", 0.0, 0.37, 0.05)):
        cerniera = abs(ths - th0)
        d = DoorState(theta_star=ths, theta_zero=th0, tol=tol)
        d.reset(th0, ths, tol, leva_rilascio=L, quota_leva=Q)
        leva_eq = cerniera * Q / (1.0 - Q)
        check(abs(d.escursione - (cerniera + leva_eq)) < 1e-9,
              f"§7 {nome}: escursione = cerniera + leva riscalata")
        # la quota di budget è la stessa nei due compiti, e vale Q
        check(abs(leva_eq / d.escursione - Q) < 1e-9,
              f"§7 {nome}: alla leva spetta esattamente quota_leva del budget")
        check(abs(cerniera / d.escursione - (1.0 - Q)) < 1e-9,
              f"§7 {nome}: alla cerniera spetta il resto")
        check(d.avanzamento == 0.0, f"§7 {nome}: a leva ferma l'avanzamento è nullo")
        d.ancora()                                   # ingresso in MOVE
        check(_avanza(d, th0, leva=+L / 2) > 0.0,
              f"§7 {nome}: girare la leva PAGA anche a cerniera ferma")
        check(abs(d.avanzamento - Q / 2) < 1e-6,
              f"§7 {nome}: mezza leva = metà della quota della leva")
        _avanza(d, th0, leva=+(L + 0.3))             # leva oltre lo sblocco
        check(abs(d.avanzamento - Q) < 1e-6,
              f"§7 {nome}: oltre lo sblocco la leva non paga più")
        _avanza(d, ths, leva=+(L + 0.3))             # cerniera al bersaglio
        check(abs(d.avanzamento - 1.0) < 1e-9,
              f"§7 {nome}: leva + cerniera = corsa completa")
        check(_avanza(d, th0, leva=0.0) == 0.0,
              f"§7 {nome}: tornare indietro non ripaga (cricchetto)")


def test_gradiente_cerniera_non_soffocato():
    """§7 — il gradiente sull'angolo della porta deve restare forte nei DUE compiti.

    È il numero che decide quale compito si risolve. Prima: 458 per radiante,
    con l'apertura ferma allo 0.21 e in calo. La cerniera deve prendersi la
    maggioranza del budget in entrambi i casi.
    """
    for task in ("close", "open"):
        cfg = UnifiedConfig.per(task)
        thr = cfg.thr
        for cerniera in (0.30, 0.39):
            d = DoorState(theta_star=cerniera, theta_zero=0.0, tol=cfg.task.tol)
            d.reset(0.0, cerniera, cfg.task.tol,
                    leva_rilascio=thr.leva_rilascio, quota_leva=thr.quota_leva)
            w = cfg.w.w_progress(d.escursione)
            check(w > 900.0,
                  f"§7 {task}: gradiente cerniera {w:.0f}/rad, era 458")
            check(abs(w * cerniera - cfg.w.budget_progress * (1 - thr.quota_leva)) < 1.0,
                  f"§7 {task}: alla cerniera va budget × (1 − quota_leva)")


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


def test_nessun_vicolo_cieco():
    """§3 — da MOVE e da HOLD si esce SEMPRE. Due deadlock misurati e chiusi.

    1. MOVE: con la porta al bersaglio e la mano nella finestra fra la soglia di
       conferma (0.035 m) e quella di perdita (0.05 m), il vecchio gate
       `A ∧ presa_ok` non scattava e nemmeno `presa_persa`: l'episodio restava
       in MOVE fino all'orizzonte con la porta GIA' al posto giusto.
    2. HOLD: con la guardia azzerata a ogni frame dentro tolleranza, una porta
       che oscilla dentro/fuori teneva timer e guardia entrambi a zero.
    """
    cfg, f = _fsm("close")
    d = DoorState(theta_star=0.006, theta_zero=0.35, tol=0.03)
    d.reset(0.35, 0.006, 0.03, cfg.thr.leva_rilascio)
    d.step(0.006)                                   # porta AL BERSAGLIO
    check(d.A, "la porta è al bersaglio")
    f.s.fase = Fase.MOVE
    # mano nella finestra morta: 0.042 m, oltre la conferma e sotto la perdita
    for _ in range(60):
        _passo(f, d, presa_ok=False, dist_mano=0.042, grip_cmd=1.0, soglia=0.75,
               contatto=True, hold_target=60)
        if f.s.fase != Fase.MOVE:
            break
    check(f.s.fase in (Fase.HOLD, Fase.RELEASE),
          "§3 al bersaglio si esce da MOVE: con le dita chiuse in HOLD, "
          "altrimenti in RELEASE dopo T_hold")
    # HOLD: porta che oscilla dentro/fuori tolleranza
    cfg, f = _fsm("close")
    d2 = DoorState(theta_star=0.006, theta_zero=0.35, tol=0.03)
    d2.reset(0.35, 0.006, 0.03, cfg.thr.leva_rilascio)
    f.s.fase = Fase.HOLD
    for i in range(4 * cfg.thr.stall_guard_steps):
        d2.step(0.006 if i % 2 == 0 else 0.20)      # dentro, fuori, dentro, fuori
        _passo(f, d2, hold_target=60)
        if f.s.fase != Fase.HOLD:
            break
    check(f.s.fase == Fase.RELEASE,
          "§3 una porta che oscilla non blocca HOLD: la guardia è cumulativa")


def test_da_ogni_stato_si_arriva_a_FINE():
    """§3 correzione 4 — PROVA ESAUSTIVA: nessuno stato senza uscita.

    Con la porta ferma AL BERSAGLIO si enumerano fase iniziale, comando del
    gripper, distanza della mano, contatto, angolo della leva e distanza dal
    punto di ritiro, e si verifica che in 600 passi si arrivi sempre a FINE.

    E' il test che avrebbe intercettato il difetto peggiore rimasto: con il
    comando dentro la banda d'isteresi (fra soglia − 0.20 e soglia) e la mano
    entro la tolleranza di perdita, MOVE non poteva ne' entrare in HOLD ne'
    tornare in REACH. Misurato prima della correzione: **128 combinazioni su
    576** non terminavano mai, e nella valutazione della chiusura erano 3
    episodi su 20 a 600 passi con errore −0.0060, cioe' a compito gia' compiuto.
    """
    for task, ths, th0, tol in (("close", 0.006, 0.35, 0.03),
                                ("open", 0.37, 0.0, 0.05)):
        cfg = UnifiedConfig.per(task); thr = cfg.thr; soglia = 0.70
        morte = tot = trappole = 0
        for fase0 in (Fase.REACH, Fase.MOVE, Fase.HOLD, Fase.RELEASE):
          for grip in (-1.0, 0.0, 0.55, 0.65, 0.80, 1.0):
            for dist in (0.02, 0.042, 0.06, 0.20):
              for cont in (True, False):
                for latch in (0.0, 0.30, 1.4):
                  for dret in (0.03, 0.30):
                    tot += 1
                    f = UnifiedFSM(cfg); f.s.fase = fase0
                    d = DoorState(ths, th0, tol)
                    d.reset(th0, ths, tol, thr.leva_rilascio); d.step(ths)
                    senza_presa = 0
                    for _ in range(600):
                        prima = f.s.fase
                        f.step(door=d,
                               presa_ok=(dist <= 0.035 and grip >= soglia and cont),
                               contatto=cont, dist_mano=dist, latch=latch,
                               hold_target=60, grip_cmd=grip, soglia=soglia,
                               door_qvel=0.0, dist_ritiro=dret)
                        if prima != Fase.HOLD and f.s.fase == Fase.HOLD and not cont:
                            senza_presa += 1
                        if f.s.fase == Fase.FINE:
                            break
                    else:
                        morte += 1
                    trappole += senza_presa
        check(morte == 0, f"§3 {task}: {morte} stati su {tot} non arrivano mai a FINE")
        check(trappole == 0,
              f"§3 {task}: {trappole} ingressi in HOLD SENZA dita chiuse. HOLD "
              f"congela il braccio e paga `hold_grip` −5: entrarci a mano libera "
              f"e' una trappola da −6.21 per passo per 62 passi, misurata")
        check(tot == 1152, f"§3 {task}: la prova copre 1152 combinazioni, non {tot}")


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
    # §5 bis — in questa variante ENTRAMBI i compiti accendono i controllori
    for t in ("close", "open"):
        st = UnifiedConfig.per(t).task
        check(st.riporto_leva is True, f"§5 bis {t}: riporto leva acceso")
        check(st.escape is True, f"§5 bis {t}: fuga accesa")
        check(st.orienta_normale is True, f"§5 bis {t}: normale orientata verso il robot")
        check(st.d_ret == 0.25, f"§5 bis {t}: corsa di fuga 0.25 m")


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


# ══ §1 i diciassette termini ═════════════════════════════════════════════
def test_diciassette_termini():
    check(len(TERMINI) == 17, f"§1 devono essere 17 termini, sono {len(TERMINI)}")
    check(set(ASSORBE) == set(TERMINI), "§5 ogni termine ha la sua mappatura")
    # copertura: 40 definizioni della chiusura + 25 dell'apertura, nessuna orfana
    sc = {k for k, v in SOPPRESSI.items() if v[0] == "chiusura"}
    so = {k for k, v in SOPPRESSI.items() if v[0] == "apertura"}
    c = {k for v in ASSORBE.values() for k in v[0]} | sc
    o = {k for v in ASSORBE.values() for k in v[1]} | so
    check(len(c) == 40, f"§5 devono essere 40 definizioni della chiusura, sono {len(c)}")
    check(len(o) == 25, f"§5 devono essere 25 definizioni dell'apertura, sono {len(o)}")
    check(len(c | o) == 41, f"§5 unione = 41, è {len(c | o)}")
    check(set(SOPPRESSI) == {"door_regress"}, "§5 una sola definizione soppressa")


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
        Fase.HOLD:    {"time", "smooth", "phi", "wrist", "target", "damp", "still", "hold_grip"},
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


def test_verso_della_leva():
    """§4 — la leva paga SOLO nel verso che abbassa la maniglia."""
    for nome, th0, ths in (("close", 0.35, 0.006), ("open", 0.0, 0.35)):
        c = UnifiedConfig.per(nome); thr = c.thr
        giu = DoorState(theta_star=ths, theta_zero=th0, tol=c.task.tol)
        su = DoorState(theta_star=ths, theta_zero=th0, tol=c.task.tol)
        for d in (giu, su):
            d.reset(th0, ths, c.task.tol, leva_rilascio=thr.leva_rilascio,
                    quota_leva=thr.quota_leva, verso_leva=thr.verso_leva)
        giu.step(th0, +thr.leva_rilascio)     # maniglia abbassata
        su.step(th0, -thr.leva_rilascio)      # maniglia alzata
        check(su.avanzamento == 0.0, f"§4 {nome}: alzare la maniglia non paga nulla")
        check(abs(giu.avanzamento - thr.quota_leva) < 1e-6,
              f"§4 {nome}: abbassarla vale esattamente la quota della leva")
        check(giu.avanzamento > su.avanzamento, f"§4 {nome}: il verso giusto avanza di piu'")


def test_progress_normalizzato():
    """§7 — percorrere l'escursione paga lo stesso budget nei due compiti."""
    w = Weights()
    for escursione in (0.35, 0.37):
        check(abs(w.w_progress(escursione) * escursione - w.budget_progress) < 1e-6,
              f"§7 budget costante per escursione {escursione}")
    # oggi i due budget divergono di 6x: 2000·0.35 = 700 contro 300·0.37 ≈ 111
    check(abs(2000 * 0.35 - w.budget_progress) < 1.0,
          "§7 il budget adottato è quello della chiusura (criterio §9)")


def test_meccanismo_di_arresto():
    """§1 termini 10-11 — rampa, rimbalzo e smorzamento: fermare la porta.

    Nella chiusura l'arresto lo produce la fisica (il pannello contro il
    telaio); nell'apertura il bersaglio e' un punto interno e la cerniera non
    ha molla di richiamo, quindi va scritto nella ricompensa. Tesi §6.4.1,
    classe F: «sono i due termini che fermano la porta sul bersaglio».
    """
    cfg = UnifiedConfig.per("open")
    f, rw = UnifiedFSM(cfg), UnifiedReward(cfg)
    d = DoorState(theta_star=0.37, theta_zero=0.0, tol=0.05); d.reset(0.0, 0.37, 0.05)
    kw = dict(fsm=f, door=d, action=np.zeros(7), delta_p=0.0, dist_mano=0.02,
              dist_xy=0.0, dz=0.0, allineamento=1.0, polso_piatto=0.0, grip_cmd=0.9,
              soglia_presa=0.7, contatto=True, latch=0.0, vel_giunti=0.0, raggio=0.02,
              dist_ritiro=0.1, dir_ritiro=None, hold_target=30, terminato=False)
    f.s.fase = Fase.HOLD
    # 1 — la rampa: piu' si e' precisi, piu' si paga
    d.step(0.37); centro = rw.compute(door_qvel=0.0, **kw)["target"]
    rw.reset(); d.step(0.34); bordo = rw.compute(door_qvel=0.0, **kw)["target"]
    check(centro > bordo > 0.0, "§1.10 dentro la tolleranza la rampa indica il CENTRO")
    # 2 — il rimbalzo: fuori tolleranza si paga, e cresce con l'errore
    rw.reset(); d.step(0.45); fuori = rw.compute(door_qvel=0.0, **kw)["target"]
    rw.reset(); d.step(0.50); piu_fuori = rw.compute(door_qvel=0.0, **kw)["target"]
    check(fuori < 0.0 and piu_fuori < fuori, "§1.10 fuori tolleranza `target` PUNISCE")
    # 3 — lo smorzamento: solo in HOLD, e zero all'ottimo
    rw.reset(); d.step(0.37)
    fermo = rw.compute(door_qvel=0.0, **kw)
    check("damp" not in fermo, "§1.11 a porta ferma `damp` non interviene")
    rw.reset(); moto = rw.compute(door_qvel=0.2, **kw)
    check(moto["damp"] < 0.0, "§1.11 la porta in movimento sul bersaglio e' penalizzata")
    rw.reset(); f.s.fase = Fase.RELEASE
    check("damp" not in rw.compute(door_qvel=0.2, **kw), "§1.11 `damp` vive solo in HOLD")
    # in RELEASE, fuori tolleranza, `target` non punisce: e' `hold` del ramo RETREAT
    rw.reset(); d.step(0.45)
    check("target" not in rw.compute(door_qvel=0.0, **kw),
          "§1.10 in RELEASE fuori tolleranza `target` vale zero, non punisce")


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


def test_soglia_di_sblocco_contro_il_simulatore():
    """§7 — la costante `leva_rilascio` verificata sulla FISICA, non sulla carta.

    E' il controllo che avrebbe intercettato la regressione peggiore di questo
    progetto: con `leva_rilascio` = 0 sulla chiusura la porta si fermava a
    0.182 rad in 15 episodi su 20 di valutazione, perche' il chiavistello
    aggancia il montante e la macchina non pagava il gesto che lo libera.

    Si applica coppia alla cerniera con la leva TENUTA a un angolo fissato e si
    guarda dove la porta si ferma. Il test e' lento (costruisce il simulatore),
    ma e' l'unico che lega un numero della configurazione a una misura fisica.
    """
    import os
    os.environ.setdefault("MUJOCO_GL", "disabled")
    from env_unified import UnifiedDoorEnv
    cfg = UnifiedConfig.per("close")
    env = UnifiedDoorEnv(cfg)
    rs = env._rs; sim = rs.sim
    hin, lat = rs.hinge_qpos_addr, rs.handle_qpos_addr
    dof = int(sim.model.jnt_dofadr[sim.model.jnt_qposadr.tolist().index(hin)])

    def theta_finale(theta0, leva, coppia, passi=250):
        sim.data.qpos[:] = 0.0; sim.data.qvel[:] = 0.0
        sim.data.qpos[hin] = theta0; sim.data.qpos[lat] = leva
        sim.forward()
        for _ in range(passi):
            sim.data.qpos[lat] = leva          # leva tenuta ferma
            sim.data.qvel[lat] = 0.0
            sim.data.qfrc_applied[dof] = coppia
            sim.step()
        sim.data.qfrc_applied[:] = 0.0
        return float(sim.data.qpos[hin])

    L = cfg.thr.leva_rilascio
    # 1 — a leva ferma il chiavistello blocca la porta in ENTRAMBI i versi
    check(theta_finale(0.36, 0.0, -2.0) > 0.15,
          "§7 a leva ferma la chiusura si blocca (montante del chiavistello)")
    check(theta_finale(0.0, 0.0, +2.0) < 0.05,
          "§7 a leva ferma l'apertura si blocca (stesso montante)")
    # 2 — oltre la soglia dichiarata la corsa e' libera, in entrambi i versi
    check(theta_finale(0.36, -L, -2.0) < 0.03,
          f"§7 con |leva| = {L} la chiusura arriva al bersaglio")
    check(theta_finale(0.0, -L, +2.0) > 0.30,
          f"§7 con |leva| = {L} l'apertura arriva al bersaglio")
    # 3 — la soglia non e' sovrastimata: appena sotto, la porta e' ancora bloccata
    check(theta_finale(0.36, -(L - 0.15), -2.0) > 0.10,
          "§7 appena sotto la soglia la chiusura e' ancora bloccata")
    env.close()


def test_budget_e_modello_migliore():
    """§9 — budget 1.5·10⁶ e salvataggio dell'ISTANTANEA MIGLIORE.

    Il progetto dell'apertura dichiara 1 500 000 passi e salva il migliore a
    650 000: il finale e' peggiore. Salvare solo l'ultimo butta via il picco.
    """
    import inspect
    import train_unified as T
    src = inspect.getsource(T)
    for a in ("--task", "--total-steps", "--eval-ogni", "--eval-episodi"):
        check(a in src, f"§9 l'opzione {a} esiste")
    check("default=1_500_000" in src, "§9 il budget predefinito è 1.5·10⁶")
    check("_crea_callback_migliore" in src, "§9 esiste il callback dell'istantanea migliore")
    check('save(os.path.join(args.run_dir, "best_model"))' in src
          and 'save(os.path.join(args.run_dir, "final_model"))' in src,
          "§9 si salvano DUE file: la migliore e la finale")
    check("sync_envs_normalization" in src,
          "§9 la statistica di VecNormalize è sincronizzata con l'ambiente di valutazione")
    check('env_addestramento.save(os.path.join(args.run_dir, "vecnormalize.pkl"))' in src,
          "§9 la normalizzazione è salvata INSIEME al modello migliore")


def test_tre_livelli_di_successo():
    """§9 — i tre livelli della tesi (§6.3.4), non uno solo.

    permissivo  la politica ha raggiunto la fase di mantenimento
    true        a fine episodio la porta e' al bersaglio E la leva e' a riposo
    clean       in piu', il ritiro e' stato effettivamente completato

    `is_success` DEVE essere il true success: e' la metrica con cui la baseline
    si misura (apertura 83.0 %, chiusura 100 %). Fermarsi a `terminato ∧ A`
    sarebbe piu' permissivo — non guarderebbe la leva — e i numeri non sarebbero
    confrontabili con quelli della tesi.
    """
    import inspect
    import env_unified, train_unified
    src = inspect.getsource(env_unified.UnifiedDoorEnv.step)
    check("successo_permissivo" in src and "successo_pulito" in src,
          "§9 l'ambiente pubblica tutti e tre i livelli")
    check("abs(m[\"latch\"]) <= self.cfg.thr.latch_term_tol" in src,
          "§9 il true success richiede la leva tornata a riposo")
    check("self._ritiro_spostato >= self.cfg.thr.ritiro_spostamento_min" in src,
          "§9 il clean success richiede il ritiro completato")
    m = inspect.getsource(train_unified.metriche)
    for k in ("successo_permissivo", "success_rate", "successo_pulito"):
        check(k in m, f"§9 le metriche riportano {k}")
    # ordinamento logico: clean ⊆ true ⊆ permissivo
    for perm, true_, clean in ((True, True, True), (True, True, False),
                               (True, False, False), (False, False, False)):
        check(not clean or true_, "§9 clean implica true")
        check(not true_ or perm, "§9 true implica permissivo")


def test_osservazione_126():
    from env_unified import OBS_DIM, OBS_BLOCCHI
    check(OBS_DIM == 126, "§6 l'osservazione unificata ha 126 dimensioni")
    check(sum(n for _, n in OBS_BLOCCHI) == 126, "§6 i blocchi sommano a 126")



def test_clean_success_come_nel_progetto_originale():
    """§7 — il clean success usa la definizione del progetto originale.

    `scratch/test_open_task_v2/_common.py`, riga 382:

        clean_success = true_success
                        and retreat_moved_max >= STUCK_MOVE_THRESH   # 0.06 m
                        and term_type == "PULITA"

    dove `retreat_moved` (open_generalized_v2/env_v2.py, riga 542) vale
    ‖eef − eef_al_rilascio‖: **di quanto la mano si è allontanata** da dove ha
    lasciato la maniglia, tenuto al MASSIMO sull'episodio.
    """
    import inspect, env_unified, fsm_unified
    src = inspect.getsource(env_unified.UnifiedDoorEnv.step)
    check("self._ritiro_spostato >= self.cfg.thr.ritiro_spostamento_min" in src,
          "§7 il criterio è lo spostamento, non la distanza da una posa")
    check("self.fsm.s.uscita_pulita" in src,
          "§7 il clean success richiede una terminazione PULITA")
    letto = inspect.getsource(env_unified.UnifiedDoorEnv._leggi)
    check("np.linalg.norm(self._eef - self._eef_rilascio)" in letto,
          "§7 lo spostamento si misura dalla posa di rilascio")
    check("self._ritiro_spostato = max(" in letto,
          "§7 si tiene il massimo sull'episodio, come l'originale")
    gate = inspect.getsource(fsm_unified.UnifiedFSM.step)
    check("s.uscita_pulita = bool(pronto)" in gate,
          "§7 l'uscita è pulita solo se per condizione, non per tetto duro")
    for task in ("close", "open"):
        check(UnifiedConfig.per(task).thr.ritiro_spostamento_min == 0.06,
              f"§7 {task}: la soglia è quella dell'originale (0.06 m)")


def test_il_gate_e_la_ricompensa_non_sono_toccati():
    """§7 — il clean success è una MISURA, non un vincolo.

    Correggerlo non deve poter cambiare il true success: il gate resta
    `passi >= N ∧ A ∧ leva`, senza condizioni sul ritiro, e la funzione di
    ricompensa è identica a quella dei modelli valutati.
    """
    import inspect
    from fsm_unified import UnifiedFSM
    gate = inspect.getsource(UnifiedFSM.step)
    ramo = gate.split("pronto = ")[1].split("if pronto")[0]
    check("dist_ritiro" not in ramo, "§7 il gate non guarda il ritiro")
    check("door.A and abs(latch) <= self.thr.latch_term_tol" in ramo,
          "§7 il gate chiede A e la leva, nient'altro")


def test_valutazione_su_piu_semi():
    """§8 — la valutazione può girare su più semi e unire i campioni.

    Un solo seme fissa quali porte vengono provate. Ripetere su semi diversi
    distingue «la politica funziona» da «il campione era fortunato», e l'unione
    degli episodi restringe l'intervallo di confidenza.
    """
    import inspect, train_unified
    src = inspect.getsource(train_unified)
    check("--eval-seeds" in src, "§8 esiste l'opzione per più semi")
    check("TUTTI I SEMI UNITI" in src, "§8 i campioni vengono uniti")
    check("def wilson(" in src, "§8 l'intervallo di Wilson è calcolato")
    m = inspect.getsource(train_unified.metriche)
    for k in ("true_IC95", "pulito_IC95", "ritiro_spostamento_medio"):
        check(k in m, f"§8 le metriche riportano {k}")
    # lo stimatore deve coincidere con quello della suite originale
    for k, n, atteso in ((200, 200, (0.981, 1.000)), (196, 200, (0.950, 0.992))):
        lo, hi = train_unified.wilson(k, n)
        check(abs(lo - atteso[0]) < 0.002 and abs(hi - atteso[1]) < 0.002,
              f"§8 Wilson({k}/{n}) = [{lo:.3f}, {hi:.3f}]")

if __name__ == "__main__":
    print("Verifica dell'implementazione contro tabella_reward_machine_unificata.md\n")
    for nome, fn in sorted(list(globals().items())):
        if nome.startswith("test_"):
            fn()
    print(f"\n{OK + KO} controlli · {OK} superati · {KO} falliti")
    sys.exit(1 if KO else 0)
