#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
reward_unified.py — i 16 termini della reward machine unificata.

Documento di riferimento: tabella_reward_machine_unificata.md
  §1  la tabella dei 16 termini e la nota «perché ogni termine è fatto così»
  §7    i due termini che cambiano: target, progress
  §8  la maschera dei controllori
  §5  ret_lat e ret_down non vanno replicati

I 16 termini assorbono le definizioni dei due sorgenti. Ne spariscono due:
`door_regress` e `hold_veldamp` — vedi SOPPRESSI.

Letteratura, invariata rispetto ai due progetti (§6):
  [Ng, Russell & Harada 1999]   shaping potenziale: F = γΦ(s′) − Φ(s)
  [Devlin & Kudenko 2012]       shaping dipendente dal tempo/fase
  [Sutton, Precup & Singh 1999] opzioni: una fase = un'opzione
  [Konidaris & Barto 2009]      soglie adattive
  [ManipForce 2015]             primitive basate sulla forza di presa
  [Tobin et al. 2017]           domain randomization, parametri in osservazione
  [ten Pas et al. 2017]         feature di presa multi-approccio
  [Krakovna et al. 2020]        specification gaming: il cricchetto e la
                                saturazione esistono per chiudere due scorciatoie
"""
from typing import Dict, Optional
import numpy as np

from fsm_unified import Fase


# elenco canonico: l'ordine è quello della tabella §1 del documento
TERMINI = ("time", "smooth", "phi", "approach", "approach_geom", "wrist", "grip",
           "progress", "contact", "target", "still", "hold_grip",
           "release", "retreat", "latch_home", "success")

# mappa termine unificato -> definizioni originali assorbite (§5 del documento)
ASSORBE = {
    "time":          (["base"], ["base"]),
    "smooth":        (["smoothness", "act_pen"], []),
    "phi":           (["phi_shape"], ["phi_shape"]),
    "approach":      (["dist_3d", "dist_xy", "dist_z"], ["dist_3d", "dist_xy", "dist_z"]),
    "approach_geom": (["app_blw", "app_top"], ["app_blw", "app_top"]),
    "wrist":         (["align", "flat", "hold_flat"], []),
    "grip":          (["grip", "grip_lost", "dist_lost", "grasp_lost_pen"], ["grip"]),
    "progress":      (["door_prog"], ["door_prog"]),
    "contact":       (["grip_contact", "lift_pen"], ["grip_contact"]),
    "target":        (["hold", "hold_bounce"], ["hold"]),
    "still":         (["hold_act", "hold_jnt_freeze", "ret_freeze", "ret_jnt_prog"],
                      ["hold_act", "ret_freeze"]),
    "hold_grip":     (["hold_grip", "hold_slip", "hold_drop_pen", "hold_dist"],
                      ["hold_grip", "hold_slip", "hold_drop_pen", "hold_dist"]),
    "release":       (["ret_grip", "ret_release"], ["ret_grip", "ret_release"]),
    "retreat":       (["ret_dir", "ret_perp", "ret_rot", "ret_lat", "ret_down"],
                      ["ret_dir", "ret_perp", "ret_rot"]),
    "latch_home":    (["latch_ret"], ["latch_ret"]),
    "success":       (["success_bonus", "phase_trans"], ["success_bonus"]),
}
# definizione soppressa -> (sorgente in cui esiste, perche' sparisce)
SOPPRESSI = {
    "door_regress": ("apertura",
                     "ridondante: se la porta si allontana dal bersaglio `target` cala "
                     "gia' da solo"),
    "hold_veldamp": ("chiusura",
                     "il sorgente dell'apertura lo ha respinto per iscritto: al suo goal "
                     "la molla ritira la porta di 0.024-0.050 rad in modo inevitabile, "
                     "quindi punire |dθ/dt| punisce la fisica. Misurato: -1.53 per passo "
                     "in HOLD sull'apertura, -0.05 sulla chiusura, dove e' inattivo "
                     "perche' il bersaglio e' il punto di equilibrio"),
}


class Potenziale:
    """§1, termine `phi` — shaping potenziale [Ng, Russell & Harada 1999].

    Φ è CUMULATIVO: ogni fase eredita il potenziale delle precedenti e vi
    aggiunge il proprio. Serve a garantire la continuità di F alle transizioni
    [Devlin & Kudenko 2012]. La forma è invariata rispetto ai due progetti; i
    pesi sono riportati alla stessa scala (§1, nota su `phi`).
    """

    def __init__(self, cfg):
        self.cfg, self.w = cfg, cfg.w
        self.prev: Optional[float] = None

    def reset(self) -> None:
        self.prev = None

    def _phi(self, fase: Fase, door, timer: float, hold_target: int,
             dist_ritiro: float, dist_mano: float, grip_cmd: float,
             raggio: float) -> float:
        w, lvl = self.w, self.cfg.curriculum_level
        k = 1.0 + w.curriculum_k * lvl

        # ── IL PREMIO DI PRESA VIVE QUI, NON IN R ────────────────────────
        # Nei due sorgenti «chiudere le dita sulla maniglia» e' un termine di
        # ricompensa da +2.5·norm(g) A OGNI PASSO. E' l'unico gradiente che
        # insegna ad afferrare, ma REACH non ha tetto di durata: misurato, stare
        # sulla maniglia con il comando appena SOTTO la soglia di conferma vale
        # +996 su 600 passi, contro i ~1150 della soluzione completa.
        # Lo stesso termine dentro Φ conserva ESATTAMENTE il gradiente e
        # l'integrale — stringere di piu' paga subito — ma a comando costante
        # vale F = γΦ − Φ = −(1−γ)Φ < 0, quindi non e' piu' mungibile.
        # [Ng, Russell & Harada 1999]: un potenziale non sposta la politica
        # ottima, quindi il compito resta lo stesso.
        vicino = dist_mano <= 1.5 * raggio + 0.005
        if fase == Fase.REACH:
            phi_grip = (w.grip_near * float(np.clip((grip_cmd + 0.85) / 1.85, 0.0, 1.0))
                        if vicino else 0.0)
        else:
            phi_grip = w.grip_near        # fase superata: al massimo, come le altre

        # ── IN REACH IL POTENZIALE E' ZERO ───────────────────────────────
        # Riga per riga dai due sorgenti (close reward_v2.py ~riga 231,
        # open reward_v2.py ~riga 113):  `if phase == REACH: phi_now = 0.0`.
        # Il peso phi_reach NON e' una costante presente ovunque: e' il primo
        # GRADINO della scala, che si acquisisce USCENDO da REACH. Il commento
        # del config originale lo dice: «sets the REACH->PUSH shaping bonus /
        # ladder base».
        #
        # Contandolo anche in REACH — come faceva questa implementazione — si
        # producono due danni insieme:
        #   1. sparisce il gradino: la transizione REACH->MOVE valeva
        #      γ·Φ_reach − Φ_reach = −0.05·Φ_reach, cioe' una PENALITA' di
        #      −0.15, dove gli originali pagano γ·Φ_reach = +2.85;
        #   2. resta la deriva −(1−γ)·Φ di ogni potenziale costante, che in
        #      MOVE (Φ piu' alto) e' PIU' severa che in REACH: la macchina
        #      tassava di piu' la fase piu' avanzata.
        # E' la spiegazione dei diari: REACH +0.35/passo, MOVE −0.91/passo.
        if fase == Fase.REACH:
            return phi_grip

        phi = phi_grip + w.phi_reach * k          # gradino acquisito
        if fase == Fase.MOVE:
            # ANGOLO CORRENTE, non il cricchetto: e' l'unico gradiente denso che
            # dice al braccio in che verso spingere, e deve poter SCENDERE se la
            # porta torna indietro. Vedi DoorState.avanzamento.
            phi += w.phi_move * k * door.avanzamento
        else:
            # fase superata: il suo potenziale entra al MASSIMO, come negli
            # originali (`w_reach_eff + w_push_eff + ...`), cosi' la scala non
            # puo' scendere passando alla fase successiva.
            phi += w.phi_move * k
        if fase == Fase.HOLD:
            t = float(np.clip(timer / max(hold_target, 1), 0.0, 1.0))
            phi += w.phi_hold * t * door.target_ramp
        elif fase > Fase.HOLD:
            phi += w.phi_hold
            phi += w.phi_release * float(np.clip(1.0 - dist_ritiro / 0.20, 0.0, 1.0))
        return phi

    def shaping(self, **kw) -> float:
        """F = γΦ(s′) − Φ(s), tagliato come nella chiusura (±10)."""
        now = self._phi(**kw)
        if self.prev is None:
            self.prev = now
            return 0.0
        f = self.cfg.sac.gamma * now - self.prev
        self.prev = now
        return float(np.clip(f, -self.w.phi_clip, self.w.phi_clip))


class UnifiedReward:
    """La reward machine. Un solo `compute`, valido per i due compiti."""

    def __init__(self, cfg):
        self.cfg, self.w = cfg, cfg.w
        self.phi = Potenziale(cfg)
        self.prev_action: Optional[np.ndarray] = None
        self.success_pagato = False
        self.transizione_pagata = False

    def reset(self) -> None:
        self.phi.reset()
        self.prev_action = None
        self.success_pagato = False
        self.transizione_pagata = False

    # ═════════════════════════════════════════════════════════════════════
    def compute(self, *, fsm, door, action: np.ndarray, delta_p: float,
                dist_mano: float, dist_xy: float, dz: float,
                allineamento: float, polso_piatto: float,
                grip_cmd: float, soglia_presa: float, contatto: bool,
                door_qvel: float, latch: float,
                vel_giunti: float, raggio: float,
                dist_ritiro: float, dir_ritiro: Optional[np.ndarray],
                hold_target: int, terminato: bool) -> Dict[str, float]:
        """Restituisce il dizionario dei termini attivi in questo step.

        L'ordine di lettura è quello della tabella §1.
        """
        w, s = self.w, fsm.s
        fase = s.fase
        r: Dict[str, float] = {}
        a_arm = np.asarray(action[:-1], dtype=np.float64)

        # ── 1 time — sempre ─────────────────────────────────────────────
        # §1: si adotta la forma dell'apertura (costante) invece della
        # ricompensa ereditata da train_close.py, che nascondeva i bonus.
        r["time"] = -w.time_cost

        # ── 2 smooth — sempre ───────────────────────────────────────────
        if self.prev_action is not None:
            r["smooth"] = (-w.smooth * float(np.linalg.norm(a_arm - self.prev_action[:-1]))
                           - w.smooth_act * float(np.linalg.norm(a_arm)))
        self.prev_action = np.asarray(action, dtype=np.float64).copy()

        # ── 3 phi — sempre ──────────────────────────────────────────────
        r["phi"] = self.phi.shaping(fase=fase, door=door, timer=s.timer_hold,
                                    hold_target=hold_target, dist_ritiro=dist_ritiro,
                                    dist_mano=dist_mano, grip_cmd=grip_cmd, raggio=raggio)

        # ── 4 approach — REACH, MOVE ────────────────────────────────────
        # CORREZIONE al §1, motivata dal sorgente originale.
        # Il documento dice che `dist_3d` e' ridondante perche' «e' la norma
        # delle altre due». E' falso: sqrt(dxy^2 + dz^2) non e' 3*dxy + 15*|dz|.
        # Il commento degli originali avverte esattamente contro questa
        # semplificazione: in REACH il potenziale non da' gradiente, quindi
        # questi termini densi sono l'UNICO segnale di avvicinamento e devono
        # restare forti «otherwise the arm stalls at mid-distance» — che e'
        # proprio il fallimento osservato (successo 0, maniglia mai raggiunta).
        # Si ripristina anche il fattore di curriculum k, presente in entrambi.
        # In REACH i tre termini densi al completo, scalati dal curriculum.
        # In MOVE gli originali usano una forma PIU' LEGGERA — solo d3d e dz,
        # senza il fattore di curriculum (close reward_v2.py righe 307-308,
        # open righe 156-158): li' serve solo a «mantenere la mano sulla
        # maniglia mentre si muove la porta», non ad avvicinarsi. Applicarvi
        # la forma piena rendeva MOVE piu' cara del dovuto di ~1.7x, cioe'
        # esattamente la fase che deve convenire.
        if fase == Fase.REACH:
            k = 1.0 + w.curriculum_k * self.cfg.curriculum_level
            r["approach"] = -k * (w.app_3d * dist_mano + w.app_xy * dist_xy
                                  + w.app_z * abs(dz))
        elif fase == Fase.MOVE:
            r["approach"] = -(w.app_3d * dist_mano + w.app_z * abs(dz))

        # ── 5 approach_geom — REACH ─────────────────────────────────────
        if fase == Fase.REACH:
            lo, hi = w.geom_band
            if dz < lo:                       # da sotto: la peggiore
                r["approach_geom"] = -w.geom_below * abs(dz - lo)
            elif dz > hi:                     # troppo alto
                r["approach_geom"] = -w.geom_above * (dz - hi)

        # ── 6 wrist — REACH, HOLD ───────────────────────────────────────
        # un solo termine con peso dipendente dalla fase, al posto di tre
        if fase in (Fase.REACH, Fase.HOLD):
            prox = float(np.exp(-10.0 * dist_mano))
            peso = w.wrist_reach if fase == Fase.REACH else w.wrist_hold
            r["wrist"] = (-peso * (1.0 - allineamento) * prox
                          - w.wrist_flat * polso_piatto * prox)

        # ── 7 grip — REACH, MOVE ────────────────────────────────────────
        # Qui resta SOLO la penalita' per chiudere le dita lontano dalla
        # maniglia. Il ramo positivo (+2.5·norm(g) sulla maniglia, in entrambi i
        # sorgenti) e' un POTENZIALE, dentro `phi`: stesso gradiente e stesso
        # integrale, ma non mungibile. Rimetterlo in R e' stato misurato:
        # accamparsi in REACH con il comando appena sotto la soglia di conferma
        # vale +996, contro i ~1150 della soluzione completa — un margine del
        # 15 % su una condotta che l'esplorazione trova per prima.
        # La guardia `grip_cmd > -0.85` e' degli originali: senza, a gripper
        # spalancato il ramo lontano diventerebbe POSITIVO (+0.15/passo).
        if fase == Fase.REACH and grip_cmd > -0.85:
            if dist_mano > 1.5 * raggio + 0.005:
                r["grip"] = -w.grip_far * (grip_cmd + 0.85)      # non chiudere in anticipo
        elif fase == Fase.MOVE and grip_cmd < soglia_presa:
            r["grip"] = -w.grip_weak_move * (soglia_presa - grip_cmd)

        # ── 8 progress — MOVE ───────────────────────────────────────────
        # §7 il peso è normalizzato sull'escursione: percorrerla tutta paga
        # `budget_progress` in entrambi i compiti. delta_p arriva già a
        # cricchetto e saturato da DoorState.step().
        # Il gate sul gripper vive dove deve, cioe' insieme al cricchetto
        # (DoorState.incassa): delta_p arriva gia' nullo se la presa non e'
        # comandata sopra la soglia adattiva.
        if fase == Fase.MOVE and delta_p > 0.0:
            r["progress"] = w.w_progress(door.escursione) * delta_p

        # ── 9 contact — MOVE ────────────────────────────────────────────
        # `grip_contact` degli originali: premia il contatto fisico DURANTE IL
        # MOTO, scalato sul progresso. Il gate su delta_p e' quel «durante il
        # moto»: MOVE non ha tetto di durata, quindi un termine positivo pagato
        # a porta ferma sarebbe una rendita, ed e' stata misurata (+0.22 per
        # passo su 500 passi di MOVE immobile). Muovendo la porta delta_p e'
        # positivo quasi a ogni passo, quindi la soluzione vera non perde nulla.
        if fase == Fase.MOVE and contatto and delta_p > 0.0:
            r["contact"] = w.contact * door.avanzamento

        # ── 10 target — HOLD, RELEASE ───────────────────────────────────
        # §1 IL TERMINE CHE CAMBIA TUTTO: una rampa al posto di un premio
        # piatto. Sostituisce sia `1 − |θ|` (chiusura) sia `1` (apertura), e il
        # ramo negativo di hold_bounce diventa superfluo perché è lo stesso
        # termine che scende a zero.
        if fase in (Fase.HOLD, Fase.RELEASE):
            r["target"] = w.target * door.target_ramp

        # ── 12 still — HOLD, RELEASE ────────────────────────────────────
        # ATTENZIONE alla differenza fra le due fasi, che NON e' simmetrica:
        #   HOLD    -> `hold_act` originale: +1 se fermo, altrimenti penalita'
        #   RELEASE -> `ret_freeze` originale: penalita' PURA, mai un bonus
        # Gli originali commentano esplicitamente il perche': un bonus di
        # immobilita' nel ritiro creerebbe l'incentivo ad "accamparsi" fermi
        # invece di completare il compito [Krakovna et al. 2020].
        if fase == Fase.HOLD:
            norm_a = float(np.linalg.norm(a_arm))
            if norm_a < w.still_eps:
                r["still"] = w.still_bonus
            else:
                r["still"] = -(w.still_act * norm_a + w.still_jnt * vel_giunti)
        elif fase == Fase.RELEASE and dist_ritiro <= self.cfg.thr.retreat_settle_dist:
            # `ret_freeze` si paga SOLO nella zona di settle, come negli
            # originali: fuori da quella zona il braccio deve muoversi verso la
            # posa di ritiro, ed e' `retreat` a guidarlo. Pagare i due termini
            # insieme, come faceva questa implementazione, chiedeva al braccio
            # di muoversi e di stare fermo nello stesso passo.
            norm_a = float(np.linalg.norm(a_arm))
            r["still"] = -(w.still_release * norm_a + w.still_jnt * vel_giunti)

        # ── 13 hold_grip — HOLD ─────────────────────────────────────────
        if fase == Fase.HOLD:
            v = w.hg_bonus if grip_cmd >= soglia_presa else -w.hg_loose * abs(grip_cmd - soglia_presa)
            if not contatto:
                v -= w.hg_slip
            if grip_cmd < 0.0:
                v -= w.hg_drop * abs(grip_cmd)
            if dist_mano > w.hg_dist_max:
                v -= w.hg_dist * (dist_mano - w.hg_dist_max)
            r["hold_grip"] = v

        # ── 14 release — RELEASE ────────────────────────────────────────
        if fase == Fase.RELEASE:
            mano_aperta = grip_cmd < -0.85
            v = w.rel_open if mano_aperta else -w.rel_closed * abs(grip_cmd + 1.0)
            # `ret_release`: rilascio pulito. Negli originali si paga solo nella
            # zona di settle, a bersaglio raggiunto e mano aperta — cosi' non
            # c'e' incentivo ad accamparsi immobili a compito non finito.
            if (mano_aperta and door.A
                    and dist_ritiro <= self.cfg.thr.retreat_settle_dist):
                v += w.rel_clean
            r["release"] = v

        # ── 15 retreat — RELEASE ────────────────────────────────────────
        # §5 ret_lat e ret_down NON sono replicati: penalizzavano assi del
        # mondo che nell'apertura coincidono con la via di fuga. La stessa
        # intenzione è espressa da ret_perp, nel sistema di riferimento giusto.
        # `ret_rot` si paga sempre in RELEASE; `ret_dir`/`ret_perp` solo FUORI
        # dalla zona di settle, dove il braccio deve ancora raggiungere la posa
        # di ritiro. Dentro la zona subentra `still` (ret_freeze).
        if fase == Fase.RELEASE:
            v = -w.ret_rot * float(np.linalg.norm(a_arm[3:6]))
            if dir_ritiro is not None and dist_ritiro > self.cfg.thr.retreat_settle_dist:
                u = np.asarray(dir_ritiro, dtype=np.float64)
                n = float(np.linalg.norm(u))
                if n > 1e-9:
                    u = u / n
                    lungo = float(np.dot(a_arm[:3], u))
                    perp = a_arm[:3] - lungo * u
                    v += (w.ret_dir * lungo
                          - w.ret_perp * float(np.linalg.norm(perp)))
            r["retreat"] = v

        # ── 16 latch_home — RELEASE ─────────────────────────────────────
        if fase == Fase.RELEASE:
            r["latch_home"] = -w.latch_home * abs(latch)

        # ── 17a transizione — una tantum all'uscita da REACH ────────────
        # `phase_trans` degli originali (+10). Si paga UNA SOLA VOLTA per
        # episodio: pagarlo a ogni transizione lo renderebbe una rendita per chi
        # oscilla fra REACH e MOVE senza mai muovere la porta.
        if s.prev_fase == Fase.REACH and fase == Fase.MOVE and not self.transizione_pagata:
            r["success"] = r.get("success", 0.0) + w.transizione
            self.transizione_pagata = True

        # ── 17 success — terminale ──────────────────────────────────────
        # Il payoff terminale della chiusura originale e' 500 + 100: i 500
        # stanno dentro `base` (train_close.py, ramo `terminated`), i 100 sono
        # `success_bonus`. Mappando `base` sul solo costo del tempo si erano
        # persi i 500, e la soluzione valeva 479 contro i 420 dell'accampamento:
        # un margine del 14 %, troppo sottile perche' la politica ci resti.
        if terminato and door.A and not self.success_pagato:
            r["success"] = w.success
            self.success_pagato = True

        # ── §8 LA MASCHERA ────────────────────────────────────────────
        # applicata qui, in un punto solo, DOPO il calcolo: così è leggibile
        # nei log quali termini sono stati sospesi e perché.
        for k in fsm.termini_mascherati():
            r.pop(k, None)

        return r

    def total(self, termini: Dict[str, float], terminato: bool = False) -> float:
        """Somma e taglio dello step (±100), TRANNE al passo terminale.

        Gli originali scrivono `if not terminated: reward = clip(...)`: il
        premio finale non va tagliato, altrimenti i 600 che valgono il compito
        arrivano alla policy come 100 e il bonus terminale smette di contare.
        """
        s = float(sum(termini.values()))
        return s if terminato else float(np.clip(s, -self.w.clip_step, self.w.clip_step))
