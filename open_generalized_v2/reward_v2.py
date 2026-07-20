#!/usr/bin/env python3
# open_generalized_v2/reward_v2.py
#
# PotentialBasedRewardOpen — reward a potenziale gerarchico per l'APERTURA generalizzata,
# SPECULARE a close_generalized_v2/reward_v2.py.
#
# Principio (Ng, Harada & Russell 1999, [3]): lo shaping è F = γ·Φ(s') − Φ(s), che NON
# altera la politica ottima (policy invariance). Il potenziale Φ cresce monotòno lungo le
# fasi REACH→PULL→HOLD_OPEN→RETREAT, così lo shaping "tira" verso il completamento del task
# senza introdurre ottimi spuri (lezione metodologica della v2: motion-quality → env-level,
# competenza del task → reward potential-based).
#
# Inversione chiusura ↔ apertura: il potenziale di fase PULL premia il progresso verso
# l'angolo-OBIETTIVO (apertura), non verso 0 (chiusura).
#
# Riferimenti: [3] Ng 1999 (shaping invariante), [13] ManipForce (contatto), [15] ten Pas.

from __future__ import annotations

import numpy as np
from typing import Dict, Tuple


class PotentialBasedRewardOpen:
    def __init__(self, cfg, gamma: float = 0.95):
        self.cfg = cfg
        self.gamma = float(gamma)
        self._prev_phi = 0.0
        self._max_door_angle = None   # ratchet di APERTURA (sale solo) per door_prog

    def reset(self):
        self._prev_phi = 0.0
        self._max_door_angle = None

    # ── Potenziali di fase ───────────────────────────────────────────────────────

    def phi_reach(self, dist_handle: float, handle_radius: float, curriculum_lvl: float) -> float:
        sigma = float(np.clip(handle_radius * 3.0, self.cfg.phi_reach_sigma * 0.25, self.cfg.phi_reach_sigma))
        w_eff = self.cfg.phi_reach_weight * (1.0 + self.cfg.curriculum_reward_k * curriculum_lvl)
        return float(w_eff * np.exp(-(dist_handle ** 2) / (2.0 * sigma ** 2)))

    def phi_pull(self, door_angle: float, goal_angle: float, door_min: float,
                 gripper_action: float, grip_thresh: float, curriculum_lvl: float) -> float:
        """
        Progresso di APERTURA verso il goal, in [0,1] (specchio di phi_push della chiusura).
        0 a porta tutta chiusa (door_min), 1 a porta = goal_angle.
        """
        denom = max(1e-6, goal_angle - door_min)
        progress = float(np.clip((door_angle - door_min) / denom, 0.0, 1.0))
        w_eff = self.cfg.phi_pull_weight * (1.0 + self.cfg.curriculum_reward_k * curriculum_lvl)
        return float(w_eff * progress)

    def phi_hold(self, hold_duration: int, target_steps: int, door_angle: float,
                 goal_angle: float, open_tol: float) -> float:
        time_frac = float(np.clip(hold_duration / max(1, target_steps), 0.0, 1.0))
        # quanto è vicino al goal (1 al goal, →0 lontano)
        open_frac = float(np.clip(1.0 - abs(goal_angle - door_angle) / max(open_tol * 5.0, 1e-6), 0.0, 1.0))
        return float(self.cfg.phi_hold_weight * time_frac * open_frac)

    def phi_retreat(self, dist_retreat: float) -> float:
        progress = float(np.clip(1.0 - dist_retreat / 0.20, 0.0, 1.0))
        return float(self.cfg.phi_retreat_weight * progress)

    # ── Compute ──────────────────────────────────────────────────────────────────

    def compute(
        self,
        *,
        fsm_state,
        phase_consts,            # (REACH, PULL, HOLD_OPEN, RETREAT)
        door_angle    : float,
        door_qvel     : float = 0.0,
        goal_angle    : float,
        door_min      : float,
        open_tol      : float,
        prev_angle    : float,
        gripper_action: float,
        grip_thresh   : float,
        dist_handle   : float,
        dist_xy       : float = None,
        height_diff   : float = None,
        dist_retreat  : float,
        eef_pos       : np.ndarray = None,
        target_steps  : int,
        curriculum_lvl: float,
        is_physically_closed: bool,
        action        : np.ndarray,
        latch_qpos    : float = 0.0,
        just_succeeded: bool,
        rs_done       : bool,
        step_count    : int,
        horizon       : int,
    ) -> Tuple[float, bool, bool, Dict[str, float]]:
        REACH, PULL, HOLD_OPEN, RETREAT = phase_consts
        rew: Dict[str, float] = {}
        terminated = False

        # base: piccolo time-penalty (specchio della chiusura)
        rew["base"] = -0.10

        # ── potential-based shaping (Ng 1999) ──
        w_reach = self.cfg.phi_reach_weight * (1.0 + self.cfg.curriculum_reward_k * curriculum_lvl)
        w_pull  = self.cfg.phi_pull_weight  * (1.0 + self.cfg.curriculum_reward_k * curriculum_lvl)
        w_hold  = self.cfg.phi_hold_weight
        ph = fsm_state.phase
        if ph == REACH:
            # §1.9.F (mirror chiusura): Phi_reach = 0 in REACH per AZZERARE la penalità di
            # discount-decay F=(gamma-1)*Phi ~ -0.05*Phi, che vicino alla maniglia diventa
            # una "barriera di potenziale" e impedisce alla policy di RESTARE sulla maniglia
            # il tempo necessario a confermare la presa (5 step). Gli offset cumulativi nelle
            # fasi successive restano → la transizione REACH→PULL dà un bonus di grasp
            # naturale ~ +gamma*w_reach e la perdita presa una penalità ~ -w_reach [Ng 1999].
            phi_now = 0.0
        elif ph == PULL:
            phi_now = w_reach + self.phi_pull(door_angle, goal_angle, door_min,
                                              gripper_action, grip_thresh, curriculum_lvl)
        elif ph == HOLD_OPEN:
            phi_now = w_reach + w_pull + self.phi_hold(
                fsm_state.hold_open_duration, target_steps, door_angle, goal_angle, open_tol)
        else:  # RETREAT
            phi_now = w_reach + w_pull + w_hold + self.phi_retreat(dist_retreat)
        rew["phi_shape"] = self.gamma * phi_now - self._prev_phi
        self._prev_phi = phi_now

        # ── termini per-fase (in R genuino, non shaping) ──
        # FASE REACH — termini DENSI di avvicinamento (mirror della chiusura v2 che
        # funziona). CRUCIALE: con lo shaping a potenziale cumulativo, in REACH il
        # gradiente di Φ è ~nullo; questi termini densi sono l'UNICO segnale che porta
        # il braccio alla maniglia. Senza, la policy resta a mezza distanza (success=0).
        # Rif.: close_generalized_v2/reward_v2.py §1.10.B; competenza-del-task → shaping [3].
        if ph == REACH:
            k = 1.0 + self.cfg.curriculum_reward_k * curriculum_lvl
            rew["dist_3d"] = -self.cfg.w_reach_dist_3d * k * dist_handle
            if dist_xy is not None:
                rew["dist_xy"] = -self.cfg.w_reach_dist_xy * k * dist_xy
            if height_diff is not None:
                rew["dist_z"] = -self.cfg.w_reach_dist_z * k * abs(height_diff)
                # geometria di avvicinamento: non passare sotto, non stare troppo sopra
                if height_diff < -0.005:
                    rew["app_blw"] = -self.cfg.w_reach_app_blw * abs(height_diff + 0.005)
                if height_diff > 0.03:
                    rew["app_top"] = -self.cfg.w_reach_app_top * height_diff
            # gestione gripper calibrata sulla soglia di presa adattiva:
            # lontano → tieni aperto; vicino → premia la chiusura
            d_near = self.cfg.fsm_grasp_dist_k_radius * 0.02 + self.cfg.fsm_grasp_dist_k_offset
            if dist_handle > d_near:
                if gripper_action > -0.85:
                    rew["grip"] = -1.0 * (gripper_action - (-0.85))
            else:
                if gripper_action > -0.85:
                    norm_g = (gripper_action - (-0.85)) / (1.0 - (-0.85))
                    rew["grip"] = self.cfg.w_reach_grip_near * norm_g

        if ph == PULL:
            # mantieni la maniglia mentre tiri (mirror dist_3d/dist_z della chiusura)
            rew["dist_3d"] = -self.cfg.w_pull_dist_3d * dist_handle
            if height_diff is not None:
                rew["dist_z"] = -self.cfg.w_pull_dist_z * abs(height_diff)

            # ── OBIETTIVO REALE: progresso di APERTURA con ratchet (mirror door_prog) ──
            # delta = nuovo angolo guadagnato verso il goal (door_angle che CRESCE).
            # _max_door_angle sale solo → oscillare avanti/indietro non ri-premia (anti-exploit).
            # È il segnale genuino R che definisce "apri la porta"; lo shaping Ng sopra
            # non ne sposta l'ottimo [Ng 1999]. Rif. close §1.10.C (door_prog).
            #
            # §1.31 — SATURAZIONE AL GOAL (opt-in, default OFF). Asimmetria con la chiusura:
            # lì door_prog spinge verso 0 = fine-corsa del giunto, quindi il progresso SATURA
            # al bersaglio per costruzione (non può superarlo). Qui invece premiava QUALSIASI
            # apertura fino a eff_max (il cap), incentivando a SUPERARE il goal e spingere fino
            # al cap. Per i goal bassi la porta finisce contro il cap e poi, al rilascio nel
            # RETREAT, deriva indietro (la molla la richiama) → open_error finale ~0.05. Con la
            # saturazione, il progresso premia solo fino a goal_angle (esatto specchio della
            # chiusura: progresso verso il bersaglio, non oltre) → la porta è guidata AL goal,
            # non oltre. Default OFF per preservare la baseline al 100% (§1.30): attivare per
            # l'A/B e confrontare l'open_error FINALE sui goal bassi.
            prog_angle = door_angle
            if getattr(self.cfg, "pull_progress_cap_at_goal", False):
                prog_angle = min(door_angle, goal_angle)
            if self._max_door_angle is None:
                self._max_door_angle = prog_angle
            if gripper_action > grip_thresh:
                delta = prog_angle - self._max_door_angle
                if delta > 0:
                    rew["door_prog"] = self.cfg.w_pull_progress * delta
                    self._max_door_angle = prog_angle

            # presa genuinamente debole durante il PULL (dolce, come §1.13 chiusura)
            if gripper_action < grip_thresh:
                rew["grip"] = -self.cfg.w_pull_grip_weak * (grip_thresh - gripper_action)

            # mantenimento contatto durante l'apertura, scalato sul progresso (§1.16 specchio)
            if is_physically_closed:
                denom = max(1e-6, goal_angle - door_min)
                opening_progress = float(np.clip((door_angle - door_min) / denom, 0.0, 1.0))
                rew["grip_contact"] = self.cfg.w_grip_contact * opening_progress

        elif ph == HOLD_OPEN:
            # §1.29 — HOLD PIATTO (REVERT del §1.28). LEZIONE FISICA DECISIVA: nella CHIUSURA
            # il bersaglio door≈0 è il punto di EQUILIBRIO della porta (latch a riposo), quindi
            # hold_bounce (−20·err) e hold_veldamp (−25·|qvel|) penalizzano uno scostamento che
            # all'ottimo NON esiste → la policy può azzerarli e fa 100%. Nell'APERTURA il goal è
            # vicino al cap, FUORI equilibrio: la molla ritira la porta di 0.024–0.050 rad in modo
            # FISICAMENTE INEVITABILE. Copiare bounce/veldamp punisce la policy per la fisica →
            # il rollout (1.0) crolla e l'eval scende. Diagnosi confermata su 20 episodi reali.
            # Rimedio: tornare all'hold piatto (lo stato che dava rollout 1.0) e risolvere il
            # residuo deterministico per via GEOMETRICA (open_tol allargato alla deriva fisica,
            # vedi config_v2 open_tol_rad), NON con altre penalità. Rif. close §HOLD (equilibrio).
            open_err = abs(goal_angle - door_angle)
            is_open_ok = open_err < open_tol
            # premio piatto quando la porta è al goal (entro tolleranza); fuori tolleranza una
            # GUIDA dolce (peso 1) verso il goal — non una penalità che combatte la molla.
            if is_open_ok:
                rew["hold"] = 1.0
            else:
                rew["hold"] = -1.0 * open_err
            # presa fisica persa (mirror hold_slip della chiusura)
            if not is_physically_closed:
                rew["hold_slip"] = -self.cfg.w_hold_slip
            # comando gripper (mirror della chiusura)
            if gripper_action > grip_thresh:
                rew["hold_grip"] = 1.0
            else:
                rew["hold_grip"] = -2.0 * abs(gripper_action - grip_thresh)
            # anti-apertura del gripper (mirror hold_drop_pen della chiusura)
            if gripper_action < 0.0:
                rew["hold_drop_pen"] = -self.cfg.w_hold_drop_pen * abs(gripper_action)
            # braccio fermo (mirror hold_act della chiusura)
            arm_norm = float(np.linalg.norm(action[:-1]))
            rew["hold_act"] = 1.0 if arm_norm < 0.05 else -2.0 * arm_norm
            # non perdere la maniglia (mirror hold_dist della chiusura)
            if dist_handle > 0.06:
                rew["hold_dist"] = -self.cfg.w_hold_dist * (dist_handle - 0.06)

        elif ph == RETREAT:
            # §1.32 — RETREAT: copia ESATTA del blocco della chiusura (stessi termini, stessi
            # pesi), col solo bersaglio invertito (porta APERTA al goal invece che chiusa).
            open_err     = abs(goal_angle - door_angle)
            door_open_ok = door_angle >= goal_angle - open_tol

            if getattr(fsm_state, "retreat_pos", None) is not None and eef_pos is not None:
                _ep = np.asarray(eef_pos, dtype=float)
                dist_to_target = float(np.linalg.norm(_ep - fsm_state.retreat_pos))
            else:
                dist_to_target = 0.20

            # Gripper: apri per rilasciare la maniglia  [close: ret_grip]
            # §1.37 — il premio +2 è RIPRISTINATO (era stato azzerato in §1.35/§1.36).
            # LEZIONE DI ECONOMIA DELLA REWARD (dal collasso §1.36, SR 0.28, ep_rew +450):
            # i redditi per-step formano un SISTEMA connesso. Togliere il reddito solo dalla
            # FINE della catena (RETREAT) ha reso il limbo REACH/PULL — che ha i suoi redditi
            # (w_reach_grip_near=2.5/step vicino alla maniglia, grip_contact in PULL) — la
            # regione più pagata del task: la policy apriva la porta fino ad APPENA SOPRA la
            # tolleranza (open_error min 0.0505–0.0811, tol 0.050: deliberato) e ci restava
            # per 400+ step. Col reddito del RETREAT ripristinato, COMPLETARE torna a pagare
            # più del limbo (come in tutti i run al 100%: §1.30, §1.34). Il reddito qui è
            # sicuro SOLO perché l'uscita è ESOGENA a durata fissa (§1.36): non estendibile,
            # quindi non farmabile — i 9 episodi perfetti del run §1.36 (RETREAT=31 pulito)
            # lo confermano.
            # §1.46 — durante il RIPORTO ATTIVO della leva (env-guided, presa chiusa per
            # costruzione) i termini di rilascio/direzione sono SOSPESI: il braccio non è
            # sotto controllo della policy, premiarlo o punirlo distorcerebbe il critic.
            _restoring = bool(getattr(fsm_state, "retreat_restoring", False))
            if not _restoring:
                if gripper_action < -0.85:
                    rew["ret_grip"] = 2.0
                else:
                    rew["ret_grip"] = -1.0 * abs(gripper_action + 1.0)

            # Torsione del polso  [close: ret_rot]
            rew["ret_rot"] = -3.0 * float(np.linalg.norm(action[3:6]))

            # §1.39 — ret_lat/ret_down (assi del MONDO) RIMOSSI dal RETREAT dell'apertura.
            # MISURA: la porta ha yaw -90°, quindi la sua NORMALE — e con essa la direzione di
            # ritiro — è al 94-99% lungo Y. `ret_lat = -5*|action[1]|` penalizzava dunque
            # ESATTAMENTE la direzione di fuga, per tutti i 12 cm (dist_handle<0.12) necessari
            # ad allontanarsi. Conto per step: ritirarsi = ret_dir(+3) + ret_lat(-4.69) +
            # ret_grip(+2) + hold(+1) = +1.31 ; stare FERMO = +3.00 → alla policy conviene NON
            # muoversi. Da qui: braccio che non si allontana, gripper aperto ma ancora sulla
            # leva, latch bloccato a ~0.79 che non rientra (traccia reale: asintotico, mentre
            # una leva LIBERA torna a <0.08 in 3-8 step).
            # L'INTENTO della chiusura ("non spazzare di lato vicino alla maniglia") resta
            # garantito da `ret_perp` (-2*||componente perpendicolare alla direzione di
            # ritiro||): stessa penalità, ma espressa nel frame CORRETTO invece che su un asse
            # del mondo che qui coincide con la via d'uscita.

            # Guida direzionale verso retreat_pos, poi SETTLE  [close: ret_dir/ret_perp/
            # ret_freeze/ret_release — §1.15 zona di settle allargata]
            if not _restoring and getattr(fsm_state, "retreat_pos", None) is not None and eef_pos is not None:
                if dist_to_target > self.cfg.fsm_retreat_settle_dist:
                    dir_to_target    = fsm_state.retreat_pos - _ep
                    dir_norm         = dir_to_target / (dist_to_target + 1e-6)
                    action_alignment = float(np.dot(np.asarray(action[:3], dtype=float), dir_norm))
                    rew["ret_dir"]  = 3.0 * action_alignment
                    perp            = np.asarray(action[:3], dtype=float) - action_alignment * dir_norm
                    rew["ret_perp"] = -2.0 * float(np.linalg.norm(perp))
                else:
                    # SETTLE: immobilizza il braccio (tutte le DOF tranne il gripper)
                    rew["ret_freeze"] = -self.cfg.w_retreat_settle * float(np.linalg.norm(action[:-1]))
                    # Rilascio pulito: bonus SOLO a porta APERTA + gripper aperto (mirror:
                    # nella chiusura era door<0.03), così non ci si "accampa" a porta persa.
                    if door_open_ok and gripper_action < -0.85:
                        rew["ret_release"] = 1.0

            # Latch monitor  [close: latch_ret]
            rew["latch_ret"] = -self.cfg.w_latch_ret * abs(latch_qpos)

            # Monitor stabilità porta nel retreat — §1.37: +1 RIPRISTINATO (vedi nota
            # ret_grip: l'economia va tenuta intera; il reddito è sicuro con uscita esogena).
            # Nota anti-trucco: "far cadere la porta per allungare fino al cap" NON paga —
            # perde hold (+1 solo entro tol), paga door_regress e il calo del potenziale, e
            # al cap arriva il TRONCAMENTO senza bonus.
            rew["hold"] = 1.0 if open_err < open_tol else 0.0
            if door_angle < goal_angle - open_tol:
                regress = max(0.0, prev_angle - door_angle)
                rew["door_regress"] = -self.cfg.w_door_regress * regress
            # (§1.33: la terminazione è spostata DOPO il calcolo di truncated, vedi sotto.
            #  Il gate |latch|<tol è stato RIMOSSO dalla terminazione: misura MuJoCo sul
            #  modello reale — la leva ha frictionloss=0.1 e damping=0, quindi il residuo di
            #  equilibrio è ≈0.1/stiffness ∈ [0.05, 0.20] sul range randomizzato [0.5, 2.0]:
            #  per stiffness ≲ 1.0 il residuo SUPERA 0.08 anche a leva perfettamente libera →
            #  gate strutturalmente irraggiungibile in ~1/3 degli episodi → terminazioni al
            #  solo hard-cap (71 esatti in 15/20 nel run §1.32). A porta CHIUSA il chiavistello
            #  aggancia il montante (contatto misurato a hinge≈0.175) e assiste il ritorno:
            #  per questo lo stesso gate nella chiusura funziona. latch_ret resta come segnale
            #  di accompagnamento appreso.)

        # ── successo / terminazione ──
        if just_succeeded:
            rew["success_bonus"] = self.cfg.success_bonus

        truncated = bool(rs_done) or (step_count >= horizon)

        # §1.36 — TERMINAZIONE ESOGENA A DURATA (la ricetta della chiusura, capita fino in
        # fondo). Il gate d'arrivo di §1.33 era INOSSERVABILE: retreat_pos è ancorato al
        # punto eef alla transizione HOLD_OPEN→RETREAT, che NON sta nell'osservazione — una
        # policy senza memoria non può sapere quando è dentro una sfera di 6-8 cm attorno a
        # un punto che non vede (run §1.34 e §1.35: RETREAT=71=cap in TUTTI gli episodi,
        # CON e SENZA reddito → il gate non scattava mai, indipendentemente dagli incentivi).
        # La chiusura non gata MAI sulla posizione: l'uscita è esogena. Qui uguale:
        #   terminazione = retreat sostenuto (durata fissa, osservabile, non estendibile)
        #                  AND porta ancora al bersaglio.
        # Il ritiro VISIBILE non è imposto dalla terminazione ma PAGATO dal reward
        # (ret_dir/ret_perp/ret_grip, con la policy che ora VEDE la fase, §1.34) — è
        # esattamente il meccanismo per cui la chiusura si allontana.
        # Guardia: al retreat_hard_cap l'episodio si chiude SEMPRE (successo se la porta è
        # tornata al bersaglio, troncato senza bonus altrimenti).
        if getattr(self.cfg, "terminate_on_retreat_complete", True) and ph == RETREAT:
            door_open_ok_t = door_angle >= goal_angle - open_tol
            # §1.42 — TERMINAZIONE = RICETTA ESATTA DELLA CHIUSURA (reward close, righe 486-490:
            #   retreat_steps >= target AND |door| < 0.03 AND |latch| < 0.08).
            # La chiusura NON termina finché la LEVA non è tornata a posto: è PROPRIO questo che
            # le da' il tempo di ritirarsi e di far rientrare la maniglia. L'apertura, dal §1.36,
            # terminava a TEMPO (free_steps >= 30) per paura del deadlock — ma la traccia reale
            # DIMOSTRA che a porta aperta la leva TORNA (scende lineare 1.57 -> ~0.88, ~0.023/step,
            # pulita): non tornava a 0 solo perche' l'episodio chiudeva a meta' (step ~37) con la
            # leva a 0.88 e il braccio a 7 cm. Ripristino il gate sul latch, adattando la sola
            # condizione porta (aperta al goal invece di chiusa). Il free-steps resta come SOGLIA
            # MINIMA (il rilascio deve essere iniziato); il gate vero e' la leva.
            free_steps  = int(getattr(fsm_state, "retreat_free_steps", 0))
            min_release = free_steps >= self.cfg.fsm_retreat_target_steps
            latch_home  = abs(latch_qpos) < getattr(self.cfg, "retreat_latch_term_tol", 0.08)
            hardcap     = fsm_state.retreat_steps >= getattr(self.cfg, "retreat_hard_cap", 120)
            # §1.44 — uscita ESOGENA: oltre questa soglia post-rilascio l'episodio chiude
            # ANCHE a leva non tornata (senza bonus). La durata del RETREAT smette di essere
            # controllabile dalla policy → il reddito per-step non e' piu' farmabile
            # tenendo la leva incastrata (run §1.43: ep_rew ~500 e sabotaggio appreso).
            exo_exit    = free_steps >= int(getattr(self.cfg, "retreat_exo_exit_steps", 60))
            # Task COMPIUTO quando la leva e' tornata a casa dopo il rilascio → termina.
            # Il bonus dipende dalla porta ancora al goal; se e' calata chiudi comunque
            # (niente deadlock) ma senza bonus — analogo alla guardia implicita della chiusura.
            if (min_release and latch_home) or exo_exit or hardcap:
                terminated = True
                # §1.43 — il BONUS solo a uscita PULITA (leva a casa dopo il rilascio).
                # Prima il hard-cap pagava comunque il bonus a porta aperta → l'incastro
                # della leva sul dito (RETREAT=201 in 5/5 episodi del diagnostico) risultava
                # "successo" al 100% e il reddito per-step (§1.37: hold+1, ret_grip+2, …)
                # rendeva il farming dell'incastro la strategia ottima. Col bonus vincolato
                # alla leva neutra, completare il ritiro torna a dominare il limbo.
                if door_open_ok_t and (min_release and latch_home):
                    rew["success_bonus"] = self.cfg.success_bonus

        reward = float(np.clip(sum(rew.values()), -50.0, 50.0))
        return reward, terminated, truncated, rew