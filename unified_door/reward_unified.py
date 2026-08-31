#!/usr/bin/env python3

"""
Letteratura, invariata rispetto ai due progetti (§6):
  [Ng, Russell & Harada 1999]   shaping potenziale: F = γΦ(s′) − Φ(s)
  [Devlin & Kudenko 2012]       shaping dipendente dal tempo/fase
  [Sutton, Precup & Singh 1999] opzioni: una fase = un'opzione
  [Konidaris & Barto 2009]      soglie adattive
  [ManipForce 2015]             primitive basate sulla forza di presa
  [Tobin et al. 2017]           domain randomization, parametri in osservazione
  [ten Pas et al. 2017]         feature di presa multi-approccio
  [Krakovna et al. 2020]        specification gaming: il cricchetto e la saturazione esistono per chiudere due scorciatoie
"""

from fsm_unified import Fase
from typing      import Dict, Optional

import numpy as np

TERMINI = ("time", "smooth", "phi", "approach", "approach_geom", "wrist", "grip",
           "progress", "contact", "target", "damp", "still", "hold_grip",
           "release", "retreat", "latch_home", "success")

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
    "damp":          (["hold_veldamp"], []),
    "still":         (["hold_act", "hold_jnt_freeze", "ret_freeze", "ret_jnt_prog"], ["hold_act", "ret_freeze"]),
    "hold_grip":     (["hold_grip", "hold_slip", "hold_drop_pen", "hold_dist"], ["hold_grip", "hold_slip", "hold_drop_pen", "hold_dist"]),
    "release":       (["ret_grip", "ret_release"], ["ret_grip", "ret_release"]),
    "retreat":       (["ret_dir", "ret_perp", "ret_rot", "ret_lat", "ret_down"], ["ret_dir", "ret_perp", "ret_rot"]),
    "latch_home":    (["latch_ret"], ["latch_ret"]),
    "success":       (["success_bonus", "phase_trans"], ["success_bonus"]),
}

SOPPRESSI = {
    "door_regress": ("apertura",
                     "una porta che si allontana dal bersaglio e' gia' penalizzata dal "
                     "calo del mantenimento e dallo smorzamento della velocita' "
                     "(tesi §6.4.1)"),
}


class Potenziale:
    def __init__(self, cfg):
        self.cfg, self.w           = cfg, cfg.w
        self.prev: Optional[float] = None

    def reset(self) -> None:
        self.prev = None

    def _phi(self, fase: Fase, door, timer: float, hold_target: int, dist_ritiro: float, dist_mano: float, grip_cmd: float, raggio: float) -> float:
        w, lvl = self.w, self.cfg.curriculum_level
        k      = 1.0 + w.curriculum_k * lvl

        vicino = dist_mano <= 1.5 * raggio + 0.005
        if fase == Fase.REACH:
            phi_grip = (w.grip_near * float(np.clip((grip_cmd + 0.85) / 1.85, 0.0, 1.0)) if vicino else 0.0)
        else:
            phi_grip = w.grip_near

        if fase == Fase.REACH:
            return phi_grip

        phi = phi_grip + w.phi_reach * k
        if fase == Fase.MOVE:
            phi += w.phi_move * k * door.avanzamento
        else:
            phi += w.phi_move * k
        if fase == Fase.HOLD:
            t = float(np.clip(timer / max(hold_target, 1), 0.0, 1.0))
            phi += w.phi_hold * t * door.target_ramp
        elif fase > Fase.HOLD:
            phi += w.phi_hold
            phi += w.phi_release * float(np.clip(1.0 - dist_ritiro / 0.20, 0.0, 1.0))
        return phi

    def shaping(self, **kw) -> float:
        now = self._phi(**kw)
        if self.prev is None:
            self.prev = now
            return 0.0

        f         = self.cfg.sac.gamma * now - self.prev
        self.prev = now
        return float(np.clip(f, -self.w.phi_clip, self.w.phi_clip))


class UnifiedReward:
    def __init__(self, cfg):
        self.cfg, self.w                       = cfg, cfg.w
        self.phi                               = Potenziale(cfg)
        self.prev_action: Optional[np.ndarray] = None
        self.success_pagato                    = False
        self.transizione_pagata                = False

    def reset(self) -> None:
        self.phi.reset()
        self.prev_action        = None
        self.success_pagato     = False
        self.transizione_pagata = False

    # ═════════════════════════════════════════════════════════════════════
    def compute(self, *, fsm, door, action: np.ndarray, delta_p: float, dist_mano: float, dist_xy: float, dz: float,
                allineamento: float, polso_piatto: float, grip_cmd: float, soglia_presa: float, contatto: bool,
                door_qvel: float, latch: float, vel_giunti: float, raggio: float, dist_ritiro: float, dir_ritiro: Optional[np.ndarray],
                hold_target: int, terminato: bool) -> Dict[str, float]:

        w, s                 = self.w, fsm.s
        fase                 = s.fase
        r: Dict[str, float]  = {}
        a_arm                = np.asarray(action[:-1], dtype = np.float64)

        r["time"] = -w.time_cost

        # ── 2 smooth — sempre ───────────────────────────────────────────
        if self.prev_action is not None:
            r["smooth"] = (-w.smooth * float(np.linalg.norm(a_arm - self.prev_action[:-1]))
                           - w.smooth_act * float(np.linalg.norm(a_arm)))
        self.prev_action = np.asarray(action, dtype=np.float64).copy()

        r["phi"] = self.phi.shaping(fase = fase, door = door, timer = s.timer_hold, hold_target = hold_target, dist_ritiro = dist_ritiro,
                                    dist_mano = dist_mano, grip_cmd = grip_cmd, raggio = raggio)

        # ── 4 approach — REACH, MOVE ────────────────────────────────────
        if fase == Fase.REACH:
            k = 1.0 + w.curriculum_k * self.cfg.curriculum_level
            r["approach"] = -k * (w.app_3d * dist_mano + w.app_xy * dist_xy + w.app_z * abs(dz))
        elif fase == Fase.MOVE:
            r["approach"] = -(w.app_3d * dist_mano + w.app_z * abs(dz))

        # ── 5 approach_geom — REACH ─────────────────────────────────────
        if fase == Fase.REACH:
            lo, hi = w.geom_band
            if dz < lo:
                r["approach_geom"] = -w.geom_below * abs(dz - lo)
            elif dz > hi:
                r["approach_geom"] = -w.geom_above * (dz - hi)

        # ── 6 wrist — REACH, HOLD ───────────────────────────────────────
        if fase in (Fase.REACH, Fase.HOLD):
            prox = float(np.exp(-10.0 * dist_mano))
            peso = w.wrist_reach if fase == Fase.REACH else w.wrist_hold
            r["wrist"] = (-peso * (1.0 - allineamento) * prox
                          - w.wrist_flat * polso_piatto * prox)

        # ── 7 grip — REACH, MOVE ────────────────────────────────────────
        if fase == Fase.REACH and grip_cmd > -0.85:
            if dist_mano > 1.5 * raggio + 0.005:
                r["grip"] = -w.grip_far * (grip_cmd + 0.85)
        elif fase == Fase.MOVE and grip_cmd < soglia_presa:
            r["grip"] = -w.grip_weak_move * (soglia_presa - grip_cmd)

        # ── 8 progress — MOVE ───────────────────────────────────────────
        if fase == Fase.MOVE and delta_p > 0.0:
            r["progress"] = w.w_progress(door.escursione) * delta_p

        # ── 9 contact — MOVE ────────────────────────────────────────────
        if fase == Fase.MOVE and contatto and delta_p > 0.0:
            r["contact"] = w.contact * door.avanzamento

        # ── 10 target — HOLD, RELEASE ───────────────────────────────────
        if fase in (Fase.HOLD, Fase.RELEASE):
            if door.A:
                r["target"] = w.target * door.target_ramp
            elif fase == Fase.HOLD:
                r["target"] = -w.target_bounce * abs(door.e)

        # ── 11 damp — solo HOLD ─────────────────────────────────────────
        if fase == Fase.HOLD and abs(door_qvel) > w.damp_floor:
            r["damp"] = -w.damp * abs(door_qvel)

        # ── 12 still — HOLD, RELEASE ────────────────────────────────────
        if fase == Fase.HOLD:
            norm_a = float(np.linalg.norm(a_arm))
            if norm_a < w.still_eps:
                r["still"] = w.still_bonus
            else:
                r["still"] = -(w.still_act * norm_a + w.still_jnt * vel_giunti)
        elif fase == Fase.RELEASE and dist_ritiro <= self.cfg.thr.retreat_settle_dist:
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
            if (mano_aperta and door.A and dist_ritiro <= self.cfg.thr.retreat_settle_dist):
                v += w.rel_clean
            r["release"] = v

        # ── 15 retreat — RELEASE ────────────────────────────────────────
        if fase == Fase.RELEASE:
            v = -w.ret_rot * float(np.linalg.norm(a_arm[3:6]))
            if dir_ritiro is not None and dist_ritiro > self.cfg.thr.retreat_settle_dist:
                u = np.asarray(dir_ritiro, dtype = np.float64)
                n = float(np.linalg.norm(u))
                if n > 1e-9:
                    u     = u / n
                    lungo = float(np.dot(a_arm[:3], u))
                    perp  = a_arm[:3] - lungo * u
                    v += (w.ret_dir * lungo
                          - w.ret_perp * float(np.linalg.norm(perp)))
            r["retreat"] = v

        # ── 16 latch_home — RELEASE ─────────────────────────────────────
        if fase == Fase.RELEASE:
            r["latch_home"] = -w.latch_home * abs(latch)

        # ── 17a transizione — una tantum all'uscita da REACH ────────────
        if s.prev_fase == Fase.REACH and fase == Fase.MOVE and not self.transizione_pagata:
            r["success"] = r.get("success", 0.0) + w.transizione
            self.transizione_pagata = True

        # ── 17 success — terminale ──────────────────────────────────────
        if terminato and door.A and not self.success_pagato:
            r["success"]        = w.success
            self.success_pagato = True

        # ── §8 LA MASCHERA ────────────────────────────────────────────
        for k in fsm.termini_mascherati():
            r.pop(k, None)

        return r


    def total(self, termini: Dict[str, float], terminato: bool = False) -> float:
        s = float(sum(termini.values()))
        return s if terminato else float(np.clip(s, -self.w.clip_step, self.w.clip_step))