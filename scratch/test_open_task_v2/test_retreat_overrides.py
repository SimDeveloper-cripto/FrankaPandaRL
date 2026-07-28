#!/usr/bin/env python3
# scratch/test_open_task_v2/test_retreat_overrides.py
"""
test_retreat_overrides — Test FUNZIONALE OFFLINE (white-box) degli override
deterministici del RETREAT nell'APERTURA v2.

È lo speculare di `test_retreat_ramp.py` della chiusura, esteso perché il RETREAT
dell'apertura contiene più macchinario: oltre al rilascio pulito (§1.17) e alla rampa
(§1.21) ci sono il riporto attivo della leva (§1.46), la gabbia (§1.50), lo sfilamento
guidato (§1.43) e la guardia di velocità della porta (§1.51).

Tecnica (identica alla chiusura): NON serve robosuite né un modello addestrato. Si
estrae dal sorgente REALE di `env_v2.py` il blocco del ramo RETREAT e lo si esegue in un
harness con un `self` finto, verificando le proprietà di sicurezza direttamente sul
codice che verrà eseguito in produzione. Deterministico, curriculum-indipendente,
qualche millisecondo.

Perché conta: questi override agiscono a successo già acquisito e NON aggiungono termini
di reward, quindi l'invarianza di policy di Ng, Russell & Harada (1999) resta esatta; ma
proprio per questo un bug qui non si vedrebbe nella reward curve — solo in un test come
questo o in un fallimento sistematico del ritiro.

Proprietà verificate:
  A  i parametri §1.17/§1.21/§1.43/§1.46/§1.50 esistono con i default attesi
  B  §1.46 riporto attivo: il braccio accompagna la leva lungo il suo arco (traslazione
     tangente + rotazione del polso coerente), con magnitudine attesa
  C  §1.50 gabbia in retroazione: bang-bang sulla larghezza REALE (apre se troppo
     stretto, chiude se troppo largo) — mai una morsa, mai un pugno
  D  §1.47 fine riporto: a leva scarica il riporto si chiude E invalida retreat_pos
     (che era ancorato alla posa d'ingresso, ormai obsoleta)
  E  §1.51 rampa del riporto: la magnitudine satura a regime
  F  §1.51 guardia di velocità: con la porta in movimento il comando è smorzato, ma
     mai sotto il pavimento (niente braccio inerte)
  G  §1.17 rilascio pulito: dita non libere → braccio CONGELATO e gripper in apertura
  H  §1.21 rampa, 1° step dopo il rilascio → braccio × (1/R), contatore → 1
  I  §1.21 rampa, step intermedio → braccio × (k/R) (lineare)
  J  §1.21 oltre R step → azione del braccio piena e invariata
  K  §1.21 interruttore: retreat_rampup_steps = 0 → nessuna rampa
  L  §1.43 sfilamento guidato: azione diretta verso retreat_pos, polso azzerato,
     rampa tenuta a zero (parte dopo)
  M  §1.43 a sfilamento compiuto (moved ≥ escape_dist) la guida cede il passo alla rampa
  N  §1.46 interruttore: retreat_restore_enabled = False → nessun riporto
  O  §1.43 interruttore: retreat_escape_enabled = False → nessuna guida
  P  sicurezza: dopo il rilascio il comando gripper è SEMPRE apertura (−1), qualunque
     cosa chieda la policy; la rampa scala SOLO il braccio

Uso:
  python scratch/test_open_task_v2/test_retreat_overrides.py     (exit 0 = tutti PASS)
"""

from __future__ import annotations

import os
import sys
import types
import textwrap
import dataclasses
import importlib.util

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _find_repo_root(start=None):
    """Risale finché trova la cartella-pacchetto open_generalized_v2. Volutamente NON
    importa _common: questo test deve girare anche senza robosuite/SB3/modello."""
    here = os.path.abspath(start or os.path.dirname(__file__))
    cur = here
    for _ in range(8):
        if os.path.isdir(os.path.join(cur, "open_generalized_v2")):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    return os.path.abspath(os.path.join(here, "..", ".."))


REPO_ROOT = _find_repo_root()
PKG_DIR = os.path.join(REPO_ROOT, "open_generalized_v2")


def _load_config():
    """Carica il SOLO config_v2 (dataclass puro) senza tirare dentro l'env."""
    name = "open_generalized_v2.config_v2"
    if "open_generalized_v2" not in sys.modules:
        pkg = types.ModuleType("open_generalized_v2")
        pkg.__path__ = [PKG_DIR]
        sys.modules["open_generalized_v2"] = pkg
    spec = importlib.util.spec_from_file_location(name, os.path.join(PKG_DIR, "config_v2.py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m.TrainConfigV2Open()


def _extract_retreat_block(env_path):
    """Ritaglia il ramo `elif phase == PHASE_RETREAT ...:` fino a subito prima di
    `self._prev_action = action.copy()`, e lo rende eseguibile come `if True:`."""
    src = open(env_path, encoding="utf-8").read()
    i = src.index("elif phase == PHASE_RETREAT")
    i = src.rfind("\n", 0, i) + 1
    j = src.index("self._prev_action = action.copy()", i)
    j = src.rfind("\n", 0, j) + 1
    block = textwrap.dedent(src[i:j])
    body = block.split("\n", 1)[1]
    return "if True:\n" + body


class _FakeSelf:
    """`self` minimo: espone solo ciò che il blocco RETREAT usa davvero."""

    def __init__(self, cfg, *, eef=(0.0, 0.0, 0.0), latch=1.2, qvel=0.0, gw=0.08,
                 restore_dir=(1.0, 0.0, 0.0), omega=(0.0, 0.0, 1.0), dir_none=False):
        self.cfg = cfg
        self._eef = np.array(eef, dtype=float)
        self._latch = float(latch)
        self._qvel = float(qvel)
        self._gw = float(gw)
        self._dir = np.array(restore_dir, dtype=float)
        self._omega = np.array(omega, dtype=float)
        self._dir_none = bool(dir_none)
        # stato mutabile letto/scritto dal blocco
        self._retreat_eef0 = None
        self._retreat_escape_eef0 = None
        self._retreat_ramp_step = 0
        self._retreat_free_steps = 0
        self._retreat_restore_done = False
        self._retreat_restore_steps = 0
        self._retreat_restore_cage = False
        self._retreat_restore_latch0 = None
        self._fsm = types.SimpleNamespace(
            state=types.SimpleNamespace(retreat_pos=None, return_hold=0))

    # API usata dal blocco
    def _eef_pos(self):
        return self._eef.copy()

    def _latch_qpos(self):
        return self._latch

    def _door_qvel(self):
        return self._qvel

    def _gripper_width_live(self):
        return self._gw

    def _latch_restore_dir(self):
        return None if self._dir_none else (self._dir.copy(), self._omega.copy())


HANDLE_RADIUS = 0.02
HANDLE_DIAM = HANDLE_RADIUS * 2.0          # 0.04, come nello step reale
POLICY_ACTION = [0.4, 0.4, 0.4, 0.2, 0.2, 0.2, 0.1, -0.9]   # braccio[:7] + gripper[-1]


def run():
    cfg = _load_config()
    block = _extract_retreat_block(os.path.join(PKG_DIR, "env_v2.py"))
    code = compile(block, "<env_v2:RETREAT>", "exec")
    results = []

    def check(cond, msg):
        results.append((msg, bool(cond)))
        print(("PASS  " if cond else "FAIL  ") + msg)

    def exec_retreat(s, pol=None, handle_diam=HANDLE_DIAM):
        a = np.array(POLICY_ACTION if pol is None else pol, dtype=float)
        exec(code, {"np": np, "handle_diam": handle_diam}, {"self": s, "action": a})
        return a

    R = int(cfg.retreat_rampup_steps)
    G = float(cfg.retreat_restore_gain)
    RR = int(cfg.retreat_restore_ramp)
    ROT = float(cfg.retreat_restore_rot_gain)
    CAGE_M = float(cfg.retreat_restore_cage_margin)
    ESC_D = float(cfg.retreat_escape_dist)
    ESC_G = float(cfg.retreat_escape_gain)
    FLOOR = float(cfg.retreat_door_qvel_floor)
    QREF = float(cfg.retreat_door_qvel_ref)

    # ── A — parametri di configurazione ──────────────────────────────────────
    check(R > 0 and cfg.retreat_clean_release and cfg.retreat_escape_enabled
          and cfg.retreat_restore_enabled and cfg.retreat_restore_cage_always,
          f"A config: §1.17={cfg.retreat_clean_release} §1.21(steps)={R} "
          f"§1.43={cfg.retreat_escape_enabled} §1.46={cfg.retreat_restore_enabled} "
          f"§1.50={cfg.retreat_restore_cage_always}")

    # ── B — riporto attivo della leva (§1.46 + §1.49) ────────────────────────
    s = _FakeSelf(cfg, latch=1.2, qvel=0.0, gw=0.08)
    s._retreat_ramp_step, s._retreat_free_steps = 5, 3
    a = exec_retreat(s)
    mag = min(0.6, G * 1.2) * min(1.0, 1.0 / RR) * 1.0        # ramp al 1° step, damp=1
    ok_b = (np.allclose(a[:3], np.clip(np.array([1.0, 0.0, 0.0]) * mag, -1, 1))
            and np.allclose(a[3:6], np.clip(np.array([0.0, 0.0, 1.0]) * (ROT * mag), -1, 1))
            and s._retreat_restore_steps == 1 and s._retreat_restore_latch0 == 1.2
            and s._retreat_ramp_step == 0 and s._retreat_free_steps == 0)
    check(ok_b, f"B §1.46 riporto: traslazione tangente ×{mag:.3f} + rotazione polso "
                f"×{ROT}, rampa/free azzerati, contatore riporto → 1")

    # ── C — gabbia in retroazione (§1.50) ────────────────────────────────────
    w_tgt = HANDLE_DIAM + CAGE_M
    s_wide = _FakeSelf(cfg, latch=1.2, gw=w_tgt + 0.02)       # troppo largo → chiudi
    a_wide = exec_retreat(s_wide)
    s_tight = _FakeSelf(cfg, latch=1.2, gw=w_tgt - 0.02)      # troppo stretto → apri
    a_tight = exec_retreat(s_tight)
    check(a_wide[-1] == 1.0 and a_tight[-1] == -1.0,
          f"C §1.50 gabbia: bang-bang sulla larghezza reale attorno a {w_tgt:.3f} m "
          f"(largo→{a_wide[-1]:+.0f}, stretto→{a_tight[-1]:+.0f})")

    # ── D — fine riporto: chiude e invalida retreat_pos (§1.47) ──────────────
    s = _FakeSelf(cfg, latch=0.2, gw=0.08)                    # 0.2 <= restore_tol 0.35
    s._fsm.state.retreat_pos = np.array([1.0, 0.0, 0.0])
    a = exec_retreat(s)
    check(s._retreat_restore_done and s._fsm.state.retreat_pos is None,
          "D §1.47 fine riporto: leva scarica → riporto concluso e retreat_pos invalidato")

    # ── E — rampa del riporto a regime (§1.51) ───────────────────────────────
    s = _FakeSelf(cfg, latch=1.2, qvel=0.0)
    s._retreat_restore_steps = RR + 5                          # oltre la rampa
    a = exec_retreat(s)
    mag_full = min(0.6, G * 1.2)
    check(np.allclose(a[:3], np.array([1.0, 0.0, 0.0]) * mag_full),
          f"E §1.51 rampa riporto: oltre {RR} step la magnitudine satura a {mag_full:.3f}")

    # ── F — guardia di velocità della porta (§1.51) ──────────────────────────
    s = _FakeSelf(cfg, latch=1.2, qvel=2.0 * QREF)             # ben oltre la soglia
    s._retreat_restore_steps = RR + 5
    a_damp = exec_retreat(s)
    s2 = _FakeSelf(cfg, latch=1.2, qvel=50.0)                  # velocità assurda
    s2._retreat_restore_steps = RR + 5
    a_floor = exec_retreat(s2)
    check(np.allclose(a_damp[:3], np.array([1.0, 0.0, 0.0]) * mag_full * FLOOR)
          and np.allclose(a_floor[:3], np.array([1.0, 0.0, 0.0]) * mag_full * FLOOR)
          and float(np.linalg.norm(a_floor[:3])) > 0.0,
          f"F §1.51 guardia qvel: porta in moto → comando smorzato al pavimento {FLOOR}, "
          f"mai a zero")

    # ── G — rilascio pulito (§1.17) ──────────────────────────────────────────
    s = _FakeSelf(cfg, gw=0.01)                                # dita NON libere
    s._retreat_restore_done = True                             # riporto già concluso
    s._retreat_ramp_step, s._retreat_free_steps = 5, 7
    a = exec_retreat(s)
    check(np.allclose(a[:-1], 0.0) and a[-1] == -1.0
          and s._retreat_ramp_step == 0 and s._retreat_free_steps == 0,
          "G §1.17 rilascio pulito: dita non libere → braccio congelato + gripper in "
          "apertura, rampa e durata utile azzerate")

    # ── H/I/J/K — rampa di avvio (§1.21), isolata (escape inattivo: retreat_pos=None)
    def _ramp_self(ramp0, cfg_=cfg):
        s = _FakeSelf(cfg_, gw=0.08)
        s._retreat_restore_done = True
        s._fsm.state.retreat_pos = None                        # nessuna guida di escape
        s._retreat_ramp_step = ramp0
        return s

    s = _ramp_self(0); a = exec_retreat(s)
    check(np.allclose(a[:-1], np.array(POLICY_ACTION[:-1]) * (1.0 / R))
          and s._retreat_ramp_step == 1 and s._retreat_free_steps == 1,
          f"H §1.21 rampa: 1° step → braccio × (1/{R}), contatore → 1")

    s = _ramp_self(4); a = exec_retreat(s)
    check(np.allclose(a[:-1], np.array(POLICY_ACTION[:-1]) * (5.0 / R))
          and s._retreat_ramp_step == 5,
          f"I §1.21 rampa: 5° step → braccio × (5/{R}) (lineare)")

    s = _ramp_self(R); a = exec_retreat(s)
    check(np.allclose(a[:-1], np.array(POLICY_ACTION[:-1])) and s._retreat_ramp_step == R,
          f"J §1.21 oltre {R} step → azione del braccio piena e invariata")

    cfg0 = dataclasses.replace(cfg, retreat_rampup_steps=0)
    s = _ramp_self(0, cfg0); a = exec_retreat(s)
    check(np.allclose(a[:-1], np.array(POLICY_ACTION[:-1])),
          "K §1.21 interruttore: retreat_rampup_steps = 0 → nessuna rampa")

    # ── L — sfilamento guidato (§1.43) ───────────────────────────────────────
    s = _FakeSelf(cfg, eef=(0.0, 0.0, 0.0), gw=0.08)
    s._retreat_restore_done = True
    s._fsm.state.retreat_pos = np.array([0.3, 0.0, 0.0])       # 0.3 m lungo +x
    a = exec_retreat(s)
    expected = np.clip(np.array([1.0, 0.0, 0.0]) * min(1.0, ESC_G * 0.3), -1, 1)
    check(np.allclose(a[:3], expected) and np.allclose(a[3:-1], 0.0)
          and s._retreat_ramp_step == 0 and a[-1] == -1.0,
          f"L §1.43 escape: braccio guidato verso retreat_pos (|a|={expected[0]:.2f}), "
          f"polso azzerato, rampa tenuta a 0")

    # ── M — escape compiuto → subentra la rampa ──────────────────────────────
    s = _FakeSelf(cfg, eef=(0.0, 0.0, 0.0), gw=0.08)
    s._retreat_restore_done = True
    s._fsm.state.retreat_pos = np.array([0.3, 0.0, 0.0])
    s._retreat_escape_eef0 = np.array([-(ESC_D + 0.05), 0.0, 0.0])   # già allontanato
    a = exec_retreat(s)
    check(np.allclose(a[:-1], np.array(POLICY_ACTION[:-1]) * (1.0 / R))
          and s._retreat_ramp_step == 1,
          f"M §1.43→§1.21: superata la distanza di sfilamento ({ESC_D} m) la guida "
          f"cede il passo alla rampa")

    # ── N — interruttore del riporto (§1.46) ─────────────────────────────────
    cfg_nr = dataclasses.replace(cfg, retreat_restore_enabled=False)
    s = _FakeSelf(cfg_nr, latch=1.2, gw=0.08)
    s._fsm.state.retreat_pos = None
    a = exec_retreat(s)
    check(s._retreat_restore_steps == 0
          and np.allclose(a[:-1], np.array(POLICY_ACTION[:-1]) * (1.0 / R)),
          "N §1.46 interruttore: retreat_restore_enabled = False → nessun riporto, "
          "si passa direttamente al rilascio/rampa")

    # ── O — interruttore dell'escape (§1.43) ─────────────────────────────────
    cfg_ne = dataclasses.replace(cfg, retreat_escape_enabled=False)
    s = _FakeSelf(cfg_ne, gw=0.08)
    s._retreat_restore_done = True
    s._fsm.state.retreat_pos = np.array([0.3, 0.0, 0.0])       # ci sarebbe, ma è spento
    a = exec_retreat(s)
    check(np.allclose(a[:-1], np.array(POLICY_ACTION[:-1]) * (1.0 / R)),
          "O §1.43 interruttore: retreat_escape_enabled = False → nessuna guida, "
          "solo rampa")

    # ── P — sicurezza sul gripper dopo il rilascio ───────────────────────────
    s = _ramp_self(2)
    a = exec_retreat(s, pol=[0.5, 0.5, 0.5, 0.3, 0.3, 0.3, 0.2, 0.55])
    check(a[-1] == -1.0 and np.allclose(a[:-1], np.array([0.5, 0.5, 0.5, 0.3, 0.3, 0.3, 0.2]) * (3.0 / R)),
          "P sicurezza: dopo il rilascio il gripper è SEMPRE in apertura (−1) anche se la "
          "policy chiede di chiudere; la rampa scala solo il braccio")

    n_pass = sum(p for _, p in results)
    print("\n" + "=" * 64)
    print(f"ESITO override RETREAT (apertura v2): {n_pass}/{len(results)} — "
          + ("TUTTI PASS" if n_pass == len(results) else "FALLITO"))
    print("=" * 64)
    return dict(passed=n_pass, total=len(results), checks={m: p for m, p in results})


def main():
    out = run()
    sys.exit(0 if out["passed"] == out["total"] else 1)


if __name__ == "__main__":
    main()
