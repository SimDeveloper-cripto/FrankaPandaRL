#!/usr/bin/env python3

import os
import sys
import types
import textwrap
import dataclasses

import numpy as np
import importlib.util

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "close_generalized_v2"))

def _load_module(name):
    if "close_generalized_v2" not in sys.modules:
        pkg          = types.ModuleType("close_generalized_v2")
        pkg.__path__ = [ROOT]

        sys.modules["close_generalized_v2"] = pkg
    spec = importlib.util.spec_from_file_location(
        f"close_generalized_v2.{name}", os.path.join(ROOT, f"{name}.py")
    )
    m = importlib.util.module_from_spec(spec)
    sys.modules[f"close_generalized_v2.{name}"] = m
    spec.loader.exec_module(m)
    return m


def _extract_retreat_block(env_path):
    src   = open(env_path, encoding="utf-8").read()
    i     = src.index("elif phase == PHASE_RETREAT:")
    i     = src.rfind("\n", 0, i) + 1
    j     = src.index("obs, _, rs_done, info = self._rs_env.step(action)", i)
    j     = src.rfind("\n", 0, j) + 1
    block = textwrap.dedent(src[i:j])
    body  = block.split("\n", 1)[1]

    return "if True:\n" + body


class _FakeSelf:
    pass


def main():
    cfg     = _load_module("config_v2").TrainConfigV2()
    block   = _extract_retreat_block(os.path.join(ROOT, "env_v2.py"))
    results = []

    def check(cond, msg):
        results.append(bool(cond))
        print(("PASS  " if cond else "FAIL  ") + msg)

    def run_retreat(width, ramp0, pol_action, cfg_):
        s     = _FakeSelf()
        s.cfg = cfg_

        s._prev_gripper_width = width
        s._retreat_ramp_step  = ramp0
        s._domain_rand        = types.SimpleNamespace(current_handle_radius = 0.02)
        s._fsm                = types.SimpleNamespace(
            state = types.SimpleNamespace(retreat_pos = None, return_hold = 0)
        )

        a = np.array(pol_action, dtype=float)
        exec(block, {"np": np}, {"self": s, "action": a, "getattr": getattr})
        return a, s._retreat_ramp_step

    R   = int(cfg.retreat_rampup_steps)
    pol = [0.4, 0.4, 0.4, 0.2, 0.2, 0.2, 0.1, -0.9]

    # A
    check(hasattr(cfg, "retreat_rampup_steps") and hasattr(cfg, "retreat_rampup_enabled")
          and R > 0,
          f"A config: §1.21 presente (enabled={cfg.retreat_rampup_enabled}, steps={R})")
    # B
    a, r = run_retreat(0.01, 5, pol, cfg)
    check(np.allclose(a[:-1], 0.0) and a[-1] == -1.0 and r == 0,
          "B retreat: dita non libere → freeze+apertura (§1.17 intatto), rampa reset")
    # C
    a, r = run_retreat(0.08, 0, pol, cfg)
    check(np.allclose(a[:-1], np.array(pol[:-1]) * (1.0 / R)) and a[-1] == pol[-1] and r == 1,
          f"C retreat: 1° step → braccio×(1/{R}), gripper intatto, contatore→1")
    # D
    a, r = run_retreat(0.08, 4, pol, cfg)
    check(np.allclose(a[:-1], np.array(pol[:-1]) * (5.0 / R)) and r == 5,
          f"D retreat: 5° step → braccio×(5/{R}) (rampa lineare)")
    # E
    a, r = run_retreat(0.08, R, pol, cfg)
    check(np.allclose(a, pol) and r == R,
          f"E retreat: oltre {R} step → azione policy piena e invariata")
    # F
    cfg0 = dataclasses.replace(cfg, retreat_rampup_steps=0)
    a, r = run_retreat(0.08, 0, pol, cfg0)
    check(np.allclose(a, pol),
          "F interruttore: rampup_steps=0 → comportamento §1.17 identico")
    # G
    a, _ = run_retreat(0.08, 2, [0, 0, 0, 0, 0, 0, 0, 0.55], cfg)
    check(a[-1] == 0.55,
          "G safety: la rampa scala SOLO il braccio, mai il comando gripper")

    print("\n" + "=" * 56)
    allpass = all(results)
    print(f"ESITO §1.21: {sum(results)}/{len(results)} — "
          + ("TUTTI PASS" if allpass else "FALLITO"))
    sys.exit(0 if allpass else 1)


if __name__ == "__main__":
    main()