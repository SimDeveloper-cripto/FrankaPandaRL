#!/usr/bin/env python3
# scratch/test_open_task_v2/ablation_variants.py

"""
ablation_variants — Interventi di ablazione per la suite di APERTURA v2.

Come nella chiusura v2, gli interventi NON riscrivono `step()` (come facevano i test v1):
  i meccanismi da ablazionare sono GIÀ dentro `env_v2.step()` e sono attivati da flag di config.
  L'ablazione li DISATTIVA in valutazione, sulla STESSA policy addestrata, e misura l'effetto sul true success.
  Un fattore per volta (Patterson et al. 2024).
"""

VARIANTS = {
    "baseline":           {},
    "no_clean_release":   {"retreat_clean_release":   False},
    "no_grip_lock":       {"grip_lock_enabled":       False},
    "no_rampup":          {"retreat_rampup_enabled":  False, "retreat_rampup_steps": 0},
    "no_escape":          {"retreat_escape_enabled":  False},
    "no_latch_restore":   {"retreat_restore_enabled": False},
    "no_cage":            {"retreat_restore_cage_always": False},
    "no_all_overrides":   {"retreat_clean_release":  False, "grip_lock_enabled":     False,
                           "retreat_rampup_enabled": False, "retreat_rampup_steps": 0,
                           "retreat_escape_enabled": False, "retreat_restore_enabled": False,
                           "retreat_restore_cage_always": False },
}

QUICK_VARIANTS = ["baseline", "no_escape", "no_latch_restore", "no_all_overrides"]