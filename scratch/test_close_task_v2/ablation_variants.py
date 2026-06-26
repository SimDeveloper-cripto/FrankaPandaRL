#!/usr/bin/env python3
# scratch/test_close_task_v2/ablation_variants.py

"""
ablation_variants — Interventi di ablazione per la suite v2.

In v2 i meccanismi di rilascio/ritiro NON sono riscritti nello step di test (come in
v1) ma sono GIÀ dentro env_v2.step(), attivati da flag di config:
  • §1.17 retreat_clean_release — apre le dita e congela il braccio finché la presa
    non è libera, prima di ritirarsi (rilascio pulito).
  • §1.18 grip_lock_enabled — impedisce aperture accidentali del gripper in PUSH/HOLD.
  • §1.21 retreat_rampup — avvio morbido (rampa) dell'azione del braccio in RETREAT.

L'ablazione v2 quindi DISATTIVA questi flag in valutazione, sulla STESSA policy
addestrata, e misura l'effetto sul true success. È l'analogo esatto degli interventi
post-policy della v1, ma più pulito (usa l'env reale, un solo fattore per volta —
Patterson et al. 2024) e ablaziona direttamente i contributi rivendicati dalla v2.

NB: la policy è addestrata con i flag ATTIVI; questi toggle agiscono solo a eval, quindi
misurano il contributo *a parità di policy*, non un re-training.
"""

VARIANTS = {
    "baseline":         {},  # tutti i default (clean_release, grip_lock, rampup attivi)
    "no_clean_release": {"retreat_clean_release": False},
    "no_grip_lock":     {"grip_lock_enabled": False},
    "no_rampup":        {"retreat_rampup_enabled": False, "retreat_rampup_steps": 0},
    "no_all_three":     {"retreat_clean_release": False, "grip_lock_enabled": False,
                         "retreat_rampup_enabled": False, "retreat_rampup_steps": 0},
}