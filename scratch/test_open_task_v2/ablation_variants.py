#!/usr/bin/env python3
# scratch/test_open_task_v2/ablation_variants.py
"""
ablation_variants — Interventi di ablazione per la suite di APERTURA v2.

Come nella chiusura v2, gli interventi NON riscrivono `step()` (come facevano i test v1):
i meccanismi da ablazionare sono GIÀ dentro `env_v2.step()` e sono attivati da flag di
config. L'ablazione li DISATTIVA in valutazione, sulla STESSA policy addestrata, e misura
l'effetto sul true success. Un fattore per volta (Patterson et al. 2024).

Sono tutti "override deterministici env-level a successo già acquisito": guidano la
qualità del moto senza aggiungere termini di reward, quindi la policy resta valida e
l'invarianza di Ng, Russell & Harada (1999) non è toccata. Ablazionarli misura quanto
del risultato finale è merito loro e quanto della policy.

Meccanismi ablazionati (tutti presenti nel sorgente di env_v2.py):
  §1.17 retreat_clean_release  — apre il gripper e CONGELA il braccio finché le dita non
                                 sono libere: evita di trascinare la porta nel rilascio.
  §1.18 grip_lock_enabled      — pavimento sul comando gripper in PULL/HOLD_OPEN: impedisce
                                 aperture accidentali della presa mentre si tira.
  §1.21 retreat_rampup         — avvio morbido (rampa lineare 0→1) dell'azione del braccio
                                 dopo il rilascio: niente strappo alla porta.
  §1.43 retreat_escape_enabled — sfilamento GUIDATO lungo la normale della porta finché il
                                 braccio non ha liberato la maniglia: senza, il dito resta
                                 nel piano della leva e la blocca (incastro).
  §1.46 retreat_restore_enabled— RIPORTO ATTIVO della leva lungo il suo arco prima del
                                 rilascio, così lo sfilamento avviene senza carico.
  §1.50 retreat_restore_cage_always — durante il riporto le dita fanno da "gabbia" invece
                                 che da morsa: la barra ruota libera, niente slip né pugno.

`no_all_overrides` disattiva tutto insieme: è il riferimento "policy nuda".
"""

VARIANTS = {
    "baseline":           {},
    "no_clean_release":   {"retreat_clean_release": False},
    "no_grip_lock":       {"grip_lock_enabled": False},
    "no_rampup":          {"retreat_rampup_enabled": False, "retreat_rampup_steps": 0},
    "no_escape":          {"retreat_escape_enabled": False},
    "no_latch_restore":   {"retreat_restore_enabled": False},
    "no_cage":            {"retreat_restore_cage_always": False},
    "no_all_overrides":   {"retreat_clean_release": False, "grip_lock_enabled": False,
                           "retreat_rampup_enabled": False, "retreat_rampup_steps": 0,
                           "retreat_escape_enabled": False, "retreat_restore_enabled": False,
                           "retreat_restore_cage_always": False},
}

# Sottoinsieme consigliato quando il tempo è poco (i tre specifici dell'apertura + il totale)
QUICK_VARIANTS = ["baseline", "no_escape", "no_latch_restore", "no_all_overrides"]
