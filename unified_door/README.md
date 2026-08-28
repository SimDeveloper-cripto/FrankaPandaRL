# unified_door — la reward machine unificata

Implementazione di `tabella_reward_machine_unificata.md` per **chiusura v2 curr 1** e
**apertura v2 curr 1**. Ogni blocco di codice cita la sezione del documento che realizza,
così ogni affermazione del documento si ritrova nel codice e viceversa.

```
unified_door/
  config_unified.py      i 17 pesi · i 9 parametri di compito · SAC invariato   (§1, §6)
  state_unified.py       σ, e, A, lavoro, p, avanzamento                         (§4, §7)
  fsm_unified.py         4 fasi · gate bilaterale · guardia · isteresi · maschera (§3, §8)
  reward_unified.py      i 17 termini e il potenziale Φ                          (§1, §2, §5)
  env_unified.py         osservazione a 126 · controllori · punto di ritiro      (§6, §8)
  train_unified.py       addestramento, play, valutazione
  tests/test_unified.py  197 controlli contro la specifica (193 senza PROGETTI_ORIGINALI)
```

## Dove ritrovare ogni sezione del documento

| documento | dove sta nel codice |
|:---|:---|
| §1 i diciassette termini e i loro pesi | `TERMINI`, `Weights`, `UnifiedReward.compute` |
| §2 lo shaping potenziale Φ | `Potenziale._phi` e `Potenziale.shaping` |
| §3 le quattro fasi e le quattro correzioni | `Fase`, `UnifiedFSM.step`, `UnifiedFSM.presa_persa` |
| §4 le grandezze derivate e i cinque usi di A | `DoorState` + `USI_DI_A` |
| §4 le soglie adattive | `UnifiedFSM.soglia_distanza`, `.soglia_presa`, `UnifiedConfig.hold_steps` |
| §5 la mappatura 41 → 17 e la definizione soppressa | `ASSORBE`, `SOPPRESSI` |
| §6 i nove parametri di compito | `TaskSpec`, `CHIUSURA_V2_CURR1`, `APERTURA_V2_CURR1` |
| §6 SAC invariato e l'osservazione a 126 | `SacHyper`, `OBS_DIM`, `OBS_BLOCCHI` |
| §7 `progress` come budget · ancoraggio · leva | `Weights.w_progress`, `DoorState.ancora/incassa/escursione` |
| §8 i controllori, la maschera, il punto di ritiro | `Controllore`, `MASCHERA`, `_applica_controllori`, `_punto_ritiro` |
| §9 la verifica: soluzione contro condotte stazionarie | `tests/test_unified.py` |

## Come si esegue

I moduli che il §6 dichiara invariati (`MultiApproachGrasp`, `ExtendedDomainRandomizer`) non
sono riscritti: sono importati dai progetti originali. Va indicata la cartella che contiene
`close_generalized_v2/` e `open_generalized_v2/`:

```bash
export PROGETTI_ORIGINALI="/percorso/che/li/contiene"
cd unified_door

python3 tests/test_unified.py                                  # 197 controlli
python3 train_unified.py --task close --total-steps 1500000
python3 train_unified.py --task open  --total-steps 1500000
python3 train_unified.py --task close --eval --episodes 20
mjpython train_unified.py --task open --play                   # su macOS
```

Senza `PROGETTI_ORIGINALI` l'ambiente parte lo stesso con sostituti neutri e lo dice a
schermo: va bene per i test, **non** per addestrare, perché le feature di presa e il
contesto fisico sarebbero tutti zeri.

## L'esperimento è A/B a una sola variabile

Fra i due comandi di addestramento cambia solo `--task`. Pesi, shaping, fasi, soglie,
osservazione e iperparametri di SAC sono gli stessi; cambiano soltanto i dieci parametri
del §6, nessuno dei quali è un termine di ricompensa.
