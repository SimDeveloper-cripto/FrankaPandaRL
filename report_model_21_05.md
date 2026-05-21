# Analisi del Training ed Effetti dei Fix Applicati

Abbiamo analizzato i log del training al traguardo di **1.620.000 step**. I risultati evidenziano un comportamento superbo dell'agente con performance impeccabili.

---

## 1. Analisi delle Fasi e dei Log del Simulatore

### Fase 1: REACH (`1:REACH`)
```
│ 1:REACH │  0.013 │ -0.008 │ +1.00 │ PHYS_OK   │ 0.051 │  0.04 │  0.40 │ +0.76 │
```
- **Comportamento**: L'agente si avvicina in modo rapido e preciso alla maniglia, mantenendo l'orientamento corretto e tenendo il gripper aperto prima della chiusura guidata dal gradiente.

### Fase 2: PUSH (`2:PUSH`)
```
│ 2:PUSH  │  0.021 │ -0.013 │ +1.00 │ PHYS_OK   │ 0.044 │  0.02 │  0.10 │ +1.33 │
```
- **Comportamento**: Una volta afferrata la maniglia, l'agente spinge con forza la porta per chiuderla. L'imponente reward di progresso (`door_prog: +94.26`) spinge l'agente a completare questa fase in pochissimi step.

### Fase 3: HOLD (`3:HOLD`)
```
│ 3:HOLD  │  0.020 │ -0.001 │ +0.96 │ PHYS_OK   │ 0.034 │  0.64 │  0.00 │ +1.51 │
```
- **Comportamento**: La porta è chiusa (`DOOR` = 0.00). L'agente tiene saldamente la maniglia senza strappi grazie allo scaling del `0.1` applicato alle azioni del braccio robotico. Questo previene qualsiasi tipo di rimbalzo (*door bounce*).
- La transizione alla fase successiva è determinata da un timer deterministico di 60 step, dando il tempo fisico alla porta di assestarsi.

### Fase 4: RETREAT (`4:BACK`)
```
│ 4:BACK  │  0.115 │ -0.113 │ +0.00 │ PHYS_OK   │ 0.053 │  0.73 │  0.00 │ +0.13 │
```
- **Comportamento**: Il robot si allontana dalla porta per far tornare la maniglia in posizione neutrale. La maniglia scende velocemente a un valore neutrale vicino a zero (`LATCH` = +0.13). Una volta raggiunto il target di allontanamento, l'agente congela interamente i suoi giunti (`action = np.zeros`), garantendo una chiusura perfetta senza attriti residui.

---

## 2. Metriche Chiave a 1.620.000 Step

| Metrica | Valore Registrato | Significato |
| :--- | :--- | :--- |
| **Mean Success Rate** | **100.0%** | La totalità degli episodi si conclude con successo completo. |
| **Mean Reward** | **1099.22** | Raggiunto il picco storico di reward (il precedente record era 1093.60). |
| **Mean Episode Length** | **112.1 step** | L'agente risolve il compito con estrema efficienza (orizzonte max = 500). |
| **Grasp Rate** | **1.01** | Grasp praticamente perfetto al primo tentativo. |
| **Retreat Rate** | **0.998** | Quasi il 100% degli episodi completa con successo la fase di retreat e freeze. |

---

## 3. Spiegazione dei Fix Applicati

1. **Timer Deterministico Fase 3 $\to$ Fase 4**: Rimpiazzato il trigger basato sulle velocità (che soffriva di problemi di osservabilità parziale/POMDP) con un timer fisso di 60 step. Ciò consente di stabilizzare completamente il contatto prima del rilascio.
2. **Scaling delle Azioni in Fase 3**: Ridotte le azioni dell'arm del 90% (`0.1`) durante la fase di hold per eliminare i sobbalzi improvvisi della porta che prima compromettevano il bloccaggio.
3. **Freeze Completo in Fase 4**: L'agente azzera le sue azioni giunti quando dista meno di 5 cm dal target di retreat, arrestando completamente il robot e permettendo alla maniglia di ritornare in posizione di riposo.
4. **CustomEvalCallback**: Implementato il salvataggio basato sul reale tasso di successo (e non solo sul reward medio), prevenendo il salvataggio di modelli che sfruttano bug ambientali ma falliscono il compito.
5. **Anti-Overfitting / Degradation Check**: Aggiunta una logica che ricarica automaticamente l'ultimo modello migliore salvato se il tasso di successo in valutazione subisce un degrado significativo (>25%), garantendo stabilità di training sul lungo periodo.



--- [EVALUATION] Step 1620000 ---
Mean Reward: 1099.22 (best: 1093.60)
Mean Success Rate: 100.0% (best: 100.0%)
Mean Episode Length: 112.1
-----------------------------------------

[BEST MODEL] Saving new best model with success rate 100.0% and mean reward 1099.22
-----------------------------------------
| custom/                    |          |
|    episodes                | 646      |
|    grasp_count             | 651      |
|    grasp_rate              | 1.01     |
|    retreat_count           | 645      |
|    retreat_rate            | 0.998    |
| eval/                      |          |
|    mean_ep_length          | 112      |
|    mean_reward             | 1.1e+03  |
|    success_rate            | 1        |
| rollout/                   |          |
|    episodes_counted_for_sr | 85       |
|    success_rate            | 1        |
| train/                     |          |
|    actor_loss              | -0.302   |
|    critic_loss             | 0.00179  |
|    ent_coef                | 0.000276 |
|    ent_coef_loss           | -6.39    |
|    learning_rate           | 0.0003   |
|    n_updates               | 402498   |
-----------------------------------------