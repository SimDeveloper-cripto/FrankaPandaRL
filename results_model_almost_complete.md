# Overview

L'architettura del sistema si fonda su un'integrazione stretta tra __Gymnasium__, __Robosuite__ e __Stable Baselines 3__, orchestrata da una __FSM__ per guidare l'apprendimento.

## 1. Stack e Componenti Principali
| Componente | File | Ruolo |
|---|---|---|
| **Env base** | `train_close.py` | Gymnasium wrapper, action smoothing (α=0.8), calcolo della soglia di successo (`_success_angle`), gestione della reward base. |
| **Env FSM** | `close_generalized/env_gen.py` | FSM a 4 fasi, full reward shaping dinamico, domain randomization (attrito, fisica). |
| **Training** | `close_generalized/train_gen.py` | Setup di SAC, curriculum callback, diagnostica avanzata del grasp. |
| **Config** | `config/train_close_config.py` | Definizione di tutti gli iperparametri chiave (horizon=600, total_steps=800k, γ=0.95). |

### La Macchina a Stati Finiti (FSM) — Funzionamento Dettagliato

L'agente apprende attraversando quattro fasi distinte, ognuna con incentivi specifici per superare problemi di underactuation e collisione:

- **Phase 1 (REACH):** Il robot deve raggiungere la maniglia ed impugnarla.
  - *Penalità*: distanze 3D/xy/z, tassa temporale (base - _W_GRIPPER_CLOSE = -2.5/step).
  - *Transizione*: Azione del gripper > 0.65, collisioni evitate (PHYS_OK), distanza < 0.02 per 5 step → Si passa alla spinta con bonus +20.
- **Phase 2 (PUSH):** Il robot deve spingere la porta assecondandone l'arco, mantenendo impugnatura.
  - *Reward*: Basato sul progresso angolare della porta (`_W_PROGRESS_GRASP=2000 * Δangle`), attivo solo se il gripper mantiene la presa intenzionalmente. L'agente viene punito in caso di rilascio prematuro, costringendolo a ripartire dalla Phase 1.
  - *Transizione*: Scatta quando l'angolo della porta supera la soglia critica (`door_qpos < close_fraction * range = ~0.023 rad`).
- **Phase 3 (HOLD):** Il robot deve smorzare il rimbalzo e permettere l'ingaggio del chiavistello.
  - *Gestione fisica*: Timer di 60 step (2s × 30Hz) che verifica la chiusura.
  - *Penalità dinamiche*: `hold_bounce` per i rimbalzi, `hold_slip` per la perdita di presa inattesa.
  - *Evoluzione recente*: Introduzione di `latch_wait` e `wait_grip` per forzare l'apertura volontaria delle dita e risolvere il deadlock meccanico del chiavistello.
- **Phase 4 (BACK):** Ritirata fluida e sicura, con blocco.
  - *Ottimizzazione*: Premia l'apertura totale del gripper (`ret_grip: +2.0`) e guida il polso all'indietro minimizzando le rotazioni spurie e massimizzando la distanza di sicurezza.

### Spazio di Osservazione (122 dimensioni)
Per fornire all'agente un contesto assoluto, l'osservazione combina:
- **Base robosuite (~114 dim):** joint pos/vel/cos/sin, eef pos/quat, gripper qpos/qvel, door/handle pos, proprio-state, object-state.
- **Custom (8 dim):** [dist_handle, handle_radius, handle_friction, flag fsm_reach, flag fsm_push, flag fsm_hold, flag fsm_retreat, hinge_qpos].

---

## 2. Fondamenta Teoriche (Letteratura di Riferimento)

L'approccio scelto è validato dalle recenti scoperte in ambito robotica RL. Questa sezione evidenzia come la letteratura abbia guidato le scelte architetturali.

| Riferimento (Paper) | Titolo e Contenuto | Implementato? | Impatto sul Progetto |
|---|---|---|---|
| **Haarnoja et al., 2018** | *Soft Actor-Critic (SAC)* | ✅ Completo | Algoritmo core: off-policy, max-entropy, double-Q, che permette l'esplorazione sicura in scenari fisici fragili. |
| **Ling & Wen, 2025** | *SAC + Hierarchical Reward Mechanism* | ✅ Integrato | Lo stage-based reward proposto (confermato dal loro 98% in nut grasping) giustifica perfettamente la nostra architettura FSM. |
| **Kumar et al., 2020** | *DisCor: Distribution Correction* | ❌ Valutato | Non implementato. Abbiamo optato per shaping della FSM invece di alterare il buffer di replay. |
| **Feix et al., 2016** | *A Comprehensive Grasp Taxonomy* | ⚠️ Implicito | Giustifica l'uso del check `is_physically_closed` basato sulla misurazione della larghezza delle dita contro il cilindro della maniglia (Power Cylindrical Grip). |
| **Cairnes et al., 2023** | *Overview of Robotic Grippers* | ✅ Informativo | Supporta l'uso della domain randomization sull'attrito (0.3–1.2×) per migliorare la generalizzazione zero-shot. |
| **Ratliff et al., 2009** | *CHOMP: Gradient Motion Planning* | ❌ Valutato | Valido teoricamente per il retreat smooth (Phase 4), ma abbiamo risolto end-to-end addestrando una policy reattiva. |

---

## 3. Diagnosi dei Colli di Bottiglia: Anatomia dei Fallimenti Precedenti

Nel percorso verso il 100%, l'agente ha incontrato ostacoli fisici non banali derivati dall'interazione in simulazione MuJoCo. Ecco le cause radice storiche e come sono state analizzate:

### Problema 1: Il Rimbalzo Elastico (Bounce) in Phase 3
- **Fenomenologia:** La porta, spinta a 0.0 rad, colpiva lo stipite e rimbalzava indietro. Il timer di Phase 3 subiva un reset secco, distruggendo i progressi dell'agente.
- **Causa:** Smorzamento insufficiente in MuJoCo, accoppiato a una punizione troppo severa (`self._hold_closed_duration = 0`).

### Problema 2: Lo Scivolamento (Grip Slip) in Phase 2
- **Fenomenologia:** Durante il movimento ad arco della porta, l'agente "perdeva" la maniglia, tornando repentinamente in Phase 1.
- **Causa:** Il gripper, ignaro della velocità tangenziale della maniglia, inseguiva in ritardo. Tolleranze geometriche rigide (0.05m) risultavano fatali durante i movimenti rapidi.

### Problema 3: L'Exploit del "Falso Progresso" in Phase 2
- **Fenomenologia:** Success rate al 45% ma reward altissimo (+300). L'agente stallava la porta a 0.04 rad (appena prima della chiusura totale) fino allo scadere del tempo (timeout 500 step).
- **Causa:** Ricompense dense troppo generose (`grip_hold: +2.0` e `near_latch_bonus`) in Phase 2 rendevano la stasi più redditizia del completamento del task (che comportava l'ingresso nella difficile Phase 3).

---

## 4. Evoluzione delle Soluzioni: Dai Fix alla Convergenza

Per affrontare le cause radice descritte, ho implementato una serie di fix architetturali che hanno preparato il terreno per la risoluzione definitiva:

### Strategie Anti-Bounce e Stabilità (Phase 3)
1. **Soft Timer Reset:** Invece di azzerare il timer, è stato introdotto un decremento proporzionale all'errore, rendendo la policy resistente alle micro-oscillazioni.
2. **Penalità di Damping (hold_veldamp):** L'agente viene ora punito severamente (`-15.0 * abs(door_qvel)`) se lascia rimbalzare la porta, costringendolo ad applicare forza attiva per contrastare l'inerzia.
3. **Controllo del Momentum (hold_dist):** Penalizza l'agente se la porta si chiude per inerzia mentre lui è lontano, forzando un contatto fisico fino all'ultimo istante.

### Strategie di Tracking e Anti-Exploit (Phase 2)
1. **Tolleranze Dinamiche:** La soglia di rottura del grasp è calcolata in base alla velocità istantanea della porta (`effective_lose_tol = np.clip(0.04 + door_speed * 0.5, 0.04, 0.10)`).
2. **Rimozione dei Falsi Incentivi:** Rimuovendo i bonus `grip_hold` e `near_latch_bonus` dalla Phase 2, questa fase è tornata ad essere "di solo passaggio", forzando l'agente a chiudere la porta il più rapidamente possibile per incassare il bonus di fine task.

### Strategia di Retreat Sicuro (Phase 4)
1. **Ridefinizione dei Pesi:** Ribilanciato il trade-off tra movimento e distanza target.
2. **Incentivo all'Apertura:** Reward massiccio (`ret_grip: +2.0`) per comandare fisicamente il rilascio totale delle dita.

---

## 5. Il Traguardo Finale: 100% Success Rate (14/05/2026)

Con la correzione definitiva delle anomalie di reward shaping che causavano stasi in Phase 2 e l'introduzione delle routine di attesa per il chiavistello (`wait_grip`), la rete ha colmato l'ultimo gap prestazionale (dal 95% al 100%).

Il modello finale a **800,000 timestep** mostra un enorme successo.

### Metriche Chiave dell'Ultimo Run
```text
---------------------------------
| eval/              |          |
|    mean_ep_length  | 298.20   |
|    mean_reward     | -12.57   |
|    success_rate    | 1.00     |  ← IL TRAGUARDO (100.00%)
| rollout/           |          |
|    ep_len_mean     | 283      |  ← Esecuzione rapida (budget 600 step)
|    ep_rew_mean     | 237      |
|    success_rate    | 0.96     |  ← Rollout SR (training in corso)
| time/              |          |
|    fps             | 265      |
|    total_timesteps | 795832   |
| train/             |          |
|    actor_loss      | -0.0278  |  ← Policy confidente e deterministica
|    critic_loss     | 0.00188  |
---------------------------------
```

### Perché Abbiamo Raggiunto il 100%
La policy ha risolto in modo definitivo il **Deadlock Meccanico**. <br />
In precedenza, il robot rilasciava la presa quando la porta era a 0 rad ma il chiavistello interno (`latch_qpos`) era ancora compresso. 
Con l'introduzione di `is_waiting` e `wait_grip` nella FSM, l'agente ha compreso che deve aprire intenzionalmente le dita stando fermo per concedere al chiavistello lo spazio e il tempo di scattare (neutro). Solo quando il meccanismo è ingaggiato, l'agente indietreggia in Phase 4.

---

## 6. Analisi dei Log: La Dimostrazione Comportamentale

I log della sessione di verifica finale illustrano perfettamente il comportamento ideale appreso dalla rete.

### Transizione Perfetta in Phase 2 (PUSH)
```text
┌─────────┬────────┬────────┬───────┬───────────┬───────┬───────┬───────┬───────┐
│  PHASE  │  DIST  │   dZ   │ GRIP  │   PHYS    │ WIDTH │ ALIGN │ DOOR  │ CONF  │
├─────────┼────────┼────────┼───────┼───────────┼───────┼───────┼───────┼───────┤
│ 2:PUSH  │  0.059 │ -0.057 │ +1.00 │ PHYS_OK   │ 0.028 │  0.42 │  0.16 │  5/5  │
└─────────┴────────┴────────┴───────┴───────────┴───────┴───────┴───────┴───────┘
  ↳ REWARDS │ smoothness: -0.42 │ base: -0.43 │ dist_z: -0.86 │ door_prog: +65.56 │ TOT: +63.52
```
*Interpretazione:* Spinta decisa (`door_prog: +65.56` in un solo step), gripper serrato (`+1.00`), assenza di exploit densi. La rete punta unicamente alla chiusura.

### Risoluzione del Deadlock in Phase 3 (HOLD)
```text
┌─────────┬────────┬────────┬───────┬───────────┬───────┬───────┬───────┬───────┐
│  PHASE  │  DIST  │   dZ   │ GRIP  │   PHYS    │ WIDTH │ ALIGN │ DOOR  │ CONF  │
├─────────┼────────┼────────┼───────┼───────────┼───────┼───────┼───────┼───────┤
│ 3:HOLD  │  0.036 │ +0.036 │ +0.71 │ PHYS_OK   │ 0.025 │  0.97 │  0.00 │  0/5  │
└─────────┴────────┴────────┴───────┴───────────┴───────┴───────┴───────┴───────┘
  ↳ REWARDS │ smoothness: -0.19 │ base: -0.50 │ hold: +1.00 │ latch_wait: -7.86 │ wait_grip: -3.43 │ TOT: -11.57
  ↳ FSM LOG │ DROP: Grip opened / not phys closed
```
*Interpretazione:* La porta è chiusa (`DOOR: 0.00`). Inizia la fase di attesa del latch (`latch_wait` penalizza il fatto che il chiavistello non è ancora scattato). La penalità `wait_grip: -3.43` avverte l'agente che deve aprire il gripper (`+0.71` sta diventando negativo). L'agente risponde aprendo il gripper prima del termine della fase!

### Il Retreat Definitivo in Phase 4 (BACK)
```text
┌─────────┬────────┬────────┬───────┬───────────┬───────┬───────┬───────┬───────┐
│  PHASE  │  DIST  │   dZ   │ GRIP  │   PHYS    │ WIDTH │ ALIGN │ DOOR  │ CONF  │
├─────────┼────────┼────────┼───────┼───────────┼───────┼───────┼───────┼───────┤
│ 4:BACK  │  0.134 │ -0.113 │ -1.00 │ PHYS_OPEN │ 0.078 │  0.75 │  0.01 │  5/5  │
└─────────┴────────┴────────┴───────┴───────────┴───────┴───────┴───────┴───────┘
  ↳ REWARDS │ base: -0.10 │ hold: +0.99 │ ret_grip: +2.00 │ ret_rot: -0.38 │ latch_ret: -0.03 │ TOT:  +2.20
```
*Interpretazione:* Ritirata impeccabile. Il gripper è spalancato (`-1.00`), attivando il bonus `ret_grip: +2.00`. L'aspetto fondamentale è **`latch_ret: -0.03`** (quasi zero!), che indica che il chiavistello è ritornato alla posizione neutra perfetta e la porta è fisicamente ancorata. Non c'è alcun rimbalzo o riapertura in corso.

---

## Conclusioni
Il modello Franka Panda RL è ora capace di:

1. Navigare robustamente fino alla maniglia senza urti involontari.
2. Spingere la porta compensando le variazioni spaziali ad alta velocità (no-slip grip).
3. Contrastare le reazioni vincolari del simulatore (hard bounce).
4. Comprendere la semantica nascosta del chiavistello, attendendo e aprendo le dita per evitare il fallimento meccanico.
5. Ritirarsi fluidamente ripristinando una configurazione sicura.

Ciò che bisogna migliorare è:

1. Grip leggermente più "profondo" nella fase di transizione tra Phase 2 e Phase 3
2. Nella hold l'intero braccio dovrebbe essere più stabile, conmeno oscillazioni residue. (Rotazione polso)
3. Confermare l'attesa dei due secondi dalla Phase 3 alla Phase 4.
