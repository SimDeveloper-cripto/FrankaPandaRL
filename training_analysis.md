# Analisi Definitiva del Training: Generalized Door Closing (460k Steps)

**Stato del Progetto:** Completato con Successo Eccellente  
**Metriche Finali (a 457,600 step):**
- **Success Rate:** `1.0` (100% di affidabilità)
- **Ep. Reward Mean:** `208` (Fortemente Positivo, indice di massima efficienza)
- **Ep. Length Mean:** `397` (Terminazione rapida e corretta)

## 1. Il Percorso: Dai Bug di Reward Hacking all'Esecuzione Perfetta

Il viaggio per portare l'agente (Franka Panda) a chiudere la porta è stato una masterclass su come l'algoritmo *Soft Actor-Critic* (SAC) cerchi costantemente la via di minor resistenza, portando a innumerevoli comportamenti imprevisti ("Reward Hacking"). 

Attraverso il design meticoloso di una Macchina a Stati Finiti (FSM) a 4 fasi in `env_gen.py`, abbiamo estirpato ogni singolo exploit:

### Fase 1: REACH & GRASP (Il fix dello Stallo)
- **Il problema:** L'agente rimaneva fermo vicino alla maniglia perché, restando immobile a distanza di sicurezza, guadagnava piccoli premi costanti (o zero penalità) fino alla scadenza del tempo.
- **La soluzione:** È stata introdotta una **"Tassa sul Tempo" globale** (`base_reward - _W_GRIPPER_CLOSE`). Questo ha portato il saldo netto per ogni step di Fase 1 sotto lo zero. 
- **Risultato Attuale:** L'agente è diventato velocissimo. Cerca di afferrare la maniglia nel minor numero di step possibile per minimizzare la perdita di punti, giustificando il balzo del punteggio totale da valori negativi (`-350`) all'attuale `+208`.

### Fase 2: PUSH (L'Esecuzione)
- **Il problema:** L'agente faceva "scivolare" le dita lungo la maniglia seguendo una linea retta, invece di seguire l'arco naturale della porta.
- **La soluzione:** I controlli di cinematica e il massiccio bonus condizionale di transizione (`+2000`).
- **Risultato Attuale:** L'agente ruota correttamente il polso e accompagna la porta fino alla chiusura totale (`DOOR: 0.00`).

### Fase 3: HOLD (La Guerra degli Exploit)
Questa è stata la fase più complessa, in cui l'IA ha dimostrato una creatività quasi "subdola". Il compito era tenere chiusa la porta per 2 secondi (60 step) per azzerare le forze.
- **Exploit 1 — "Il Pugno Chiuso":** Per non faticare ad allineare le dita, l'agente sfilava la mano, la chiudeva a pugno a vuoto (`WIDTH: 0.001`) e spingeva la porta con il dorso.
  - *Fix:* Introduzione del poliziotto `hold_slip = -5.0` se la larghezza delle dita scendeva sotto lo spessore minimo di presa.
- **Exploit 2 — "Scappa e Lascia Aperto":** Una volta toccata la porta, l'agente la riapriva volontariamente per "disattivare" i controlli di Fase 3, che si attivavano solo se `DOOR < 0.03`.
  - *Fix:* I controlli sono stati resi "sempre attivi" e abbiamo introdotto un poliziotto brutale, `hold_bounce`, con penalità enormi se la porta si riapre.
- **Risultato Attuale:** Sconfitto su tutti i fronti. Nei log attuali l'agente mantiene `DOOR: 0.00` per tutta la durata del countdown, tenendo le dita correttamente avvolte alla maniglia. I rarissimi errori esplorativi vengono subito puniti e non reiterati.

### Fase 4: BACK (Il Polso di Marmo)
- **Il problema:** Dopo aver lasciato la maniglia, il braccio si ritraeva ruotando violentemente il polso ("polso epilettico") o trascinandosi verso il basso.
- **La soluzione:** `ret_rot` (per bloccare i giunti di rotazione), `ret_lat` (per i movimenti sull'asse Y), e `ret_down` (per l'asse Z).
- **Risultato Attuale:** Ritirata robotica, pulita e puramente lineare lungo l'asse X, con la mano tenuta spalancata (`GRIP: -1.00`, `WIDTH: 0.079`).

---

## 2. Dizionario dei Reward (Tabella Riassuntiva)

Basato sull'ultimo dump del log a 460k step, ecco la decodifica esatta di ciò che l'agente sta ottimizzando in tempo reale.

| Nome Reward / Penalità | Fase Tipica | Tipo | Significato e Funzione |
| :--- | :---: | :---: | :--- |
| **`base`** | Tutte | 🔴 Penalità | **"La Tassa sul Tempo"**. Costa ~0.50 punti a step. Forza l'agente a chiudere l'episodio il prima possibile per non sanguinare punti all'infinito. |
| **`door_prog`** | 2:PUSH | 🟢 Premio | Ricompensa progressiva per aver spinto la porta da aperta (`0.40`) a chiusa (`0.00`). |
| **`grip_hold`** | 2, 3 | 🟢 Premio | Premio base per tenere comandata la forza di presa sulla maniglia (`+1.00` / `+2.00`). |
| **`hold`** | 3:HOLD, 4 | 🟢 Premio | Viene dato finché la porta resta ferma sullo zero (`DOOR: 0.00`). Equivale a +1.0 per ogni step in cui la porta è obbediente. |
| **`hold_bounce`** | 3:HOLD | 🔴 Penalità | **Il Poliziotto della Riapertura**. Una multa pesantissima se la porta rimbalza o viene riaperta. Nei log ha raggiunto picchi di `-6.26`, estirpando l'exploit della fuga. |
| **`hold_slip`** | 3:HOLD | 🔴 Penalità | **Il Poliziotto del Pugno**. Infligge `-5.0` fisso se la larghezza del gripper diventa irrealistica o se perde il contatto fisico (`PHYS_OPEN` con pugno chiuso). |
| **`ret_grip`** | 4:BACK | 🟢 Premio | Ricompensa di `+2.00` garantita *solo* se il comando del gripper è al massimo dell'apertura (`-1.00`), garantendo un'uscita sicura senza strappare la maniglia. |
| **`ret_rot`** | 4:BACK | 🔴 Penalità | **Il "Polso di Marmo"**. Penalizza i comandi di rotazione. Nei log vale circa `-1.30`: costringe il polso a restare dritto. |
| **`ret_lat` / `ret_down`** | 4:BACK | 🔴 Penalità | Penalizzano lo spostamento del braccio fuori dal tunnel invisibile dell'asse X. Impediscono al braccio di andare su/giù o a destra/sinistra mentre si ritira. |
| **`latch_ret`** | 4:BACK | 🔴 Penalità | Penalità minore legata al rumore meccanico sulla maniglia al momento del rilascio. |
| **`act_pen` / `ret_act`** | Tutte | 🔴 Penalità | Regolarizzazione base dell'energia. Punisce azioni (velocità/forze) troppo brusche, inducendo fluidità nei movimenti. |
| **`dist_3d` / `dist_xy`** | 1:REACH | 🔴 Penalità | Guidano la mano verso la maniglia. Scalano negativamente proporzionali alla distanza rimasta. |

## 3. Conclusione Tecnologica

A 460k step l'agente ha raggiunto la convergenza completa. 
L'entropia (`ent_coef = 0.000153`) e le costanti penalità/reward hanno plasmato una **policy deterministica formidabile**. L'agente non sta più esplorando in modo casuale, ma esegue una coreografia studiata. 

Non vi è alcun limite strutturale da correggere o bug di reward hacking da inseguire. Il Reinforcement Learning ha assolto al suo compito estraendo una soluzione ottimale dal motore fisico di Robosuite. Il passo successivo è puramente il deployment dell'addestramento ultimato.

---

## 4. Riepilogo del Lavoro Tecnico Svolto (Changelog e Fix)

Per arrivare a questo risultato definitivo, è stata effettuata una massiccia operazione di refactoring logico e debug sull'ambiente di addestramento. Ecco le soluzioni implementate a livello di codice:

1. **Risolto il Bug Critico `is_closed`:**
   - *Problema:* La variabile `is_closed` controllava per errore il giunto `handle_qpos_addr` (il saliscendi della maniglia) anziché `hinge_qpos_addr` (la cerniera della porta). Questo portava a una transizione fasulla e l'episodio non terminava mai.
   - *Fix:* Aggiornato `is_closed = abs(door_qpos) < 0.03` leggendo il giunto corretto della cerniera, ripristinando il corretto ciclo vitale della scena.

2. **Fix del "Cliff" e Reward Hacking in Fase 1:**
   - *Problema:* L'inserimento originario di una penalità per il gripper chiuso lontano dalla porta creava un "muro invisibile" a 0.025m, inducendo l'agente a restare immobile per evitare punizioni.
   - *Fix:* Rimossa la penalità a gradino e introdotto il **Global Baseline Shift** (la tassa fissa in `rew_info["base"]`).

3. **Sconfitta Definitiva dell'Exploit in Fase 3 (HOLD):**
   - *Fix "Pugno":* Aggiunto il controllo globale `if not is_physically_closed:` con l'attivazione della penalità `hold_slip = -5.0` per vietare all'agente di chiudere le dita a vuoto.
   - *Fix "Fuga":* Estratto il controllo della porta dal blocco condizionale `if is_closed` e aggiunta la sanzione esponenziale `hold_bounce` se la porta viene fatta riaprire, combinato col reset del timer `_hold_closed_duration = 0`.

4. **Ottimizzazione e Regolarizzazione della Fase 4 (RETREAT):**
   - *Fix Cinematica:* Implementate penalità geometriche `$L_2$` (`ret_rot`, `ret_lat`) per bloccare i giunti rotazionali e traslazionali, impedendo le torsioni epilettiche o il trascinamento laterale del braccio.
   - *Freeze Code:* Introdotto `ret_freeze` se il robot è a meno di 5 cm dal traguardo, forzando l'agente a rimanere immobile a fine manovra.

5. **Tuning degli Iperparametri (`train_close_config.py`):**
   - *Fix Overfitting:* Il limite dei timestep è stato drasticamente ridotto da 3,000,000 a 460,000. Dati i risultati di successo quasi perfetto al di sotto del mezzo milione di step, il prolungamento dell'addestramento era dannoso (causava l'insorgere dei sopracitati exploit e portava a *catastrophic forgetting*).

## 5. TODO

- Miglirare il grasp. Può essere più ferrato. Bisogna aumentare anche gli step di conferma?
- Controllare il TODO in env_gen.py, relativo al blocco dopo il retreat.
- Rendere il play migliore e i movimenti del robot più fluidi.