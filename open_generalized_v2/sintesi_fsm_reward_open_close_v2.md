# FSM e Reward/Penalty Machine — sintesi (close_v2 + open_v2)

Documento breve di riferimento. Per il dettaglio completo: `update_v2.md` (chiusura) e
`update_open_v2.md` (apertura).

---

## 1. La FSM (macchina a stati a soglie adattive)

Entrambi i task usano una FSM a **4 fasi**, formalizzabile come *options* `[1]`: ogni fase
è un'opzione con la sua policy interna e la sua condizione di terminazione (le soglie). La
novità rispetto alla v1 è che le soglie non sono costanti ma **adattive alla fisica
corrente** (frizione, raggio maniglia, rigidità latch), fornita dal domain randomizer — è
ciò che permette la generalizzazione.

| Fase | Chiusura | Apertura | Cosa fa |
|------|----------|----------|---------|
| 0 | REACH | REACH | Avvicinati e afferra la maniglia |
| 1 | PUSH | PULL | Muovi la porta verso il bersaglio |
| 2 | HOLD | HOLD_OPEN | Mantieni la porta al bersaglio per un tempo adattivo |
| 3 | RETREAT | RETREAT | Rilascia e allontanati (logica deterministica env-level) |

**Soglie adattive (identiche nei due task):**
- distanza di presa: `1.5 × radius + 0.005` (adattiva al raggio maniglia);
- soglia gripper: `0.75 − 0.10 × norm_friction` (adattiva alla frizione);
- timer di HOLD: `base + extra(stiffness)` (adattivo alla rigidità del latch);
- conferma presa: **5 step consecutivi** di presa valida (REACH→fase successiva).

**L'unica vera inversione logica** è la transizione di fase 1→2:
- **chiusura:** `PUSH→HOLD` quando `door_angle ≤ success_angle` (porta arrivata a ~0);
- **apertura:** `PULL→HOLD_OPEN` quando `door_angle ≥ goal_angle − tol` (porta arrivata al
  bersaglio alto) **E** la presa è chiusa sopra la soglia **adattiva** `g_thresh`.

È la stessa condizione con la disuguaglianza invertita. L'arco di ritorno (fase 1 → REACH
su presa persa) è identico nei due task.

> **Nota (§1.30, apertura).** Il gate di presa di `PULL→HOLD_OPEN` usa la soglia *adattiva*
> `g_thresh` (come `REACH→PULL`), non un letterale `0.80`. Con il grip-lock che floora il
> comando a `g_thresh + 0.10`, un `0.80` fisso bloccava la transizione per le maniglie a
> bassa frizione (porta al goal ma episodio fermo in PULL) — era l'unico residuo di "numero
> magico" incoerente con l'impianto adattivo, ed è stato eliminato.

```
        ┌─ presa persa ──────────────────┐
        ▼                                │
  ┌─────────┐  presa confermata   ┌──────────┐  porta al goal  ┌────────────┐  hold done  ┌──────────┐
  │  REACH  │  (5 step adattivi)  │ PUSH/PULL│  (≤0  o  ≥goal) │ HOLD/HOLD_  │ (timer adatt)│ RETREAT  │ ─► SUCCESS
  │         │ ───────────────────►│          │ ───────────────►│   OPEN      │ ────────────►│          │
  └─────────┘                     └──────────┘                 └────────────┘              └──────────┘
```

---

## 2. La Reward/Penalty Machine

Principio: **competenza del task → reward potential-based; qualità del movimento →
logica deterministica env-level (reward zero)**.

### 2.1 Shaping potenziale (Ng 1999, `[3]`)
A ogni step si aggiunge `F = γ·Φ(s') − Φ(s)`, che **non cambia la policy ottima**
(invarianza). Il potenziale Φ cresce lungo le fasi (REACH < fase1 < fase2 < RETREAT), così
lo shaping "tira" verso il completamento. Due accorgimenti chiave:
- pesi **piccoli** `O(1–5)`: con pesi grandi la "tassa di sosta" `(γ−1)·Φ` rende negativo
  il valore di HOLD/RETREAT;
- **Φ = 0 in REACH**: annulla quella tassa così la policy può restare sulla maniglia i 5
  step necessari a confermare la presa.

L'unica differenza tra i task è il *verso* del progresso: il potenziale di fase 1 premia
l'avvicinamento a `door=0` (chiusura) o a `door=goal_angle` (apertura).

### 2.2 Termini densi per fase

**REACH (identico nei due task)** — porta il braccio alla maniglia (il solo Φ non basta):
penalità di distanza 3D/XY/Z, penalità geometriche di approccio (non troppo sotto/sopra la
maniglia), gestione del gripper (aperto se lontano, premio alla chiusura se vicino).

**Fase 1 — PUSH (chiusura) / PULL (apertura)** — il cuore del task è il **progresso con
ratchet** (`door_prog`):
- chiusura: premia l'angolo *nuovo* verso 0; `min_door_angle` **scende** soltanto;
- apertura: premia l'angolo *nuovo* verso il goal; `max_door_angle` **sale** soltanto.

Il ratchet anti-exploit (oscillare avanti/indietro non ri-premia) è identico; cambia solo
la direzione. Si aggiungono penalità per non perdere la maniglia e premio al mantenimento
del contatto.

**Fase 2 — HOLD (chiusura) / HOLD_OPEN (apertura)** — stabilità al bersaglio. **Qui la
simmetria si ROMPE**, ed è la lezione fisica più importante del progetto.
- **Chiusura:** il bersaglio `door≈0` è il **punto di equilibrio** della porta. Usa un
  blocco di stabilizzazione forte — `hold = 1 − err`, `hold_bounce = −20 × err`,
  `hold_veldamp = −25 × |door_qvel|` — che all'ottimo (`err≈0`, `qvel≈0`) vale **zero**:
  non costa nulla, e porta la porta a `door_end ≈ ±0.004` ferma. Fa 100% true success.
- **Apertura:** il goal alto è **fuori equilibrio** — la molla di richiamo ritira la porta
  di 0.024–0.050 rad in modo *inevitabile*. Un tentativo (§1.28) di copiare `hold_bounce`/
  `hold_veldamp` qui **regrediva** il rollout: quei termini punivano la policy per la fisica.
  La soluzione (§1.29) è l'**hold piatto** (`hold = +1.0` entro tolleranza, guida dolce di
  peso 1 fuori), **senza** bounce/veldamp, e chiudere il residuo deterministico per via
  **geometrica**: `open_tol_rad = 0.05`, larga quanto la deriva fisica reale (la chiusura
  "si permette" 0.03 perché 0.03 è la finestra attorno all'equilibrio, deriva nulla).

In comune restano i termini che **non** combattono la molla: presa mantenuta, anti-apertura
del gripper, braccio fermo, maniglia vicina. La differenza è il bersaglio dell'errore
(`|door_qpos|` chiusura vs `|door_angle − goal_angle|` apertura) **e** il fatto che i due
termini di stabilizzazione vivono solo dove il bersaglio è un equilibrio.

**Fase 3 — RETREAT (entrambi)** — rilascio e ritiro:
- premio alla stabilità (la porta deve restare al bersaglio), penalità se il moto va nel
  verso sbagliato (ri-apertura per la chiusura, **ri-chiusura** `w_door_regress` per
  l'apertura);
- **`latch_ret = −1.0 × |latch_qpos|`** (identico nei due task): penalizza la leva ancora
  ruotata, insegnando ad accompagnarla a posto prima di staccarsi.

### 2.3 Override deterministici del RETREAT (env-level, reward zero)
Questi non sono reward: modificano l'azione *prima* dell'unico `sim.step()`.
- **rilascio pulito**: apre il gripper solo quando le dita sono libere dalla maniglia;
- **rampa di avvio** del ritiro (avvio morbido fermo→policy);
- **accompagnamento leva**: tiene la presa e congela il braccio finché la molla riporta la
  leva a neutro.

**Differenza dell'apertura (cap temporale):** nella chiusura la porta va a 0 e il latch
torna neutro da solo, quindi l'accompagnamento termina. Nell'apertura la porta resta aperta
e la leva può **non** tornare neutra → senza un limite il braccio resterebbe aggrappato per
sempre. Si aggiunge perciò un **cap** (`retreat_latch_max_steps`): dopo N step si procede
comunque a rilascio + ritiro.

### 2.4 Terminazione dell'episodio
- **chiusura:** `door < 0.03 AND latch < 0.08` (raggiungibile: a porta chiusa il latch
  scatta a zero per fisica);
- **apertura:** `retreat sostenuto AND door ≥ goal − tol` — **senza** gate sul latch,
  perché a porta aperta la leva non torna a zero da sola e bloccherebbe l'episodio.

La lezione: la condizione di terminazione va allineata allo **stato fisicamente
raggiungibile** dal task, non copiata alla lettera.

---

## 3. In una riga

Stessa impalcatura (FSM a soglie adattive + reward potenziale invariante + override
deterministici nel RETREAT). L'apertura riusa quasi tutto della chiusura; cambiano solo il
**verso** (ratchet che sale, progresso verso `goal_angle`), un **iperparametro di
esplorazione** (`target_entropy` più alto, perché afferrare la maniglia di lato è più
difficile da innescare) e tre soglie **riallineate alla fisica del bersaglio non-equilibrio**:
la **terminazione** (porta aperta invece di latch neutro), la **tolleranza di successo**
(`open_tol = 0.05`, larga quanto la deriva della molla — niente penalità anti-molla, §1.29) e
il **gate di presa** `PULL→HOLD_OPEN` (`g_thresh` adattivo invece del letterale `0.80`, §1.30).
La lezione trasversale: in un sistema adattivo, ogni soglia è funzione della fisica corrente,
mai un numero copiato per simmetria.
