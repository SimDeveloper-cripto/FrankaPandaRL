# Report suite di test — task di APERTURA generalizzata v2

**Sessione del 29 luglio 2026, 08:47** · commit `9543781` · preset **`full`** · 6/6 batterie · durata 8 min 3 s

| | |
|---|---|
| **Repo** | `/Users/simone/workspace/FrankaPandaRL` |
| **Modello** | `runs/open_gen_v2/best_model.zip` |
| **Curriculum** | 1.0 — posa variabile (l'apertura v2 è addestrata solo a questo livello) |
| **Ambiente** | Python 3.10.20 · macOS arm64 · robosuite 1.5.1 · SB3 2.7.1 · torch 2.10.0 · numpy 2.2.6 · scipy 1.15.3 |
| **Episodi** | evaluate 200 + 200 · phase 100 · robustness 300 · ablation 50 × 8 (appaiati) |

> **Come si legge.** Tutti i tassi sono **stime a intervallo**: il numero puntuale da solo non
> è interpretabile. `[a, b]` è l'intervallo di confidenza al 95%.
> Metriche e lettura → `guida_risultati_test_suite_openv2.md`.
> Analisi estesa batteria per batteria → `risultati_test_suite_openv2.md`.
> Piano di correzione lato training → `piano_correzione_training_openv2.md`.

---

## Verdetto

**La policy risolve la manipolazione al 100% e si ritira correttamente. Non modula però
l'apertura sul goal richiesto: spalanca fino al fine corsa.**

| criterio | risultato (n=200, det) | |
|---|---|---|
| apre la porta fino al goal | **100.0%** [98.1, 100.0] | ✅ capacità acquisita |
| la porta **resta** al goal e la leva torna a riposo | **83.0%** [77.2, 87.6] | ⚠️ 17% fuori bersaglio |
| + ritiro effettivamente completato | **83.0%** [77.2, 87.6] | ✅ il ritiro non perde nulla |
| braccio fermo sulla maniglia | **0/196** [0.0, 1.9] | ✅ problema §1.55 risolto |
| **riferimento banale** (ignora il goal, spalanca sempre) | **80.5%** [74.5, 85.4] | 🔴 **guadagno della policy: +2.5 punti** |

**L'ultima riga è il numero decisivo.** Una politica costante che non guarda mai il goal
otterrebbe l'80.5% sugli stessi episodi, perché con `open_tol = 0.05` il fine corsa cade dentro
la tolleranza per la maggior parte dei goal campionati. Il `true_success` misura quindi **in
gran parte la geometria del compito, non l'apprendimento**: è un difetto di **specifica della
metrica**, non della policy.

### Stato per batteria

| # | Batteria | Tipo di test | n | Esito | Tempo |
|---|---|---|---|---|---|
| 0 | `test_retreat_overrides` | white-box sul sorgente | — | ✅ **16/16 PASS** | 0.0 s |
| 1 | `physics_unit_tests` | property-based sull'ambiente | 200 reset | ✅ **4/4 PASS** | 9.7 s |
| 2 | `evaluate_policy` | stima statistica | 200+200 | ⚠️ true success 83% (banale 80.5%) | 136.9 s |
| 3 | `phase_diagnostics` | diagnostica di fase | 100 | ⚠️ errore preesistente al ritiro | 35.4 s |
| 4 | `robustness_analysis` | osservazionale stratificato | 300 | 🔴 fragilità **solo** sul goal | 92.7 s |
| 5 | `ablation_study` | esperimento controllato appaiato | 50×8 | ✅ 3 override portanti | 219.7 s |

### I tre difetti del sistema

| # | difetto | frequenza | causa nel codice |
|---|---|---|---|
| **1** | **Overshoot al fine corsa** | 24/200; **42%** di successo sui goal bassi | `pull_progress_cap_at_goal = False` + gate unilaterali + reward piatto dentro la tolleranza |
| **2** | **Stallo in HOLD_OPEN** | 4/200 (600 step bruciati) | timer di hold decrementato di 7/step senza via d'uscita |
| **3** | **Leva oltre il neutro** | 6/200 (`ESOGENA`) | riporto §1.46 senza banda morta, smorzamento del giunto = 0 |

### Confronto con la sessione precedente

| | 14:30 (`standard`, n=100) | 21:15 (`full`, n=200) |
|---|---|---|
| true success det | 80.0% [71.1, 86.7] | **83.0%** [77.2, 87.6] |
| semiampiezza dell'intervallo | ±7.8 punti | **±5.2 punti** |
| fascia peggiore del goal | 32% (n=25) | **42%** (n=50) |
| det vs sto (McNemar appaiato) | p = 0.021 | **p = 0.0037** |

I **100 episodi condivisi sono identici a meno di 1e−12**: l'ambiente non è cambiato, le
differenze vengono solo dal campione più grande. **Ogni intervallo si è ristretto e nessuna
conclusione si è invertita.**

---

## 0. White-box — override deterministici del RETREAT

**Tipo:** verifica di *implementazione*, non di comportamento. Estrae dal sorgente reale di
`env_v2.py` il ramo RETREAT e ne verifica le proprietà logiche in un harness fittizio, senza
MuJoCo e senza modello.
**Perché serve:** questi override non aggiungono reward, quindi un loro bug **non comparirebbe
mai nella reward curve**.

**Esito: 16/16 PASS** in 4 ms.

| Gruppo | Proprietà | Esito |
|---|---|---|
| Config | §1.17 · §1.21 (8 step) · §1.43 · §1.46 · §1.50 attivi | ✅ A |
| Riporto leva §1.46/§1.49 | traslazione tangente ×0.150 + rotazione polso ×0.5; contatori azzerati | ✅ B |
| Gabbia §1.50 | bang-bang sulla larghezza reale attorno a 0.055 m | ✅ C |
| Fine riporto §1.47 | chiude il riporto **e invalida** `retreat_pos` obsoleto | ✅ D |
| Rampa riporto §1.51 | satura a 0.600 dopo 4 step | ✅ E |
| Guardia velocità §1.51 | smorza al pavimento 0.25, **mai a zero** | ✅ F |
| Rilascio pulito §1.17 | dita non libere → braccio congelato + gripper aperto | ✅ G |
| Rampa avvio §1.21 | ×1/8 · ×5/8 · piena oltre 8 · interruttore a 0 | ✅ H I J K |
| Sfilamento §1.43 | guida verso `retreat_pos`, polso azzerato; cede il passo dopo 0.15 m | ✅ L M |
| Interruttori | `retreat_restore_enabled` · `retreat_escape_enabled` = False | ✅ N O |
| Sicurezza | dopo il rilascio il gripper è **sempre** in apertura | ✅ P |

→ La macchina del ritiro è implementata come documentata. È il presupposto che rende
interpretabile la §5, e concorda con il dato empirico (0 incastri su 196).

---

## 1. Physics unit tests

**Tipo:** property-based, deterministici, senza policy. Verificano che il task sia
**fisicamente ben posto**: se falliscono, le batterie successive misurano l'ambiente credendo
di misurare l'agente.

**Esito: 4/4 PASS.**

| Test | Misura | Risultato | Lettura |
|---|---|---|---|
| **T1** molla del latch | ritorno a neutro | sotto **0.15** in **15 step (0.50 s)**; sotto **0.05** in 18 step; k=1.000, **c=0.000**; finale −0.0226 | ✅ la leva torna **da sola**. Smorzamento nullo → oltrepassa lo zero: è il meccanismo del difetto 3 |
| **T2** ritenzione | la porta resta al goal? | deriva a riposo **−1.4·10⁻⁹ rad** in 60 step; con impulso −0.3 rad/s percorre **0.1786 rad**; k del cardine = **0.000** | ✅ task ben posto. Ma **sensibile agli urti**: 3,5× la tolleranza |
| **T7** randomization | 5 parametri nei range | raggio [0.0140, 0.0280] · frizione [0.304, 1.198] · latch× [0.515, 1.994] · damp× [0.307, 1.480] · massa× [0.501, 2.000] | ✅ |
| **T8** goal | campionamento | [0.3402, 0.3992], μ 0.3704, σ 0.0166 ⊆ [0.85, 1.00]×0.400 | ✅ |

Misura T2 validata: `valid = True`, nessun contatto braccio↔porta, seed 12345, 1 tentativo.

> **T2 falsifica due ipotesi scritte nel codice.** `reward_v2.py:168–175` attribuisce l'errore
> finale al fatto che «la molla la richiama» durante il RETREAT; `reward_v2.py:198–207` (§1.29)
> giustifica l'hold piatto e l'allargamento di `open_tol` dicendo che «la molla ritira la porta
> di 0.024–0.050 rad in modo fisicamente inevitabile». La misura dice che il cardine **non ha
> molla** (rigidità 0.000) e che la porta lasciata al goal **non si muove**. Quei
> «0.024–0.050 rad» coincidono numericamente con l'overshoot misurato (media +0.0243, coda
> ~0.05): **il fenomeno attribuito alla molla era l'overshoot stesso, letto con il segno
> sbagliato.** Il rimedio proposto (il cap) resta valido; cambia il meccanismo.

### Un risultato nascosto in T7: la finestra della FSM è mal calibrata

| | valore |
|---|---|
| range **reale** della frizione | [0.304, 1.198] → base ≈ **1.0** |
| finestra assunta dal config | `fsm_friction_min = 0.24`, `fsm_friction_max = 0.96` → base 0.8 |
| **campioni sopra il massimo** | **28.0%** |

In oltre un quarto degli episodi la normalizzazione satura a 1 e `grip_thresh` resta bloccata
al minimo (0.65): **l'adattività alla frizione dichiarata in §3.1 non è operativa** in quella
frazione. Impatto pratico modesto (la frizione non è un asse di fragilità), ma è una
discrepanza fra ciò che il progetto afferma e ciò che il codice fa. Correzione di due righe.

---

## 2. Valutazione rigorosa (deterministica + stocastica)

**Tipo:** stima statistica su 200 episodi seedati per modalità. Wilson per i tassi, IQM +
bootstrap e CVaR per le grandezze continue, tre livelli di successo più il riferimento banale.

| Metrica | Deterministico | Stocastico | Lettura |
|---|---|---|---|
| `success` (permissivo) | 100.0% [98.1, 100.0] | 100.0% [98.1, 100.0] | apre **sempre** |
| `true_success` | **83.0%** [77.2, 87.6] | **91.0%** [86.2, 94.2] | ⚠️ vedi nota det/sto |
| `clean_success` | **83.0%** [77.2, 87.6] | **90.5%** [85.6, 93.8] | ✅ ≈ true success |
| **riferimento banale** | **80.5%** [74.5, 85.4] | 80.5% | 🔴 guadagno +2.5 / +10.5 pt |
| lunghezza IQM (media) | 142.9 (156.3) | 159.6 (178.5) | il divario è dovuto ai 4 stalli |
| lunghezza CVaR 10% | 286.7 | 339.2 | coda dominata dagli stalli |
| `open_error` finale IQM | 0.0278 | 0.0241 | tolleranza = 0.05 |
| **errore CON SEGNO** | **+0.0243 — 92% oltre il goal** | **+0.0198 — 86% oltre** | 🔴 **troppo aperta**, mai troppo chiusa |
| `open_error` **minimo** IQM | **0.0013** | **0.0005** | la porta **transita** dal bersaglio e prosegue |
| allontanamento IQM | 0.1856 m | 0.1918 m | ✅ 3× la soglia (0.06) |
| fermo sulla maniglia | 0/196 [0.0, 1.9] | 0/198 [0.0, 1.9] | ✅ |

**Esiti (deterministico)**

| Esito | n | 95% CI |
|---|---|---|
| SUCCESS | 166 | [77.2, 87.6] |
| **RETREAT overshoot (oltre il goal)** | **24** | [8.2, 17.2] |
| RETREAT latch not neutral | 6 | [1.4, 6.4] |
| HOLD_OPEN regress / timeout | 4 | [0.8, 5.0] |
| **RETREAT door regress (sotto il goal)** | **0** | [0.0, 1.9] |
| REACH timeout · PULL timeout · stuck · RETREAT timeout | 0 | — |

**Terminazioni (det):** `PULITA` 190 · `ESOGENA` 6 · `HARD-CAP` 0 · `troncata` 4.
In stocastico: 195 · 3 · 0 · 2.

**Tempi per fase (mediana):** REACH 18 · PULL 18 · HOLD_OPEN 33.5 · RETREAT 72 step.

> **La manipolazione è risolta.** Zero `REACH timeout`, zero `PULL timeout`, 196/200 episodi
> arrivano al RETREAT: afferrare una maniglia di raggio variabile di un fattore 2, con frizione
> variabile di un fattore 4, su una porta di massa variabile di un fattore 4, e aprirla fino al
> goal, **riesce sempre**.

> **190 terminazioni pulite ma 166 successi.** I 24 di differenza hanno un ritiro impeccabile e
> la porta fuori bersaglio: senza tenere separate le due classificazioni sarebbero stati
> contati come successi.

> **La policy stocastica batte la deterministica, e ora in modo netto.** Stessi seed →
> confronto appaiato: **22 casi in cui riesce solo la stocastica contro 6 in cui riesce solo la
> deterministica**, **McNemar esatto p = 0.0037**. È la firma del difetto 1: l'argmax
> deterministico satura al fine corsa nel **79%** degli episodi (entro 5 mrad dal cap) contro
> il **46%** della stocastica. Rispetto al banale, la stocastica guadagna +10.5 punti contro i
> +2.5 della deterministica: **il rumore di esplorazione contiene più informazione sul goal di
> quanta ne usi la policy deterministica.**

---

## 3. Diagnostica fasi HOLD_OPEN / RETREAT

**Tipo:** stima statistica a grana fine su 100 episodi deterministici con tracce passo-passo.
Risponde a «*dove* si rompe», dopo che `evaluate` ha detto «*quanto* spesso».

| Metrica | Valore | Lettura |
|---|---|---|
| **T3** ‖azione braccio‖ in HOLD_OPEN | IQM **0.392** [0.386, 0.397] su 4040 step; **0.2%** < 0.05; **69.0%** > 0.30 | ⚠️ il reward premia il braccio fermo, ma la policy **non sta mai ferma**: la porta continua a essere sollecitata |
| **T4** torsione polso in RETREAT | IQM **0.686** su 7075 step; 99.2% > 0.1 | ✅ l'override §1.46 **è attivo**. ⚠️ è comando dell'env, non della policy |
| **T5a** `open_error` alla transizione | media **0.0212** ± 0.0147; **3.0%** [1.0, 8.5] oltre tolleranza | 🔴 **l'errore esiste già prima del ritiro** |
| **T5b** `latch` alla transizione | media **+1.120** ± 0.142, range [+0.709, +1.408] | la leva è **sempre quasi a fondo corsa**: è il carico che il riporto deve scaricare |
| **T6** scostamenti in HOLD_OPEN | **per episodio: 9/100** [4.8, 16.2] OLTRE il goal, **3/100** sotto (gli stalli) | il bias è **unidirezionale** |
| **T7** allontanamento in RETREAT | IQM **0.1845 m**; CVaR 10% **0.1360 m**; **0/99** fermi (1 non raggiunge il RETREAT) | ✅ ritiro ampio e affidabile anche nella coda |

> **T5a è la misura spartiacque**: con l'errore già a 0.0212 rad *quando inizia il ritiro*,
> l'errore finale non nasce nel RETREAT. Insieme a T2 (nessuna deriva a riposo), localizza il
> problema nel PULL/HOLD_OPEN.

> **T3 è il segnale secondario.** Il reward prevede un premio per il braccio fermo durante
> l'hold: la policy lo incassa nello 0.2% dei campioni. Combinato con T2 (un impulso costa
> 0.18 rad) e con le punte di velocità del cardine fino a 1.05 rad/s, è la catena causale più
> plausibile per gli episodi che scendono sotto il goal e per lo stallo del difetto 2.

> ⚠️ **Due correzioni attive da questa sessione.** T6 è ora citato **per episodio**: il
> conteggio per step non è episodio-pesato e i 3 episodi peggiori pesano il **91%** dei 640
> step registrati, tanto da invertire la proporzione fra sessioni con *n* diverso. T7 è ora
> calcolato **solo sugli episodi che raggiungono il RETREAT**: l'unico «fermo» del conteggio
> grezzo era un episodio che il RETREAT non l'aveva mai raggiunto, e il valore corretto (0/99)
> concorda con `evaluate` (0/196). Le altre cinque batterie sono identiche a meno di 1e−12
> rispetto alla sessione precedente.

---

## 4. Inviluppo operativo (robustezza)

**Tipo:** studio osservazionale stratificato su 300 episodi, **50 per fascia**. Si registra il
parametro *realizzato*, si raggruppa per quantili, si calcola il `true_success` per fascia.
Valutare **per regioni e non in media** è la pratica raccomandata per le policy addestrate con
domain randomization.

True success complessivo: **88.0%** [83.8, 91.2] · clean success **88.0%**.

| Asse | 6 fasce (true success) | Verdetto |
|---|---|---|
| **goal di apertura** | **42%** · 98% · 96% · 100% · 98% · 94% | 🔴 **fragilità netta e isolata** |
| frizione maniglia | 92 · 96 · 88 · 80 · 78 · 94 % | ✅ piatto |
| raggio maniglia | 86 · 80 · 84 · 100 · 92 · 86 % | ✅ piatto |
| rigidità latch | 92 · 86 · 96 · 86 · 90 · 78 % | ✅ piatto |
| smorzamento cerniera | 90 · 94 · 84 · 84 · 84 · 92 % | ✅ piatto |
| massa porta | 84 · 88 · 90 · 90 · 92 · 84 % | ✅ piatto |
| distanza porta | 90 · 88 · 88 · 88 · 88 · 86 % | ✅ piatto |

**Dipendenza dal goal, sui dati per-episodio (n=200, det):**

| goal richiesto | 0.340–0.350 | 0.350–0.364 | 0.364–0.375 | 0.375–0.388 | 0.388–0.400 |
|---|---|---|---|---|---|
| true success | **32%** | 95% | 100% | 95% | 92% |

Correlazione goal ↔ successo **r = +0.473**. **Il compito più facile è quello che riesce
peggio** — un andamento che non si spiega con la difficoltà fisica, solo con un difetto di
specifica.

> **Cautela.** Con 50 episodi per fascia l'intervallo è ≈ ±11 punti: le oscillazioni fra 78% e
> 100% sugli assi fisici **non sono interpretabili** — si esclude una dipendenza *grande*, non
> una dipendenza. Il crollo del goal, invece, è ampiamente sopra il rumore.

**Heatmap.** Griglia 3×3 con **n annotato in ogni cella**; in questa sessione ogni cella ha fra
**25 e 42 episodi** e **nessuna** è sotto la soglia di interpretabilità (10). Restano figure di
supporto: l'inviluppo 1-D dice le stesse cose con più forza, e non c'è evidenza di interazione
fra fattori.

---

## 5. Ablazione degli override del RETREAT

**Tipo:** esperimento controllato con **disegno appaiato** su 50 episodi per braccio (stessi
seed per baseline e varianti). Fisher esatto + Newcombe + Holm-Bonferroni su 7 confronti + h di
Cohen.

Baseline: true success **45/50 = 90.0%** [78.6, 95.7] · clean success 90.0%.

| Variante | true | Δ | 95% CI | p (Holm) | h | Δ lungh. | Δ allont. | Verdetto |
|---|---|---|---|---|---|---|---|---|
| `no_grip_lock` §1.18 | **0/50** | **−90.0** | [−95.7, −76.6] | **4.8e−22** | −2.50 | **+446.9** | **−0.180** | 🔴 **portante** |
| `no_latch_restore` §1.46 | **0/50** | **−90.0** | [−95.7, −76.6] | **4.8e−22** | −2.50 | −2.1 | **−0.153** | 🔴 **portante** |
| `no_all_overrides` | **0/50** | **−90.0** | [−95.7, −76.6] | **4.8e−22** | −2.50 | +446.9 | −0.180 | 🔴 |
| `no_escape` §1.43 | **3/50** | **−84.0** | [−90.9, −68.7] | **3.6e−18** | −2.00 | +7.0 | −0.026 | 🔴 **portante** |
| `no_cage` §1.50 | 36/50 | −18.0 | [−32.8, −2.5] | 1.2e−01 | −0.47 | **+18.1** | −0.005 | ⚙️ efficienza |
| `no_rampup` §1.21 | 43/50 | −4.0 | [−17.4, +9.4] | 1.0 | −0.12 | +0.0 | +0.004 | 🛡️ sicurezza |
| `no_clean_release` §1.17 | 45/50 | ±0.0 | [−12.7, +12.7] | 1.0 | +0.00 | −0.6 | −0.000 | ➖ ridondante |

**Tre modi di rottura diversi.** I Δ sul solo `true_success` dei portanti sarebbero quasi
indistinguibili (−90, −90, −84): sono le **metriche secondarie** a rivelare che si rompono in
tre punti diversi della catena.

- **`no_grip_lock`** → +447 step (tutti all'orizzonte), allontanamento a zero: la presa si apre
  in PULL e **la porta non viene mai aperta**. Rottura all'**inizio**.
- **`no_latch_restore`** → lunghezza invariata ma allontanamento quasi azzerato: il braccio
  arriva al RETREAT e **non riesce a sfilarsi** (sfilamento friction-limited sotto carico).
  Rottura alla **fine**.
- **`no_escape`** → si ritira quasi normalmente, ma **la leva non torna a casa** e l'uscita
  pulita non scatta. Rottura nella **condizione di terminazione**.

`no_all_overrides` coincide esattamente con i singoli portanti (−90): buon controllo di
coerenza interna.

**Il caso `no_cage`, da riportare con precisione.** L'intervallo di Newcombe [−32.8, −2.5]
**non contiene lo zero**, ma il p corretto con Holm vale **0.12**. Non è una contraddizione: i
due strumenti rispondono a domande diverse — l'intervallo è *per-confronto*, il p-value è
*per-famiglia* su 7 test. La conclusione onesta è **«effetto non stabilito»**. Il dato solido
su `no_cage` è un altro: **+18.1 step di lunghezza**, cioè un guadagno di **efficienza**, non
di affidabilità.

> **Due precisazioni obbligatorie.**
>
> 1. **«Nessun effetto» non è dimostrato** per i bracci non significativi. Con n = 50 gli
>    intervalli sono ampi decine di punti: si può escludere un effetto *grande*, non un
>    effetto. La formulazione corretta è «nessun effetto **rilevabile a questa numerosità**».
> 2. **È un'ablazione del controllore dispiegato, non dell'algoritmo di apprendimento.** Gli
>    override vengono disattivati su una policy *già addestrata con quegli override attivi*:
>    si misura quanto il comportamento finale ne dipende — domanda legittima e ben posta — ma
>    **non** che senza di essi non si sarebbe potuto imparare qualcos'altro.

---

## Diagnosi del difetto principale: overshoot al fine corsa

**Otto evidenze indipendenti, da quattro batterie diverse.**

| # | evidenza | batteria |
|---|---|---|
| 1 | **24 overshoot e 0 regressi**; errore con segno **+0.0243 rad**; **92%** oltre il goal | evaluate |
| 2 | angolo finale entro 5 mrad dal fine corsa nel **79%** degli episodi, *indipendentemente dal goal* | evaluate |
| 3 | `open_error` **minimo** IQM **0.0013 rad**: la porta **transita** per il bersaglio e prosegue | evaluate |
| 4 | il **riferimento banale** vale 80.5%: la metrica è quasi insensibile al goal | evaluate |
| 5 | fascia dei goal bassi al **42%** contro 94–100%; **r = +0.473** | robustness |
| 6 | errore già a **0.0212 rad** alla transizione HOLD_OPEN→RETREAT | phase T5a |
| 7 | in HOLD_OPEN **9 episodi su 100 escono dalla tolleranza verso l'alto**, 3 verso il basso | phase T6 |
| 8 | rigidità del cardine **0.000**, deriva a riposo **1.4·10⁻⁹ rad** | physics T2 |

**Causa nel codice — quattro elementi concorrenti:**

**(a)** Il progresso non è limitato al goal (`reward_v2.py:176–185`): con
`pull_progress_cap_at_goal = False`, ogni radiante in più è pagato fino al cap, con il peso più
alto della funzione di reward (300).

**(b)** Tutti i gate di successo sono **unilaterali** (`≥ goal − tol`): `fsm_v2.py:252`,
`fsm_v2.py:265`, `reward_v2.py:238`, `reward_v2.py:348`.

**(c)** Dentro la tolleranza il reward è **piatto** (`reward_v2.py:213`, `+1.0` costante):
non esiste alcun gradiente che spinga verso il **centro** del bersaglio invece che verso il
suo bordo.

**(d)** La specifica dichiarata è invece **bilaterale**: `config_v2.py` afferma «*il "successo
fisico" è |door_angle − goal_angle| ≤ tolleranza*».

> **La specifica è bilaterale, i gate e il reward sono unilaterali.** La policy ha ottimizzato
> correttamente ciò che le è stato chiesto davvero — *«apri almeno fino al goal»* — e non ciò
> che si voleva — *«apri fino al goal»*.

Il rimedio era già previsto (`reward_v2.py:174–175`). ⚠️ `pull_progress_cap_at_goal` **cambia
il reward: richiede un ri-addestramento.**

---

## Azioni raccomandate

| # | azione | effetto atteso | costo |
|---|---|---|---|
| **1** | **A/B con `pull_progress_cap_at_goal = True`**, valutando anche la bilateralizzazione dei gate | elimina l'overshoot; la fascia dei goal bassi risale dal 42%; il guadagno sul banale diventa significativo | **ri-addestramento** |
| **2** | **Guardia di stallo in HOLD_OPEN** | recupera il 2%; taglia la coda di lunghezza; impedisce a pochi episodi di dominare le statistiche per step | modifica minima |
| **3** | **Banda morta nel riporto §1.46** (fermarsi a `\|latch\| ≤ 0.15`) | recupera il 3% di `ESOGENA` | modifica minima |
| **4** | **Allineare la frizione**: `fsm_friction_min = 0.30`, `fsm_friction_max = 1.20` | riattiva l'adattività §3.1 nel 28% superiore del range | due righe |
| **5** | **Registrare l'azione pre-override** (`info["action_policy"]`) | rende T4 interpretabile | una riga |
| **6** | **Restringere `open_tol`** *dopo* l'A/B | rende la metrica selettiva e abbassa il riferimento banale | esperimento successivo |

---

## Limiti dichiarati

Sono limiti di **disegno**, non di esecuzione: dichiararli è parte del risultato
(Henderson et al. 2018).

| # | limite | conseguenza | cosa servirebbe |
|---|---|---|---|
| 1 | **Un solo seed di addestramento** | gli intervalli descrivono la variabilità fra **episodi**, non fra **seed**: le conclusioni valgono per *questa* policy, non per il metodo | ≥5 seed e aggregazione fra run |
| 2 | **Ablazione del controllore dispiegato** | misura la dipendenza, non la necessità | un ri-addestramento per variante |
| 3 | **La metrica è poco selettiva** | il banale ottiene già l'80.5%: il `true_success` misura in gran parte la geometria del compito | tolleranza più stretta o goal più lontani dal cap |
| 4 | **Numerosità** | con 200 episodi differenze sotto ~5 punti non sono risolvibili; le fasce (n=50) non escludono dipendenze fino a ~11 punti | più episodi, o meno fasce |
| 5 | **T4 non separa policy e override** | la torsione del polso in RETREAT è imposta dall'ambiente | registrare l'azione pre-override |
| 6 | **T1/T2 sulla porta base** | descrivono il regime nominale, non tutti i regimi campionati | test fisici sotto randomizzazione |

---

## Note di correzione alla suite

Sei correzioni, tutte attive e validate in questa sessione. Le prime quattro dalle sessioni
precedenti, le ultime due emerse **proprio dall'aumento di numerosità**.

1. **Errore con segno.** L'esito «door regress» era assegnato su un valore assoluto e
   marchiava come «porta che si richiude» quello che era overshoot.
2. **Campioni degeneri.** Il test di Welch su serie a varianza nulla emetteva un
   `RuntimeWarning` di precisione: ora gestito esplicitamente.
3. **T2 contaminato dal contatto col robot.** Reset seedato, rilevamento dei contatti,
   fino a 8 tentativi, esito `NON VALIDO` per le misure contaminate.
4. **Istogrammi su supporto discreto.** Il bootstrap mostrava un «pettine» di bin vuoti per
   costruzione: ora i bin sono allineati al reticolo dei valori.
5. **T6 contava step invece di episodi.** Un episodio in stallo resta centinaia di step fuori
   tolleranza: con 30 episodi il conteggio dava il 91% di scostamenti «oltre», con 100 il 12%.
   La suite riporta ora la statistica **per episodio** come primaria (9/100 oltre, 3 sotto),
   più un **indice di dominanza** (qui: i 3 episodi peggiori pesano il 91% degli step).
6. **T7 usava il denominatore sbagliato.** Contava come «fermo sulla maniglia» anche episodi
   che il RETREAT non l'avevano raggiunto: dava 1/100, il valore corretto è **0/99**. Ora
   concorda con `evaluate` (0/196).

Nessuna correzione cambia le conclusioni: le rendono più nette e **stabili al variare della
numerosità**.

---

## Bibliografia

**Metodologia di valutazione**

1. Agarwal et al. (2021) *Deep RL at the Edge of the Statistical Precipice.* NeurIPS 34 (Outstanding Paper) — IQM, bootstrap stratificato.
2. Colas et al. (2018) *How Many Random Seeds?* arXiv:1806.08295.
3. Colas et al. (2019) *A Hitchhiker's Guide to Statistical Comparisons of RL Algorithms.* arXiv:1904.06979.
4. Henderson et al. (2018) *Deep RL that Matters.* AAAI — varianza fra seed, riproducibilità, controllo banale.
5. Chan et al. (2020) *Measuring the Reliability of RL Algorithms.* ICLR — dispersione e rischio.
6. Rockafellar & Uryasev (2000) *Optimization of Conditional Value-at-Risk.* Journal of Risk 2(3) — definizione del CVaR.
7. Patterson et al. (2024) *Empirical Design in RL.* JMLR 25(318) — disegno appaiato.
8. Wilson (1927); Brown, Cai & DasGupta (2001) — intervallo per proporzioni.
9. Newcombe (1998) — intervallo per differenze di proporzioni.
10. Holm (1979) *A Simple Sequentially Rejective Multiple Test Procedure.* Scand. J. Statist. 6(2).

**Contributi v2 dell'apertura (ablazionati / diagnosticati)**

11. Ng, Russell & Harada (1999) *Policy Invariance Under Reward Transformations.*
12. Devlin & Kudenko (2012) *Dynamic Potential-Based Reward Shaping.*
13. Sutton, Precup & Singh (1999) *Between MDPs and Semi-MDPs.*
14. Konidaris & Barto (2009) *Skill Chaining.*
15. Tobin et al. (2017); Zhao et al. (2020); Mehta et al. (2020) — randomizzazione fisica.
16. ten Pas et al. (2017) *Grasp Pose Detection in Point Clouds.*
17. ManipForce (2015) — soglie di presa adattive al contatto.
18. Handa et al. (2020) *DexPilot.*
