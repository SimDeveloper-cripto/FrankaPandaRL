# La reward machine unificata

Una sola reward machine risolve **apertura v2 curr 1** e **chiusura v2 curr 1**.

Principio: *la chiusura è l'apertura con bersaglio zero*. Fra i due compiti **non
cambia nessun peso e nessuna soglia**: restano diversi **cinque** parametri su
dieci — i quattro della specifica più la consegna della leva (§4).

| | apertura | chiusura |
|:---|---:|---:|
| **true success** (294 episodi distinti) | **1.000** | **0.997** |
| baseline: progetto separato | 0.830 | 1.000 |
| **clean success** | **1.000** | **0.997** |
| distanza finale dalla maniglia | **0.094 m** | **0.067 m** |

---

## 1. La macchina

![Le cinque fasi e le transizioni](figure/macchina_fasi.png)

`A` significa **porta al bersaglio**: `|θ − θ*| ≤ tol`. È **bilaterale** — vale
uguale se la porta è corta o se ha superato il bersaglio — ed è la stessa
condizione nei due compiti. La macchina la usa in cinque punti.

| transizione | condizione |
|:---|:---|
| REACH → MOVE | presa confermata per **5** frame consecutivi |
| MOVE → REACH | presa persa, con isteresi, per **3** frame |
| MOVE → HOLD | **A ∧ gripper ≥ soglia ∧ dita chiuse** |
| REACH/MOVE → RELEASE | **A per T_hold frame**, comunque sia messa la mano |
| HOLD → RELEASE | timer ≥ T_hold, **oppure** stallo a **20** frame cumulativi fuori tolleranza |
| RELEASE → FINE | ≥ **30** passi ∧ A ∧ \|leva\| ≤ 0.15, **oppure** tetto duro a **120** passi |

Tre proprietà non ovvie, tutte necessarie:

- **il gate è bilaterale**, quindi la sovra-apertura non passa;
- **la guardia di stallo conta in modo cumulativo**: azzerandola a ogni frame
  dentro tolleranza, una porta che oscilla terrebbe a zero sia il timer sia la
  guardia e HOLD non finirebbe mai;
- **la porta è il compito**: se `A` regge per T_hold frame senza passare da
  HOLD si va comunque in RELEASE. Senza questa riga esistono stati da cui FINE
  è irraggiungibile.

---

## 2. I 17 termini

| # | termine | che cosa fa | fasi | peso |
|---:|:---|:---|:---|:---|
| 1 | `time` | costo del tempo | sempre | 0.1 |
| 2 | `smooth` | movimento fluido | sempre | 0.1 · 0.005 |
| 3 | `phi` | shaping potenziale (§3) | sempre | taglio ±10 |
| 4 | `approach` | avvicina la mano alla maniglia | REACH, MOVE | 5 · 3 · 15 |
| 5 | `approach_geom` | non arrivare da sotto né dall'alto | REACH | 3.0 · 1.5 |
| 6 | `wrist` | orienta il polso, solo da vicino | REACH, HOLD | 1.5 · 0.5 · 2.0 |
| 7 | `grip` | solo costi: non chiudere presto, non mollare | REACH, MOVE | 1.0 · 2.0 |
| 8 | `progress` | **il motore del compito** (§3) | MOVE | budget 700 |
| 9 | `contact` | contatto vero *mentre* la porta si muove | MOVE | 0.5 |
| 10 | `target` | precisione sul bersaglio, punisce chi lo supera | HOLD, RELEASE | 1.0 · 20.0 |
| 11 | `damp` | ferma la porta sul bersaglio | HOLD | 25.0 |
| 12 | `still` | tieni fermo il braccio | HOLD, RELEASE | 1 · 2 · 1 · 20 |
| 13 | `hold_grip` | salute della presa | HOLD | 1 · 5 · 2 · 10 · 3 |
| 14 | `release` | apri la mano al momento giusto | RELEASE | 2.0 · 1.0 · 1.0 |
| 15 | `retreat` | allontanati dritto dalla porta | RELEASE | 3 · 2 · 3 |
| 16 | `latch_home` | riporta la maniglia a riposo | RELEASE | 1.0 |
| 17 | `success` | **+10** alla prima uscita da REACH · **+600** a compito finito | transizione, terminale | 10 · 600 |

La somma di ogni passo è tagliata a **±100**, tranne al passo terminale:
altrimenti i 600 arriverebbero alla politica come 100. `success` si paga **una
volta sola** per episodio.

---

## 3. Il budget di `progress` e lo shaping

Percorrere tutta la corsa paga **700** in entrambi i compiti. Il punto delicato
è **come** si ripartisce fra i due tratti — girare la maniglia e muovere la
porta — perché è il numero che da solo decide quale compito si risolve.

Il catenaccio blocca la porta in **entrambi** i versi: a maniglia ferma si
arresta a 0.182 rad chiudendo e a 0.015 aprendo. Girare la maniglia è quindi
lavoro utile in tutti e due i casi.

| ripartizione | gradiente sulla porta | budget maniglia | chiusura | apertura |
|:---|---:|---:|:---|:---|
| un budget solo | 458 /rad | 563 | 1.00 | 0.21, in calo |
| maniglia esclusa | 2333 /rad | 0 | **0.05** | 0.37 |
| **`quota_leva = 0.30`** | **1257–1311 /rad** | **210** | **0.997** | **1.000** |

Sommare i due tratti in una sola escursione li mette **in concorrenza**: la
maniglia ha una corsa di 1.23 rad contro i ~0.37 della porta, quindi si prende
l'80 % del budget. Alla chiusura va bene; all'apertura, dove la porta *è* il
compito, resta troppo poco segnale. Togliendo la maniglia l'apertura riparte ma
la chiusura si pianta contro il catenaccio — misurato, 18 valutazioni su 20
ferme a errore +0.1755, sempre identico. Ogni tratto riceve quindi **un budget
suo**: la maniglia è un mezzo, la porta è il fine.

**Lo shaping** è potenziale [Ng, Russell & Harada 1999], dipendente dalla fase
[Devlin & Kudenko 2012]: `F = γ·Φ(s′) − Φ(s)` con γ = 0.95 e taglio ±10. Φ vale
zero in REACH e cresce a gradini: ogni fase superata entra al suo massimo, così
la scala non scende cambiando fase. Il premio per stringere la maniglia vive
**dentro Φ**, non nella ricompensa: negli originali era pagato a ogni passo e
restare sulla maniglia senza fare nulla rendeva quasi quanto risolvere il
compito. Dentro Φ conserva lo stesso gradiente ma a comando costante vale
`−(1−γ)Φ < 0`, quindi non è più mungibile.

---

## 4. Che cosa distingue i due compiti

I dieci parametri di `TaskSpec`. **Nessuno è un termine di ricompensa.** Di
questi, **cinque** distinguono i due compiti: i quattro che dicono *che cosa si
vuole ottenere*, più uno che dipende da come si comporta la porta.

| parametro | apertura | chiusura | |
|:---|:---|:---|:---|
| bersaglio θ\* (frazione della corsa) | 0.85–1.00, campionato a ogni reset | 0.015 fisso | **diverso** |
| partenza θ₀ | 0.0 | 0.70–1.00 | **diverso** |
| tolleranza | 0.05 rad | 0.03 rad | **diverso** |
| tempo di mantenimento | 1.0 s | 2.0 s | **diverso** |
| distanza di ritiro | 0.25 m | 0.25 m | uguale |
| alzata nel ritiro | 0.0 | 0.0 | uguale |
| ritiro lungo la normale orientata | sì | sì | uguale |
| riporto attivo della maniglia | sì | sì | uguale |
| controllore di fuga | sì | sì | uguale |
| consegna della leva alla molla | 0.15 rad | 0.40 rad | **diverso** |

I quattro parametri di geometria e di controllo sono diventati identici: dare
alla chiusura gli stessi controllori dell'apertura è ciò che ha portato la
chiusura da 0.983 a **0.997** e il suo ritiro da 0.654 a **0.860** (§9 e §10).

**La consegna della leva** è la soglia sotto la quale il riporto attivo lascia e
subentra la fuga. Non è cosmetica: finché il riporto è attivo la mano accompagna
l'arco della leva e **resta a 2–4 cm dalla maniglia**, e con la consegna alla
soglia di uscita il riporto occupa l'**85 %** della fase. Il valore differisce
perché differisce la porta: chiusa e agganciata al catenaccio, la molla riporta
la leva da sola — senza riporto finisce a 0.075, sotto la soglia; aperta, la
molla non ce la fa e la leva resta a 0.137–0.436 mentre la porta scivola via dal
bersaglio (§10).

**Il verso della maniglia** è anch'esso comune ai due compiti, perché è geometria
della porta: la maniglia si **abbassa**, come nel mondo reale. Sul modello MuJoCo
il giunto `Door_latch_joint` porta la maniglia a z = 1.000 con leva +1.5 e a
z = 1.150 con leva −1.5, contro z = 1.075 a riposo. Il tratto di leva conta in
`progress` **solo nel verso positivo**: senza questa specifica alzare e abbassare
valgono identico e il segno lo decide l'inizializzazione della rete — misurato,
un addestramento è finito a −1.506, cioè con la maniglia alzata, in 40 episodi su
40. Con la specifica, entrambi i modelli la abbassano in 40 su 40.

**SAC è invariato** rispetto agli originali: rete (512, 512), lr 3·10⁻⁴, γ 0.95,
buffer 10⁶, batch 256, τ 0.005, 8 ambienti, orizzonte 600, 30 Hz. L'unico valore
per compito è `target_entropy` (+1.0 apertura, −3.0 chiusura): appartiene
all'ottimizzatore, non alla reward machine.

I controllori — blocco del braccio in HOLD, morsa sulla presa, riporto della
maniglia, fuga — **non sono ricompensa**: sostituiscono l'azione prima che
raggiunga il simulatore, e i termini che dipendono da un'azione non più libera
sono **mascherati**, in un punto solo del codice.

---

## 5. Come si misura il successo

Tre livelli, gli stessi della tesi (§6.3.4):

| livello | definizione |
|:---|:---|
| **permissivo** | la politica ha raggiunto la fase di mantenimento |
| **true** | a fine episodio la porta è al bersaglio **e** la maniglia è tornata a riposo |
| **clean** | in più, la mano si è allontanata di **≥ 6 cm** da dove ha lasciato la maniglia, e l'episodio è terminato **per condizione soddisfatta**, non per tetto duro |

`true success` è la metrica di confronto, perché è quella con cui si misura la
baseline. Gli intervalli sono di **Wilson al 95 %**.

---

## 6. Apertura — addestramento

![Curva di addestramento dell'apertura](figure/apertura_addestramento.png)

| | valore |
|:---|---:|
| budget | 1 500 000 passi |
| success rate ≥ 0.95 | a **446 k** passi |
| success rate ≥ 0.99 | a **456 k** passi |
| success rate finale | **1.000** |
| ritorno medio finale | +1049 |
| lunghezza media finale | 95.2 passi |

La politica supera la baseline separata (0.830) a poco più di **un quarto** del
budget, e da 0.6 M passi in poi resta piatta a 1.000.

---

## 7. Apertura — valutazione, 200 episodi per seme

![Valutazione dell'apertura per seme](figure/apertura_valutazione.png)

| seme | permissivo | **true** | **clean** | passi RELEASE | maniglia |
|---:|---:|---:|---:|---:|---:|
| 42 | 1.000 | **1.000** | **1.000** | 32.6 | 0.094 m |
| 101 | 1.000 | **1.000** | **1.000** | 32.4 | 0.094 m |
| 7 | 1.000 | **1.000** | **1.000** | 32.3 | 0.094 m |

I tre semi danno **lo stesso identico valore**: il risultato non dipende da quali
porte vengono provate.

**Stima complessiva.** I semi 7, 42 e 101 su 200 episodi coprono gli episodi
7–206, 42–241 e 101–300: si sovrappongono, quindi le 600 esecuzioni
corrispondono a **294 episodi distinti**. La stima va fatta su questi.

| | true success | IC 95 % (Wilson) |
|:---|---:|:---|
| **apertura, macchina unificata** | **1.000** (294/294) | **[0.987, 1.000]** |
| apertura, progetto separato | 0.830 (166/200) | [0.772, 0.876] |
| riferimento banale (apre sempre a fondo corsa) | 0.805 | — |

**Gli intervalli non si toccano**: il miglioramento non è rumore di
campionamento. Ma il numero che conta di più è il **segno dell'errore**: la
baseline sbaglia sistematicamente in eccesso (+0.0243) perché apre sempre fino
al fine corsa — è quasi la politica banale. Qui l'errore medio è **−0.0193**,
distribuito sui due lati: la porta si **ferma al bersaglio**. Si batte il
riferimento banale per la ragione giusta.

**Nessun fallimento su 294.** I due che c'erano — la porta ferma corta di
~0.18 rad — sono spariti alzando il tetto del riporto della leva (§10).

---

## 8. Apertura — un episodio, transizione per transizione

Episodio riuscito: 95 passi, ritorno **+1067.9**, errore finale −0.0427,
leva +0.097.

| fase | passi | R | R/passo | termini principali |
|:---|---:|---:|---:|:---|
| REACH | 17 | −36.0 | −2.12 | `approach` −28.6 · `wrist` −3.9 · `grip` −2.0 |
| MOVE | 16 | **+525.6** | +32.85 | `progress` +517.8 · `success` +10.0 · `contact` +3.8 |
| HOLD | 30 | +11.6 | +0.39 | `damp` −34.1 · `hold_grip` +30.0 · `target` +26.2 · `still` +25.4 |
| RELEASE | 31 | −32.6 | −1.05 | `latch_home` −24.7 · `target` +21.4 · `phi` −18.5 |
| FINE | 1 | **+599.2** | +599.19 | `success` +600.0 |

La struttura si legge in una riga: REACH costa poco e dura poco, MOVE incassa il
budget di `progress`, HOLD mantiene, RELEASE paga il ritiro e riporta la
maniglia, FINE riscuote i 600.

![I 17 termini sull'episodio](figure/apertura_termini.png)

| termine | somma | termine | somma | termine | somma |
|:---|---:|:---|---:|:---|---:|
| `success` | +610.0 | `phi` | −30.6 | `smooth` | −3.5 |
| `progress` | +517.8 | `hold_grip` | +30.0 | `retreat` | −3.4 |
| `target` | +47.6 | `still` | +25.4 | `release` | −3.4 |
| `damp` | −34.1 | `latch_home` | −24.7 | `grip` | −2.0 |
| `approach` | −32.5 | `wrist` | −22.5 | `approach_geom` | −0.5 |
| | | `time` | −9.5 | `contact` | +3.8 |

Due termini valgono il 92 % del bilancio positivo — `success` e `progress` — e
nessun termine negativo supera i 35 punti: la macchina premia il compito e usa
il resto per correggere la forma del movimento, non per competere con
l'obiettivo.

---

## 9. Chiusura — addestramento e valutazione

![Curva di addestramento della chiusura](figure/chiusura_addestramento.png)

| | senza controllori | **con i controllori** |
|:---|---:|---:|
| success rate ≥ 0.95 | 635 k passi | **269 k passi** |
| success rate ≥ 0.99 | 645 k passi | **330 k passi** |
| success rate finale | 1.000 | 1.000 |
| ritorno medio finale | +885 | **+1066** |
| lunghezza media finale | 141.2 passi | **121.9 passi** |

Con i controllori la chiusura impara in **meno della metà** dei passi e in un
episodio più corto.

![Valutazione della chiusura per seme](figure/chiusura_valutazione.png)

| seme | permissivo | **true** | **clean** | \|errore\| medio | passi | ritorno | ritiro |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.995 | **0.995** | **0.995** | 0.0074 | 124.30 | +1032.7 | 0.138 m |
| 101 | 0.995 | **0.995** | **0.995** | 0.0074 | 124.22 | +1023.0 | 0.137 m |
| 7 | 0.995 | **0.995** | **0.995** | 0.0075 | 124.21 | +1035.4 | 0.138 m |

I tre semi danno **lo stesso identico valore**. Rispetto alla versione senza
controllori (0.985 / 0.980 / 0.990 di true e 0.975 / 0.960 / 0.975 di clean),
**true e clean salgono su tutti e tre i semi**: la regola di adozione era
esattamente questa.

| | true success | IC 95 % (Wilson) |
|:---|---:|:---|
| **chiusura, macchina unificata** | **0.997** (293/294) | **[0.981, 0.999]** |
| chiusura, versione precedente | 0.983 (289/294) | [0.961, 0.993] |
| chiusura, progetto separato | 1.000 (100/100) | [0.963, 1.000] |

**Resta un solo fallimento su 294**, e non è più la maniglia:

| episodio | passi | errore | leva | |
|---:|---:|---:|---:|:---|
| 163 | 600 | +0.3940 | −0.017 | la porta non arriva al bersaglio; la maniglia è a posto |

Prima i fallimenti erano cinque e **quattro riguardavano la leva** — porta chiusa
alla perfezione, maniglia ancora girata (+1.674, +1.284, +1.572, +0.724). Con il
riporto attivo della maniglia sono spariti tutti.

Lo stesso episodio letto dal lato della ricompensa (121 passi, ritorno +1163.5):

| fase | passi | R | R/passo | termini principali |
|:---|---:|---:|---:|:---|
| REACH | 14 | −31.7 | −2.26 | `approach` −22.7 · `grip` −5.4 · `wrist` −2.1 |
| MOVE | 16 | **+522.3** | +32.64 | `progress` +512.5 · `success` +10.0 · `contact` +4.7 |
| HOLD | 60 | +70.9 | +1.18 | `hold_grip` +60.0 · `still` +54.8 · `target` +47.6 |
| RELEASE | 30 | **+2.8** | +0.09 | `target` +25.4 · `phi` −17.6 · `latch_home` −15.4 · `release` +11.5 |
| FINE | 1 | **+599.2** | +599.24 | `success` +600.0 |

**La fase RELEASE passa da −111.1 a +2.8**, e il termine `retreat`, che prima
costava −153.5 per episodio, ora vale **+2.9**. Il ritiro non è più un movimento
che la macchina paga: è un movimento che la macchina premia. È la firma del fatto
che la direzione di fuga adesso punta dove il braccio va davvero.

---

## 10. Il ritiro del braccio

La grandezza che conta è **quanto la pinza finisce lontana dalla maniglia**.
Accanto si misurano la rotazione del polso durante la fase e la sua durata.

![Distanza finale pinza-maniglia](figure/distanza_maniglia.png)

**Il vincolo è il riporto della leva.** Finché è attivo la mano **accompagna
l'arco della maniglia** e non può andarsene: misurato su un episodio tipico, la
distanza resta fra 0.023 e 0.043 m per 24 passi su 30, e alla fuga ne restano
quattro. Il riporto occupava l'**85 %** della fase nella chiusura e l'**88 %**
nell'apertura.

Due parametri lo liberano, e **nessuno dei due allunga l'episodio**:

| | che cosa fa | riporto, dopo |
|:---|:---|---:|
| **tetto del riporto** 0.6 → **1.0**, per entrambi | la leva torna prima | 68 % apertura · 41 % chiusura |
| **consegna della leva** 0.15 → **0.40**, solo chiusura | il riporto lascia, la molla finisce | 41 % |

Il tetto era saturo per quasi tutta la corsa — `2.0·|leva|` supera 0.6 già a
0.3 rad — quindi la leva scendeva di appena 0.05 rad per passo. Alzandolo a 1.0
torna più in fretta e la fuga eredita i passi risparmiati.

**Risultato su 294 episodi distinti per compito:**

| | true | clean | maniglia | |
|:---|---:|---:|---:|:---|
| apertura, prima | 292/294 | 292/294 | 0.083 m | |
| **apertura, dopo** | **294/294** | **294/294** | **0.094 m** | +14 %, e due fallimenti in meno |
| chiusura, prima | 293/294 | 293/294 | 0.035 m | |
| **chiusura, dopo** | **293/294** | **293/294** | **0.067 m** | +92 %, stesso unico fallimento |

**Zero episodi peggiorati** in nessuno dei due compiti, e l'apertura arriva al
**100 %**: i due episodi che perdeva — la porta ferma corta di ~0.18 rad —
riescono entrambi ora che la leva torna prima e la fase non si allunga.

### Il confronto con i progetti separati

![Confronto del ritiro](figure/ritiro_confronto.png)

| | rotazione del polso | passi di RELEASE |
|:---|---:|---:|
| **apertura** unificata | **23.7** | **32.4** |
| apertura separata | 49.4 | 72.6 |
| **chiusura** unificata | 34.2 | **31.0** |
| chiusura separata | 26.1 | 33.4 |

L'apertura ritira con **metà** della rotazione di polso e in **meno della metà**
dei passi. La chiusura è alla pari sulla durata e paga un po' di rotazione in
più.

Una nota di onestà sulla forma del movimento: il rapporto fra spostamento netto
e lunghezza del percorso — che prima valeva 0.860 nella chiusura e 0.900
nell'apertura — scende a 0.523 e 0.735. Non è un peggioramento del gesto: la
traiettoria adesso ha **due tratti per costruzione**, l'arco della leva e poi la
fuga, e un rapporto che misura la rettilineità complessiva li conta come una
deviazione. La grandezza che descrive il risultato voluto — quanto la mano
finisce lontana — è quasi raddoppiata.

Il margine è stretto e misurato, e ci si è fermati sul bordo: nella chiusura,
consegnare la leva a 0.60 costa un episodio sul seme 7, a 0.80 ne costa tre su
200. Nell'apertura la consegna anticipata non si può fare affatto, perché lì la
molla non riporta la leva da sola — senza riporto finisce a 0.137–0.436 invece
che sotto 0.15, l'episodio arriva al tetto duro e nel frattempo la porta aperta,
che è **metastabile**, scivola via dal bersaglio: su 18 fallimenti provocati
allungando la fase, **16 sono la porta fuori tolleranza**, non il braccio.

### Lo sgombero dopo l'episodio

Dentro l'episodio il ritiro non può andare oltre: ogni tentativo di allungarlo o
accelerarlo costa episodi, ed è stato misurato quattro volte. Ma **quando
l'episodio è finito non c'è più niente da perdere**: successo, ricompensa e
metriche sono già stati calcolati e chiusi.

Da lì il braccio completa il gesto in tre tempi: **si alza** di pochi centimetri
per uscire dall'arco della maniglia, **si allontana** lungo la normale di fuga a
mano aperta, **si ferma**. Come un robot che libera lo spazio di lavoro a compito
concluso.

| | a fine episodio | dopo lo sgombero |
|:---|---:|---:|
| chiusura · distanza dalla maniglia | 0.076 m | **0.325 m** |
| chiusura · quota della pinza | 1.028 m | **1.101 m** (+0.072) |
| apertura · distanza dalla maniglia | 0.095 m | **0.324 m** |
| apertura · quota della pinza | 1.008 m | **1.067 m** (+0.060) |

L'alzata viene prima perché sfilarsi radenti significa passare accanto alla barra
della maniglia: pochi centimetri in su e il dito ne è fuori.

Lo sgombero gira **solo con `--play`**, ed è subordinato a
`render_mode == "human"`: in addestramento e in valutazione non viene mai
eseguito. Le distanze riportate in questo documento restano quelle
**dell'episodio**, non quelle dello sgombero. Verificato: 60 episodi per compito,
con e senza, **0 differenze** e ritorno medio identico all'ottavo decimale.

---

## 11. Il bilancio rispetto alla baseline

| | baseline separata | macchina unificata | |
|:---|---:|---:|:---|
| apertura, true success | 0.830 | **1.000** | **+0.170**, intervalli disgiunti |
| apertura, errore medio con segno | +0.0243 | **−0.0193** | non satura più il fine corsa |
| apertura, distanza dalla maniglia | — | **0.094 m** | +14 % rispetto a prima |
| apertura, passi di RELEASE | 72.6 | **32.4** | **−55 %** |
| apertura, rotazione del polso | 49.4 | **23.7** | **−52 %** |
| chiusura, true success | 1.000 | **0.997** | un episodio su 294, intervalli sovrapposti |
| chiusura, clean success | — | **0.997** | |
| chiusura, errore assoluto | — | 0.0074 rad | un quarto di quello dell'apertura |
| chiusura, distanza dalla maniglia | — | **0.067 m** | +92 % rispetto a prima |
| chiusura, passi per episodio | — | 124.2 | **−12 %** rispetto ai 141.1 di prima |
| codice | due progetti separati | **una macchina, 17 termini** | stessi pesi, cinque parametri diversi |

**L'apertura era il compito irrisolto della tesi** — 83.0 % — ed è risolta al
**100 %**, con intervalli di confidenza disgiunti e un movimento migliore sotto
ogni grandezza misurata.

**La chiusura era già risolta al 100 %** dal progetto dedicato, e la macchina
unificata la riproduce al **99.7 %**: un solo episodio su 294, e non è più un
fallimento sulla maniglia ma una porta che non arriva al bersaglio. Gli
intervalli — [0.981, 0.999] contro [0.963, 1.000] — si sovrappongono quasi per
intero: sui dati disponibili le due soluzioni non sono distinguibili.

Il punto della tesi, però, non è il pareggio: è che **una sola specifica, con gli
stessi 17 termini e gli stessi pesi, ottiene su entrambi i compiti quello che due
progetti separati ottenevano su uno solo ciascuno** — e sull'apertura fa
nettamente meglio.

---

## 12. Ablazione dei meccanismi

Stessa metodologia delle suite dei due progetti separati: gli override sono già
dentro l'ambiente e accesi da parametri; l'ablazione li **spegne in valutazione,
sulla stessa politica addestrata**, un fattore per volta [Patterson et al. 2024].
Si ablaziona il controllo, non i pesi: in valutazione la ricompensa non entra
nella decisione.

25 episodi, seme 42. Δ è la variazione di true success rispetto alla baseline.

| meccanismo spento | apertura | chiusura |
|:---|---:|---:|
| **riporto della leva** | **−0.920** | +0.000 |
| **morsa sulla presa** | **−0.600** | −0.040 |
| controllore di fuga | −0.120 | **−0.160** |
| normale orientata verso il robot | −0.120 | **−0.160** |
| tetto del riporto riportato a 0.6 | −0.040 | +0.000 |
| consegna della leva alla soglia di uscita | — | +0.000 |
| guardia di stallo in HOLD | — | +0.000 |
| **tutti gli override insieme** | **−1.000** | **−0.160** |

Si legge in tre righe:

- **l'apertura vive sugli override**: senza, non riesce **nemmeno un episodio su
  25**. Il riporto della leva da solo vale 0.92 di success rate;
- **la chiusura è molto più robusta**: senza alcun override perde 0.16, e senza
  il solo riporto della leva **non perde niente** — la molla il lavoro lo fa da
  sé, ed è la stessa asimmetria che spiega perché la consegna della leva è
  l'unico parametro di controllo diverso fra i due compiti (§4);
- **la guardia di stallo non interviene mai** su questi episodi: è una rete di
  sicurezza per gli stati degeneri, non un meccanismo di regime.

Nella chiusura, spegnere il riporto porterebbe la mano a 0.146 m invece di
0.067 — ma su 200 episodi costa 2 clean success, quindi è stato scartato.

I risultati completi sono in `unified_door/ablazione_close.json` e
`ablazione_open.json`.

---

## 13. Come riprodurre le misure

```bash
cd unified_door
export PROGETTI_ORIGINALI="$(cd .. && pwd)"

# valutazione a tre semi, 200 episodi ciascuno
python3 train_unified.py --task open  --eval --eval-seeds 42,101,7 --episodes 200
python3 train_unified.py --task close --eval --eval-seeds 42,101,7 --episodes 200

# episodi a schermo, con transizioni e bilancio dei termini
mjpython train_unified.py --task open  --play --slow 2
mjpython train_unified.py --task close --play

# più episodi di fila: con --episodes il play dichiara posa e fisica di ognuno
mjpython train_unified.py --task close --play --episodes 5

# verifica dell'implementazione contro questo documento
python3 tests/test_unified.py        # 240 controlli
```

I risultati completi sono in `risultati_apertura.txt`, `risultati_chiusura.txt`
e `risultati_chiusura_addestramento.txt`.
