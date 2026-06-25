#!/usr/bin/env python3
# scratch/test_close_task_v1/stats_utils.py
"""
stats_utils — Nucleo statistico della suite di test scientifica (task di chiusura v1).

Questo modulo NON dipende da MuJoCo / robosuite / SB3: contiene solo gli strumenti
statistici raccomandati dalla letteratura per valutare in modo rigoroso un agente di
Deep RL. Può quindi essere importato ed eseguito in isolamento (e i suoi self-test in
fondo al file girano senza simulatore).

Perché questi strumenti — riferimenti bibliografici
----------------------------------------------------
La valutazione di un agente di Deep RL con *point estimate* (media/mediana di pochi
episodi) è inaffidabile: ignora l'incertezza dovuta al numero finito di prove. La
comunità scientifica raccomanda di riportare **stime a intervallo** e metriche robuste:

  [A] Agarwal, Schwarzer, Castro, Courville, Bellemare (2021).
      "Deep Reinforcement Learning at the Edge of the Statistical Precipice".
      NeurIPS 34:29304-29320 (Outstanding Paper). Libreria: `rliable`.
      → Interquartile Mean (IQM), stratified bootstrap CI, performance profiles,
        probability of improvement, optimality gap.

  [B] Colas, Sigaud, Oudeyer (2018). "How Many Random Seeds? Statistical Power
      Analysis in Deep Reinforcement Learning Experiments". arXiv:1806.08295.
      → dimensionamento del campione, test t di Welch e bootstrap CI per confronti.

  [C] Colas, Sigaud, Oudeyer (2019). "A Hitchhiker's Guide to Statistical Comparisons
      of Reinforcement Learning Algorithms". arXiv:1904.06979.

  [D] Henderson, Islam, Bachman, Pineau, Precup, Meger (2018).
      "Deep Reinforcement Learning that Matters". AAAI 2018.
      → riproducibilità, riportare la varianza, evitare conclusioni da pochi run.

  [E] Chan, Fishman, Korattikara, Canny, Guadarrama (2020).
      "Measuring the Reliability of Reinforcement Learning Algorithms". ICLR 2020.
      Libreria: `rl-reliability-metrics`.
      → dispersione (IQR) e rischio (CVaR) come metriche di affidabilità.

  [F] Patterson, Neumann, White, White (2024). "Empirical Design in Reinforcement
      Learning". JMLR 25(318):1-63.
      → disegno sperimentale controllato, confronti appaiati (paired/blocked).

  [G] Brown (Wilson, 1927; Brown, Cai, DasGupta, 2001). Intervallo di Wilson per la
      proporzione binomiale: preferito a Wald per n piccolo e p vicino a 0/1
      (caso tipico del success rate ~100%).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Callable, Sequence

import numpy as np
from scipy import stats


# ─────────────────────────────────────────────────────────────────────────────
# Tipi di ritorno
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Interval:
    """Stima puntuale + intervallo di confidenza (lo, hi) al livello `conf`."""
    point: float
    lo: float
    hi: float
    conf: float = 0.95

    def as_dict(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:
        pct = int(round(self.conf * 100))
        return f"{self.point:.4f} [{self.lo:.4f}, {self.hi:.4f}] ({pct}% CI)"


@dataclass
class Comparison:
    """Risultato di un confronto fra due gruppi (es. baseline vs ablazione)."""
    name_a: str
    name_b: str
    diff: Interval                 # differenza (a - b) con bootstrap CI
    p_value: float                 # test appropriato (Fisher / Welch)
    test_name: str
    effect_size: float             # Cohen's d (continuo) o Cliff's delta / h (proporzioni)
    effect_label: str
    prob_improvement: float        # P(a > b) stile Agarwal 2021

    def as_dict(self) -> dict:
        d = asdict(self)
        d["diff"] = self.diff.as_dict()
        return d


# ─────────────────────────────────────────────────────────────────────────────
# Proporzioni (success rate) — intervallo di Wilson [G]
# ─────────────────────────────────────────────────────────────────────────────
def wilson_ci(successes: int, n: int, conf: float = 0.95) -> Interval:
    """
    Intervallo di confidenza di Wilson per una proporzione binomiale.

    Da preferire al classico Wald (p ± z·sqrt(p(1-p)/n)) quando p è vicino a 0 o 1
    o quando n è piccolo — esattamente il regime di un success rate ~100% su poche
    decine di episodi. Per 50/50 successi Wald darebbe [1.0, 1.0] (intervallo nullo,
    falsamente certo); Wilson restituisce un limite inferiore < 1 onesto. Rif. [G].
    """
    if n <= 0:
        return Interval(float("nan"), float("nan"), float("nan"), conf)
    z = stats.norm.ppf(1.0 - (1.0 - conf) / 2.0)
    p = successes / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return Interval(point=p, lo=max(0.0, center - half), hi=min(1.0, center + half), conf=conf)


def required_n_for_proportion(p_expected: float, half_width: float, conf: float = 0.95) -> int:
    """
    Numero di episodi necessario perché il CI (Wald) di una proporzione abbia
    semi-ampiezza `half_width`. Strumento di *power analysis* per pianificare quanti
    episodi servono — non per scuse a posteriori. Rif. [B].

    Esempio: per stimare un success rate ~0.95 con ±0.03 al 95% servono ~203 episodi.
    """
    z = stats.norm.ppf(1.0 - (1.0 - conf) / 2.0)
    p = min(max(p_expected, 1e-6), 1 - 1e-6)
    return int(np.ceil((z * z * p * (1 - p)) / (half_width * half_width)))


# ─────────────────────────────────────────────────────────────────────────────
# Metriche robuste su campioni continui — IQM, bootstrap CI [A], CVaR [E]
# ─────────────────────────────────────────────────────────────────────────────
def iqm(samples: Sequence[float]) -> float:
    """
    Interquartile Mean: media del 50% centrale dei dati (scarta il 25% più basso e
    il 25% più alto). Più robusta della media agli outlier e più efficiente della
    mediana. Metrica aggregata raccomandata in [A].
    """
    x = np.sort(np.asarray(samples, dtype=float))
    n = len(x)
    if n == 0:
        return float("nan")
    lo = int(np.floor(n * 0.25))
    hi = int(np.ceil(n * 0.75))
    core = x[lo:hi] if hi > lo else x
    return float(np.mean(core))


def cvar(samples: Sequence[float], alpha: float = 0.1, lower_tail: bool = True) -> float:
    """
    Conditional Value at Risk: media della coda peggiore (frazione `alpha`).
    Metrica di *rischio* in [E]: per la robotica cattura il caso peggiore
    (es. l'episodio in cui la porta resta più aperta), non solo il caso medio.
    `lower_tail=True` → media del 10% di valori più BASSI (peggio = valore basso,
    es. min_door_angle dove vicino a 0 è bene → usare lower_tail=False).
    """
    x = np.sort(np.asarray(samples, dtype=float))
    n = len(x)
    if n == 0:
        return float("nan")
    k = max(1, int(np.floor(n * alpha)))
    tail = x[:k] if lower_tail else x[-k:]
    return float(np.mean(tail))


def _statistic_fn(name: str) -> Callable[[np.ndarray], float]:
    return {
        "mean": lambda a: float(np.mean(a)),
        "median": lambda a: float(np.median(a)),
        "iqm": iqm,
    }[name]


def bootstrap_ci(
    samples: Sequence[float],
    statistic: str = "iqm",
    n_boot: int = 10_000,
    conf: float = 0.95,
    seed: int = 0,
) -> Interval:
    """
    Intervallo di confidenza via bootstrap percentile (ricampionamento con
    reinserimento). Non assume normalità — adatto a metriche asimmetriche come la
    lunghezza d'episodio o il min_door_angle. Rif. [A] (stratified bootstrap), [B].
    """
    x = np.asarray(samples, dtype=float)
    n = len(x)
    fn = _statistic_fn(statistic)
    if n == 0:
        return Interval(float("nan"), float("nan"), float("nan"), conf)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = np.array([fn(x[i]) for i in idx])
    a = (1.0 - conf) / 2.0
    lo, hi = np.percentile(boots, [100 * a, 100 * (1 - a)])
    return Interval(point=fn(x), lo=float(lo), hi=float(hi), conf=conf)


# ─────────────────────────────────────────────────────────────────────────────
# Confronti fra due gruppi (baseline vs ablazione) [A][B][C][F]
# ─────────────────────────────────────────────────────────────────────────────
def cohens_d(a: Sequence[float], b: Sequence[float]) -> float:
    """Effect size standardizzato per campioni continui (pooled std)."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    sp2 = ((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1)) / (na + nb - 2)
    sp = np.sqrt(sp2)
    return float((np.mean(a) - np.mean(b)) / sp) if sp > 0 else 0.0


def cliffs_delta(a: Sequence[float], b: Sequence[float]) -> float:
    """
    Cliff's delta: effect size non parametrico = P(a>b) - P(a<b) ∈ [-1, 1].
    Robusto e adatto a metriche non normali. |δ|<0.147 trascurabile, <0.33 piccolo,
    <0.474 medio, oltre grande (soglie di Romano et al. 2006).
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    gt = sum((a[:, None] > b[None, :]).sum(axis=1))
    lt = sum((a[:, None] < b[None, :]).sum(axis=1))
    return float((gt - lt) / (len(a) * len(b)))


def probability_of_improvement(a: Sequence[float], b: Sequence[float]) -> float:
    """
    P(a > b): probabilità che un campione di A superi uno di B (ties = 0.5).
    Metrica di confronto raccomandata in [A] — interpretabile senza assunzioni
    distribuzionali. Identica al concetto di "common-language effect size".
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    gt = (a[:, None] > b[None, :]).sum()
    eq = (a[:, None] == b[None, :]).sum()
    return float((gt + 0.5 * eq) / (len(a) * len(b)))


def compare_continuous(
    a: Sequence[float], b: Sequence[float],
    name_a: str = "A", name_b: str = "B",
    n_boot: int = 10_000, conf: float = 0.95, seed: int = 0,
) -> Comparison:
    """
    Confronto di due metriche continue (es. lunghezza d'episodio baseline vs ablazione):
    test t di Welch (varianze diseguali, [B]) + bootstrap CI della differenza delle medie
    + Cohen's d + probability of improvement [A].
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if len(a) >= 2 and len(b) >= 2:
        t = stats.ttest_ind(a, b, equal_var=False)
        p = float(t.pvalue)
    else:
        p = float("nan")
    rng = np.random.default_rng(seed)
    boots = np.array([
        np.mean(rng.choice(a, len(a), replace=True)) - np.mean(rng.choice(b, len(b), replace=True))
        for _ in range(n_boot)
    ])
    al = (1.0 - conf) / 2.0
    lo, hi = np.percentile(boots, [100 * al, 100 * (1 - al)])
    diff = Interval(point=float(np.mean(a) - np.mean(b)), lo=float(lo), hi=float(hi), conf=conf)
    return Comparison(
        name_a=name_a, name_b=name_b, diff=diff,
        p_value=p, test_name="Welch t-test",
        effect_size=cohens_d(a, b), effect_label="Cohen's d",
        prob_improvement=probability_of_improvement(a, b),
    )


def compare_proportions(
    succ_a: int, n_a: int, succ_b: int, n_b: int,
    name_a: str = "A", name_b: str = "B", conf: float = 0.95,
) -> Comparison:
    """
    Confronto di due success rate (es. baseline vs ablazione): test esatto di Fisher
    (robusto per n piccolo / proporzioni estreme) + Newcombe CI della differenza di
    proporzioni (da due intervalli di Wilson) + Cohen's h come effect size.
    """
    table = [[succ_a, n_a - succ_a], [succ_b, n_b - succ_b]]
    try:
        _, p = stats.fisher_exact(table)
        p = float(p)
    except Exception:
        p = float("nan")

    # Newcombe (1998): combina i due intervalli di Wilson
    wa, wb = wilson_ci(succ_a, n_a, conf), wilson_ci(succ_b, n_b, conf)
    pa, pb = wa.point, wb.point
    lo = (pa - pb) - np.sqrt((pa - wa.lo) ** 2 + (wb.hi - pb) ** 2)
    hi = (pa - pb) + np.sqrt((wa.hi - pa) ** 2 + (pb - wb.lo) ** 2)
    diff = Interval(point=pa - pb, lo=float(lo), hi=float(hi), conf=conf)

    # Cohen's h (effect size per proporzioni)
    phi = lambda x: 2 * np.arcsin(np.sqrt(min(max(x, 0.0), 1.0)))
    h = float(phi(pa) - phi(pb))
    return Comparison(
        name_a=name_a, name_b=name_b, diff=diff,
        p_value=p, test_name="Fisher exact",
        effect_size=h, effect_label="Cohen's h",
        prob_improvement=float("nan"),  # non definita per due singole proporzioni
    )


def holm_bonferroni(p_values: Sequence[float]) -> list[float]:
    """
    Correzione di Holm-Bonferroni per confronti multipli (es. molte ablazioni vs un
    baseline). Più potente di Bonferroni puro mantenendo il controllo del FWER.
    Restituisce i p-value aggiustati nell'ordine originale. Rif. [C].
    """
    p = np.asarray(p_values, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * p[idx]
        running = max(running, val)
        adj[idx] = min(1.0, running)
    return adj.tolist()


# ─────────────────────────────────────────────────────────────────────────────
# Self-test (gira senza simulatore): `python stats_utils.py`
# ─────────────────────────────────────────────────────────────────────────────
def _selftest() -> None:
    print("stats_utils self-test")
    print("-" * 60)

    # Wilson vs Wald per 50/50: Wald darebbe [1,1]; Wilson < 1.
    w = wilson_ci(50, 50)
    assert w.point == 1.0 and w.lo < 1.0, "Wilson deve dare lo<1 per 50/50"
    print(f"  Wilson 50/50 successi        : {w}")
    print(f"  Wilson 47/50 successi        : {wilson_ci(47, 50)}")

    n = required_n_for_proportion(0.95, 0.03)
    assert 150 < n < 260, n
    print(f"  Episodi per p~0.95 ±0.03     : {n}")

    rng = np.random.default_rng(1)
    x = rng.normal(120, 8, size=200)
    print(f"  IQM lunghezza (sim)          : {iqm(x):.2f}")
    print(f"  bootstrap CI IQM             : {bootstrap_ci(x, 'iqm')}")
    print(f"  CVaR 10% coda bassa          : {cvar(x, 0.1):.2f}")

    a = rng.normal(122, 8, size=120)
    b = rng.normal(128, 9, size=120)
    cmp_c = compare_continuous(a, b, "var", "base")
    assert cmp_c.diff.hi < 0, "media a<b deve dare diff CI negativo"
    print(f"  compare_continuous diff      : {cmp_c.diff}  p={cmp_c.p_value:.3g}  d={cmp_c.effect_size:.2f}")
    print(f"  P(improvement) a>b           : {cmp_c.prob_improvement:.3f}")

    cmp_p = compare_proportions(48, 50, 50, 50, "var", "base")
    print(f"  compare_proportions diff     : {cmp_p.diff}  Fisher p={cmp_p.p_value:.3g}  h={cmp_p.effect_size:.2f}")

    adj = holm_bonferroni([0.01, 0.04, 0.03, 0.005])
    assert all(0 <= v <= 1 for v in adj)
    print(f"  Holm-Bonferroni (4 test)     : {[round(v,4) for v in adj]}")

    print("-" * 60)
    print("OK — tutti i controlli passati.")


if __name__ == "__main__":
    _selftest()
