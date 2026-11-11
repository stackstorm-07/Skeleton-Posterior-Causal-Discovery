"""Small statistics helpers for the M3 comparisons (paired differences, block bootstrap)."""
from __future__ import annotations

from typing import Dict

import numpy as np
from scipy import stats


def default_block_length(n: int) -> int:
    """Common rule of thumb for the moving-block bootstrap: l ~ n^(1/3) (about 8 for n = 589), at least 2."""
    return max(2, int(round(n ** (1.0 / 3.0))))


def block_bootstrap_indices(n: int, block_len: int, rng: np.random.Generator) -> np.ndarray:
    """Row indices of one moving-block bootstrap resample of a series with ``n`` rows.

    Overlapping blocks of ``block_len`` consecutive rows (start uniform in 0..n-block_len) are drawn with replacement
    and concatenated; the result is cut to exactly ``n`` rows.  Inside a block the time order (and so the
    autocorrelation) is preserved, which an i.i.d. row bootstrap would destroy."""
    block_len = int(min(max(block_len, 1), n))
    n_blocks = -(-n // block_len)
    starts = rng.integers(0, n - block_len + 1, size=n_blocks)
    idx = (starts[:, None] + np.arange(block_len)[None, :]).ravel()[:n]
    return idx


def paired_summary(a, b, higher_is_better: bool = True, alpha: float = 0.05) -> Dict[str, float]:
    """Paired comparison of two methods over the same units (graphs, bootstrap resamples, seeds).

    d_i = a_i - b_i.  Returns the mean and SD of d, a t-based CI of the mean, the percentile interval of the d_i themselves
    (for bootstrap resamples this IS the bootstrap interval of the difference), win/loss/tie counts of ``a`` against
    ``b`` (``higher_is_better`` says which direction is a win), a paired t-test p-value and an exact/approximate
    Wilcoxon signed-rank p-value (both two-sided).  p = 1 when all differences are zero."""
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    d = (a - b)[ok]
    n = len(d)
    out: Dict[str, float] = dict(n=n, mean_a=float(a[ok].mean()) if n else float("nan"), mean_b=float(b[ok].mean()) if n else float("nan"))
    if n == 0:
        return out
    sign = 1.0 if higher_is_better else -1.0
    out.update(mean_diff=float(d.mean()), sd_diff=float(d.std(ddof=1)) if n > 1 else float("nan"),
               wins=int((sign * d > 0).sum()), losses=int((sign * d < 0).sum()), ties=int((d == 0).sum()))
    if n > 1:
        se = d.std(ddof=1) / np.sqrt(n)
        h = stats.t.ppf(1 - alpha / 2, n - 1) * se
        out.update(ci_mean_lo=float(d.mean() - h), ci_mean_hi=float(d.mean() + h))
        out.update(ci_pct_lo=float(np.percentile(d, 100 * alpha / 2)), ci_pct_hi=float(np.percentile(d, 100 * (1 - alpha / 2))))
        out["p_ttest"] = float(stats.ttest_rel(a[ok], b[ok]).pvalue) if d.std(ddof=1) > 0 else 1.0
        out["p_wilcoxon"] = float(stats.wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
    return out


# ---------------------------------------------------------------------------------------------------------
# M3 aggregation helpers
# ---------------------------------------------------------------------------------------------------------
def two_way_anova(y, a, b) -> list:
    """Two-way ANOVA with interaction (sum-to-zero coding, Type III sums of squares; identical to Type I/II for balanced data).

    ``y`` values, ``a`` and ``b`` factor labels of equal length.  Returns one dict per effect (a, b, a:b) with ss, df, F, p and the
    partial eta^2, plus the residual.  Observations are treated as independent (a graph that appears under both methods is
    not modelled as a repeated measure; the paired tests in the paired-test table do that)."""
    y = np.asarray(y, dtype=float)
    a = np.asarray(a); b = np.asarray(b)
    ok = np.isfinite(y)
    y, a, b = y[ok], a[ok], b[ok]
    la, lb = sorted(set(a.tolist()), key=str), sorted(set(b.tolist()), key=str)

    def coding(x, levels):
        k = len(levels)
        M = np.zeros((len(x), k - 1))
        for j, lv in enumerate(levels[:-1]):
            M[:, j] = (x == lv).astype(float) - (x == levels[-1]).astype(float)
        return M

    A, Bm = coding(a, la), coding(b, lb)
    AB = np.column_stack([A[:, i] * Bm[:, j] for i in range(A.shape[1]) for j in range(Bm.shape[1])]) if A.shape[1] and Bm.shape[1] else np.zeros((len(y), 0))
    one = np.ones((len(y), 1))

    def rss(*blocks):
        X = np.column_stack([one, *blocks]) if blocks else one
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        r = y - X @ beta
        return float(r @ r), int(np.linalg.matrix_rank(X))

    full, rk = rss(A, Bm, AB)
    df_res = len(y) - rk
    ms_res = full / df_res if df_res > 0 else float("nan")
    out = []
    for name, others, df in (("a", (Bm, AB), A.shape[1]), ("b", (A, AB), Bm.shape[1]), ("a:b", (A, Bm), AB.shape[1])):
        if df == 0:
            continue
        ss = rss(*others)[0] - full
        F = (ss / df) / ms_res if ms_res and ms_res > 0 else float("nan")
        p = float(stats.f.sf(F, df, df_res)) if np.isfinite(F) else float("nan")
        out.append(dict(effect=name, ss=ss, df=df, F=F, p=p, partial_eta2=ss / (ss + full) if ss + full > 0 else float("nan")))
    out.append(dict(effect="residual", ss=full, df=df_res, F=float("nan"), p=float("nan"), partial_eta2=float("nan")))
    return out


def bootstrap_mean_ci(x, B: int = 2000, alpha: float = 0.05, seed: int = 0):
    """Percentile bootstrap CI of the mean of ``x`` (resampling units).  Returns (mean, lo, hi)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), size=(B, len(x)))].mean(axis=1)
    return float(x.mean()), float(np.percentile(m, 100 * alpha / 2)), float(np.percentile(m, 100 * (1 - alpha / 2)))


def bootstrap_paired_ci(a, b, B: int = 2000, alpha: float = 0.05, seed: int = 0):
    """Percentile bootstrap CI of mean(a - b) over paired units (e.g. RCA cases).  Returns (mean_diff, lo, hi, p_two_sided)
    where p is the share of resampled differences on the other side of 0, doubled (a bootstrap p, not an exact test)."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return (float("nan"),) * 4
    rng = np.random.default_rng(seed)
    m = d[rng.integers(0, len(d), size=(B, len(d)))].mean(axis=1)
    p = 2 * min(float((m <= 0).mean()), float((m >= 0).mean()))
    return float(d.mean()), float(np.percentile(m, 100 * alpha / 2)), float(np.percentile(m, 100 * (1 - alpha / 2))), float(min(p, 1.0))


def welch_diff(x, y, alpha: float = 0.05) -> Dict[str, float]:
    """Welch comparison of two independent samples (mean(x) - mean(y)): difference, t CI, p.  Used to compare an effect measured on
    PetShop resamples with the same effect measured on synthetic graphs."""
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    x, y = x[np.isfinite(x)], y[np.isfinite(y)]
    if len(x) < 2 or len(y) < 2:
        return dict(n_x=len(x), n_y=len(y))
    vx, vy = x.var(ddof=1) / len(x), y.var(ddof=1) / len(y)
    se = np.sqrt(vx + vy)
    df = (vx + vy) ** 2 / (vx ** 2 / (len(x) - 1) + vy ** 2 / (len(y) - 1)) if se > 0 else len(x) + len(y) - 2
    diff = float(x.mean() - y.mean())
    h = stats.t.ppf(1 - alpha / 2, df) * se
    return dict(n_x=len(x), n_y=len(y), mean_x=float(x.mean()), mean_y=float(y.mean()), mean_diff=diff, ci_lo=float(diff - h),
                ci_hi=float(diff + h), p=float(stats.ttest_ind(x, y, equal_var=False).pvalue) if se > 0 else 1.0)


def holm(pvals):
    """Holm-Bonferroni adjusted p-values (same order as the input; NaN stays NaN)."""
    p = np.asarray(pvals, dtype=float)
    out = np.full(len(p), np.nan)
    idx = [i for i in np.argsort(np.where(np.isfinite(p), p, np.inf)) if np.isfinite(p[i])]
    m, run = len(idx), 0.0
    for rank, i in enumerate(idx):
        run = max(run, min(1.0, (m - rank) * p[i]))
        out[i] = run
    return out
