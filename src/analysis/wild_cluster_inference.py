"""Wild-cluster-bootstrap-t inference engine (Round 12, Task E/F).

Every v2 regression report in this repo must carry: coefficient, classical
SE, cluster SE (by CounDist), p from t(G-1), wild-cluster-bootstrap-t p
(Webb six-point weights, 9,999 replications, fixed seed), 95% CI, N, and
the number of clusters G. This module computes all of them from one OLS
design, with a fast exact wild bootstrap (boottest-style sufficient
statistics — no refitting the ~500k-row regression 9,999 times).

METHOD (Cameron, Gelbach & Miller 2008; Webb 2024 weights; MacKinnon &
Webb boottest implementation):
  * OLS:      y = X beta + e, X = [focus, X2] (focus = ideology regressor).
  * Classical SE:  sigma^2 (X'X)^{-1}.
  * Cluster SE:    CR1 sandwich with the G/(G-1) finite-sample factor,
    clustered by council district; p and CI use t(G-1).
  * Wild cluster bootstrap-t imposing the null:
      - restricted fit   y = X2 gamma + u  (focus excluded under H0)
      - x1~ = M_{X2} x1  (Frisch-Waugh residualized focus)
      - bootstrap DGP    y* = X2 gamma_hat + w . u,  w_g iid WEBB weights
        per cluster: {+-1, +-(5-sqrt5)/10, +-(5+sqrt5)/10} each w.p. 1/6.
      - beta*_1 = (sum_g w_g a_g) / B,  a_g = x1~_g' u_g, B = x1~'x1~.
      - e* = M_X y* = sum_h w_h m_h,  m_h = M_X (u . 1_h)  (precomputed
        once per spec: G solves using the cached (X'X)^{-1}).
      - CR1 variance of beta*_1 from d_g = x1~_g' e* = sum_h w_h D[g,h],
        D = diag(a) - P B_h where P[g,:] = x1~_g' X and B_h columns are
        the coefficients of the per-cluster solves.
      - t* = beta*_1 / SE*; p_wild = (#{|t*| >= |t_hat|} + 1) / (reps + 1).
    All per-replication work is G x G, so 9,999 reps run in milliseconds.

Seeds are fixed (default 42) so every verification re-derivation reproduces
the p-value bit-for-bit.
"""
import numpy as np
import pandas as pd
from scipy import stats as sps

WEBB_WEIGHTS = np.array([
    -1.0, -(5 + np.sqrt(5)) / 10.0, -(5 - np.sqrt(5)) / 10.0,
    (5 - np.sqrt(5)) / 10.0, (5 + np.sqrt(5)) / 10.0, 1.0,
])
DEFAULT_REPS = 9999
DEFAULT_SEED = 42

class WildClusterOLS:
    """One OLS design + full inference suite (classical/cluster/t(G-1)/wild)."""

    def __init__(self, y, X, clusters, focus_idx, const_idx=0):
        self.y = np.asarray(y, dtype=np.float64)
        Xa = np.asarray(X, dtype=np.float64)
        if Xa.ndim != 2 or self.y.shape[0] != Xa.shape[0]:
            raise ValueError('y/X shape mismatch')
        self.focus_idx = int(focus_idx)
        self.const_idx = int(const_idx)
        self.clusters = np.asarray(clusters)
        self.n = Xa.shape[0]
        self.k = Xa.shape[1]
        # ---- numerical stability: solve in standardized space, report in
        # the ORIGINAL units of the focus column. Column standardization
        # leaves t-statistics and wild-bootstrap p-values exactly invariant;
        # reported coef/SE/CI for the focus are rescaled back by s_focus.
        # The intercept column is left untouched (centering it would zero
        # it out of the column space).
        self.col_mean = Xa.mean(axis=0)
        self.col_std = np.ones(self.k)
        for j in range(self.k):
            if j == self.const_idx:
                self.col_mean[j] = 0.0
                continue
            s = Xa[:, j].std()
            if s > 0:
                self.col_std[j] = s
        self.X = (Xa - self.col_mean) / self.col_std
        self.s_focus = float(self.col_std[self.focus_idx])
        self.g_labels, self.g_idx = np.unique(self.clusters, return_inverse=True)
        self.G = int(self.g_labels.size)

    def _solve(self, A, b):
        if A.shape[0] == A.shape[1]:
            return np.linalg.solve(A, b)
        return np.linalg.lstsq(A, b, rcond=None)[0]

    def _cr1_factor(self):
        """Stata/statsmodels CR1 small-sample correction
        (n-1)/(n-k) * G/(G-1) — matches statsmodels cov_type='cluster'."""
        return ((self.n - 1.0) / max(self.n - self.k, 1)
                * self.G / max(self.G - 1, 1))

    def fit(self):
        X = self.X
        self.XtX_inv = np.linalg.pinv(X.T @ X)
        self.beta = self.XtX_inv @ (X.T @ self.y)
        self.resid = self.y - X @ self.beta
        rss = float(self.resid @ self.resid)
        sigma2 = rss / max(self.n - self.k, 1)
        self.vcov_classical = sigma2 * self.XtX_inv
        Xu = X * self.resid[:, None]
        meat = np.zeros((self.k, self.k))
        for g in range(self.G):
            Xug = Xu[self.g_idx == g].sum(axis=0)
            meat += np.outer(Xug, Xug)
        self.vcov_cluster = self._cr1_factor() * (self.XtX_inv @ meat @ self.XtX_inv)
        self.se_classical = np.sqrt(np.clip(np.diag(self.vcov_classical), 0, None))
        self.se_cluster = np.sqrt(np.clip(np.diag(self.vcov_cluster), 0, None))
        # focus quantities in ORIGINAL (unscaled) units; t/p invariant
        self.beta_focus = float(self.beta[self.focus_idx]) / self.s_focus
        self.se_classical_focus = float(self.se_classical[self.focus_idx]) / self.s_focus
        self.se_cluster_focus = float(self.se_cluster[self.focus_idx]) / self.s_focus
        self.t_stat = self.beta_focus / self.se_cluster_focus
        self.p_t = float(2.0 * sps.t.sf(abs(self.t_stat), self.G - 1))
        tcrit = float(sps.t.ppf(0.975, self.G - 1))
        self.ci_low = self.beta_focus - tcrit * self.se_cluster_focus
        self.ci_high = self.beta_focus + tcrit * self.se_cluster_focus
        ymean = float(self.y.mean())
        tss = float(((self.y - ymean) ** 2).sum())
        self.r2 = 1.0 - rss / max(tss, 1e-300)
        return self

    def wild_bootstrap(self, reps=DEFAULT_REPS, seed=DEFAULT_SEED):
        """Fast exact wild-cluster-bootstrap-t (see module docstring)."""
        X = self.X
        f = self.focus_idx
        others = np.array([i for i in range(self.k) if i != f])
        X2 = X[:, others]
        x1 = X[:, f]
        XtX2 = X2.T @ X2
        gamma = self._solve(XtX2, X2.T @ self.y)
        u = self.y - X2 @ gamma
        x1_t = x1 - X2 @ self._solve(XtX2, X2.T @ x1)
        B = float(x1_t @ x1_t)
        if B <= 0:
            raise ValueError('degenerate focus column after residualization')
        a = np.zeros(self.G)
        Xu_c = np.zeros((self.G, self.k))
        Px = np.zeros((self.G, self.k))
        for g in range(self.G):
            sel = np.flatnonzero(self.g_idx == g)
            x1g = x1_t[sel]
            ug = u[sel]
            a[g] = float(x1g @ ug)
            Xu_c[g] = X[sel].T @ ug
            Px[g] = x1g @ X[sel]
        Ccoef = self.XtX_inv @ Xu_c.T           # k x G
        D = np.diag(a) - Px @ Ccoef             # G x G
        cr1 = self._cr1_factor()
        rng = np.random.default_rng(seed)
        w_idx = rng.integers(0, WEBB_WEIGHTS.size, size=(reps, self.G))
        W = WEBB_WEIGHTS[w_idx]                 # reps x G
        num = W @ a / B                         # beta*_1 per rep
        d = W @ D.T                             # reps x G
        var = cr1 * (d ** 2).sum(axis=1) / (B * B)
        se_star = np.sqrt(var)
        t_star = num / se_star
        p_wild = float((np.abs(t_star) >= abs(self.t_stat)).sum() + 1) \
            / (reps + 1)
        # bootstrap-t percentile CI (Webb weights, cluster-robust scale)
        lo = float(np.quantile(t_star, 0.025)) * self.se_cluster_focus \
            + self.beta_focus
        hi = float(np.quantile(t_star, 0.975)) * self.se_cluster_focus \
            + self.beta_focus
        return dict(p_wild=p_wild, ci_wild_low=lo, ci_wild_high=hi,
                    reps=reps, seed=seed, weights='webb6')

    def report(self, reps=DEFAULT_REPS, seed=DEFAULT_SEED):
        r = self.wild_bootstrap(reps=reps, seed=seed)
        return {
            'coef': self.beta_focus,
            'se_classical': self.se_classical_focus,
            'se_cluster': self.se_cluster_focus,
            'p_t_G_minus_1': self.p_t,
            'p_wild_cluster_bootstrap_t': r['p_wild'],
            'ci95_low_t': self.ci_low,
            'ci95_high_t': self.ci_high,
            'ci95_low_wild': r['ci_wild_low'],
            'ci95_high_wild': r['ci_wild_high'],
            'n': int(self.n),
            'n_clusters_G': self.G,
            'r_squared': float(self.r2),
            'wild_bootstrap': {'weights': 'webb6', 'reps': reps,
                               'seed': seed},
        }


def fit_report(y, X, clusters, focus_col, reps=DEFAULT_REPS,
               seed=DEFAULT_SEED):
    """Convenience wrapper. X: DataFrame containing `focus_col`;
    clusters: array-like. Returns the full inference dict."""
    focus_idx = list(X.columns).index(focus_col)
    ols = WildClusterOLS(np.asarray(y, dtype=float),
                         X.to_numpy(dtype=np.float64),
                         np.asarray(clusters), focus_idx).fit()
    return ols.report(reps=reps, seed=seed)


