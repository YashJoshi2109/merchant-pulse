"""Detection ensemble (figure 3).

Three detectors with different blind spots vote on every series-day:
  ResidualDetector      robust Holt-Winters (weekly seasonality, damped trend, holiday + SNAP regressors)
                        in log space; score = standardized one-step-ahead residual. Stand-in for Prophet/SARIMAX.
  IsolationForestDetector  multivariate outlier score on the feature row (sklearn).
  AutoencoderDetector   dense autoencoder on 28-day standardized windows (sklearn MLP); reconstruction
                        error on the last 7 days. Stand-in for the LSTM autoencoder.
Each threshold is set on a clean calibration slice to a false-alert budget, then the vote needs 2 of 3.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import nbinom, norm
from sklearn.ensemble import IsolationForest
from sklearn.neural_network import MLPRegressor

from . import config
from .features import FEATURE_COLS

HORIZON_DAYS = {"spike": 1, "drop": 3, "drift": 7, "stockout": 5, "price_error": 2}


def _pivot(feats: pd.DataFrame, col: str) -> pd.DataFrame:
    return feats.pivot(index="date", columns="id", values=col).sort_index()


def hw_one_step(y: np.ndarray, alpha: np.ndarray, beta: np.ndarray, gamma: np.ndarray, effects: np.ndarray | None = None,
                phi: float = 0.98, m: int = 7, clip_sigma: np.ndarray | None = None) -> np.ndarray:
    """Vectorized additive Holt-Winters, one-step-ahead forecasts for every column at once.

    y: (T, N). alpha/beta/gamma: (N,) smoothing params per series. effects: (T, N) known regressor
    effects (holiday, SNAP) added to the forecast. clip_sigma: (N,) robust sigma; innovations are clipped
    (already multiplied by the caller) before updating state so a single anomaly does not drag the level with it.
    """
    T, N = y.shape
    init = y[: 4 * m]
    level = init.mean(0)
    trend = np.zeros(N)
    season = np.stack([init[k::m].mean(0) - level for k in range(m)])
    yhat = np.empty_like(y)
    eff = effects if effects is not None else np.zeros_like(y)
    for t in range(T):
        s = season[t % m]
        f = level + phi * trend + s + eff[t]
        yhat[t] = f
        e = y[t] - f
        if clip_sigma is not None:
            e = np.clip(e, -clip_sigma, clip_sigma)
        level, trend = level + phi * trend + alpha * e, phi * trend + beta * e
        season[t % m] = s + gamma * e
    return yhat


def _rolling_sum(a: np.ndarray, w: int) -> np.ndarray:
    """Rolling sum over axis 0 with a shrinking window at the start."""
    c = np.cumsum(a, axis=0)
    out = c.copy()
    out[w:] = c[w:] - c[:-w]
    return out


def hw_run(y: np.ndarray, alpha, beta, gamma, effects: np.ndarray, phi: float = 0.98, m: int = 7,
           clip_sigma: np.ndarray | None = None, update_mask: np.ndarray | None = None, censor_log: float = np.log1p(2.0)):
    """Like hw_one_step but also returns the post-update states: level (T,N), trend (T,N), season (T,m,N).

    Outlier-robust filtering: a day is not learned from when update_mask[t] is False (an alerted day), or when
    sales are zero while the forecast is above `censor_log` (a likely stockout: demand was censored, not absent).
    The forecast is still produced for that day; only the state update is skipped.
    """
    T, N = y.shape
    init = y[: 4 * m]
    level = init.mean(0); trend = np.zeros(N)
    season = np.stack([init[k::m].mean(0) - level for k in range(m)])
    yhat = np.empty_like(y); L = np.empty_like(y); B = np.empty_like(y); S = np.empty((T, m, N))
    for t in range(T):
        s = season[t % m]
        f = level + phi * trend + s + effects[t]
        yhat[t] = f
        e = y[t] - f
        if clip_sigma is not None:
            e = np.clip(e, -clip_sigma, clip_sigma)
        learn = ~((y[t] == 0) & (f > censor_log))
        if update_mask is not None:
            learn &= update_mask[t]
        e = np.where(learn, e, 0.0)
        level, trend = level + phi * trend + alpha * e, phi * trend + beta * e
        season[t % m] = s + gamma * e
        L[t] = level; B[t] = trend; S[t] = season
    return yhat, L, B, S


def frozen_origin_test(u: np.ndarray, L: np.ndarray, B: np.ndarray, S: np.ndarray, eff: np.ndarray, w: int,
                       k: np.ndarray, bias: np.ndarray, phi: float = 0.98, m: int = 7) -> np.ndarray:
    """z for the window ending at t: sum of (actual - forecast made at t-w) over the window / sqrt(sum var).

    The forecast origin is frozen before the window, so a drop or drift inside the window cannot be absorbed
    by the model's own adaptation. Returns (z, deviation_ratio), each (T, N) with NaN for t < w, where
    deviation_ratio = (actual - expected) / expected over the window (the effect size).
    """
    T, N = u.shape
    D = np.zeros((T - w, N)); V = np.zeros((T - w, N)); E = np.zeros((T - w, N))
    o = np.arange(T - w)
    for h in range(1, w + 1):
        S_h = sum(phi ** j for j in range(1, h + 1))
        seas = S[o, (o + h) % m, :]
        f = L[o] + B[o] * S_h + seas + eff[o + h]
        mu = (np.expm1(f) * bias).clip(min=0.05)
        D += u[o + h] - mu
        V += mu + mu ** 2 / k
        E += mu
    z = np.full((T, N), np.nan); dev = np.full((T, N), np.nan)
    z[w:] = D / np.sqrt(V)
    dev[w:] = D / np.maximum(E, 1e-6)
    return z, dev


@dataclass
class ResidualDetector:
    """Forecast-residual detector on the count scale, at four time scales.

    Fast Holt-Winters (per-series tuned smoothing) gives the expected units mu for today; z_1d is the exact
    negative-binomial tail of today's count and z_3d the 3-day cumulative residual over Poisson-gamma noise
    sqrt(mu + mu^2/k). z_7d, z_14d and z_28d are frozen-origin tests on a slow Holt-Winters: the forecast is
    made w days ago and the window's shortfall is summed, so a drop or drift inside the window cannot be
    absorbed by adaptation. Each scale is standardized on train and thresholded on the calibration slice.
    """
    name: str = "residual"
    params: pd.DataFrame | None = None          # per-series alpha, beta, gamma, sigma_log, k (dispersion)
    event_effect: dict = field(default_factory=dict)   # (event_name, cat) -> mean log effect
    snap_effect: dict = field(default_factory=dict)    # (state, cat) -> mean log effect
    elasticity: dict = field(default_factory=dict)     # cat -> log-log price elasticity (promo regressor)
    item_elasticity: dict = field(default_factory=dict)  # id -> shrunk per-series elasticity
    yearly: dict = field(default_factory=dict)         # cat -> Fourier coefficients for day-of-year effect
    thresholds: dict = field(default_factory=lambda: {s: 3.0 for s in ("1d", "3d", "7d", "14d", "28d")})
    slow_params: tuple = (0.05, 0.0, 0.03)
    clip_mult: float = 1.5
    n_harmonics: int = 3
    scale_sd: dict = field(default_factory=dict)           # innovations clipped at clip_mult * sigma before updating state

    def effects_long(self, feats: pd.DataFrame) -> np.ndarray:
        """Known log-scale effects per row of feats: holiday + SNAP + price elasticity + yearly Fourier."""
        n = len(feats)
        ev = np.zeros(n)
        if self.event_effect:
            s = pd.Series(self.event_effect)
            key = pd.MultiIndex.from_arrays([feats.event_name.values, feats.cat_id.values])
            ev = s.reindex(key).fillna(0.0).values
        sn = np.zeros(n)
        if self.snap_effect:
            s = pd.Series(self.snap_effect)
            key = pd.MultiIndex.from_arrays([feats.state_id.values, feats.cat_id.values])
            sn = s.reindex(key).fillna(0.0).values * feats.snap.values
        el_coef = feats.id.map(self.item_elasticity)
        el_coef = el_coef.fillna(feats.cat_id.map(self.elasticity)).fillna(0.0).values
        el = el_coef * np.log(feats.price_ratio.clip(0.2, 5.0).values)
        yr = np.zeros(n)
        if self.yearly:
            X = self._fourier(pd.to_datetime(feats.date))
            for cat, coef in self.yearly.items():
                m = (feats.cat_id == cat).values
                yr[m] = X[m] @ coef
        return ev + sn + el + yr

    def _effects_matrix(self, feats: pd.DataFrame, ids: list[str]) -> np.ndarray:
        tmp = feats[["date", "id"]].copy(); tmp["eff"] = self.effects_long(feats)
        return tmp.pivot(index="date", columns="id", values="eff").sort_index()[ids].values

    def _fourier(self, dates: pd.Series) -> np.ndarray:
        doy = dates.dt.dayofyear.values / 365.25
        cols = []
        for h in range(1, self.n_harmonics + 1):
            cols += [np.sin(2 * np.pi * h * doy), np.cos(2 * np.pi * h * doy)]
        return np.stack(cols, 1)

    def _long_resid(self, resid: np.ndarray, dates, ids, mask, feats: pd.DataFrame) -> pd.DataFrame:
        r = pd.DataFrame(resid[mask], index=dates[mask], columns=ids).stack()
        r.index.names = ["date", "id"]; r = r.rename("resid").reset_index()
        return r.merge(feats[["date", "id", "cat_id", "state_id", "event_name", "snap", "price_ratio"]], on=["date", "id"], how="left")

    @staticmethod
    def _promo_runs(feats: pd.DataFrame, train_end: pd.Timestamp, u: np.ndarray, dates, ids: list[str]) -> pd.DataFrame:
        """One row per promo run in train: mean log price ratio, log lift vs the 28 non-promo days before it."""
        fr = feats[feats.date < train_end].sort_values(["id", "date"])
        chg = fr.groupby("id").promo.diff().fillna(0).ne(0)
        fr = fr.assign(run=chg.groupby(fr.id).cumsum())
        g = fr[fr.promo == 1].groupby(["id", "run"]).agg(start=("date", "min"), n=("date", "size"), units=("units", "mean"),
                                                        lp=("price_ratio", lambda x: float(np.log(x.clip(0.2, 5.0)).mean())),
                                                        cat_id=("cat_id", "first"), item_id=("item_id", "first"), event=("event", "max"))
        g = g[(g.n >= 4) & (g.event == 0)].reset_index()
        if g.empty:
            return g
        col = {s: i for i, s in enumerate(ids)}
        pos = pd.Series(np.arange(len(dates)), index=dates)
        promo_m = fr.pivot(index="date", columns="id", values="promo").reindex(dates)[ids].values
        base = []
        for _, r in g.iterrows():
            i = int(pos.get(r.start, -1)); c = col[r.id]
            lo = max(i - 35, 0)
            if i - lo < 10:
                base.append(np.nan); continue
            seg = u[lo:i, c]; pm = promo_m[lo:i, c]
            seg = seg[(pm == 0) | np.isnan(pm)]
            base.append(seg.mean() if len(seg) >= 7 else np.nan)
        g["baseline"] = base
        g = g[np.isfinite(g.baseline) & (g.baseline >= 1.0)]
        g["lift"] = np.log((g.units + 0.5) / (g.baseline + 0.5))
        return g

    def fit(self, feats: pd.DataFrame, train_end: pd.Timestamp) -> "ResidualDetector":
        Y = _pivot(feats, "log_units"); U = _pivot(feats, "units")
        ids = list(Y.columns); dates = Y.index
        y = Y.values; u = U.values.astype(float)
        train_mask = np.asarray(dates < train_end)
        tm = train_mask.copy(); tm[: 4 * 7] = False     # skip init period
        N = len(ids)
        best_mse = np.full(N, np.inf); best = np.zeros((N, 3))
        grid = list(itertools.product([0.1, 0.2, 0.35], [0.0, 0.02], [0.05, 0.15, 0.3]))
        for a, b, g in grid:
            yh = hw_one_step(y, np.full(N, a), np.full(N, b), np.full(N, g))
            mse = ((y - yh)[tm] ** 2).mean(0)
            better = mse < best_mse
            best_mse[better] = mse[better]; best[better] = (a, b, g)
        alpha, beta, gamma = best[:, 0], best[:, 1], best[:, 2]
        resid = y - hw_one_step(y, alpha, beta, gamma)
        r_long = self._long_resid(resid, dates, ids, train_mask, feats)
        ev = r_long[r_long.event_name != ""].groupby(["event_name", "cat_id"]).resid.mean()
        self.event_effect = {k: float(v) for k, v in ev.items()}
        sn = r_long[(r_long.snap == 1) & (r_long.event_name == "")].groupby(["state_id", "cat_id"]).resid.mean()
        self.snap_effect = {k: float(v) for k, v in sn.items()}
        # promo regressor: elasticity from promo runs against a frozen pre-promo baseline (an adaptive model would
        # absorb part of the lift and bias the slope toward zero). Pooled per item across stores, shrunk to category.
        a_s, b_s, g_s = self.slow_params
        self.elasticity, self.item_elasticity = {}, {}
        runs = self._promo_runs(feats, train_end, u, dates, ids)
        if len(runs):
            for cat, grp in runs.groupby("cat_id"):
                if len(grp) >= 20 and grp.lp.var() > 0:
                    self.elasticity[cat] = float(np.clip(np.cov(grp.lp, grp.lift)[0, 1] / grp.lp.var(), -4.0, 0.0))
            item_el = {}
            for item, grp in runs.groupby("item_id"):
                prior = self.elasticity.get(grp.cat_id.iloc[0], -1.0)
                if len(grp) >= 4 and grp.lp.var() > 0:
                    slope = float(np.clip(np.cov(grp.lp, grp.lift)[0, 1] / grp.lp.var(), -5.0, 0.5))
                    w = len(grp) / (len(grp) + 8.0)
                    item_el[item] = float(np.clip(w * slope + (1 - w) * prior, -4.0, 0.0))
            for sid in ids:
                item = sid.rsplit("_", 2)[0]
                if item in item_el:
                    self.item_elasticity[sid] = item_el[item]
        eff = self._effects_matrix(feats, ids)
        resid_s = y - hw_one_step(y, np.full(N, a_s), np.full(N, b_s), np.full(N, g_s), effects=eff)
        rs = self._long_resid(resid_s, dates, ids, train_mask, feats)
        rs = rs[rs.event_name == ""]
        X_all = self._fourier(pd.to_datetime(rs.date))
        self.yearly = {}
        for cat in rs.cat_id.unique():
            m = (rs.cat_id == cat).values
            X = X_all[m]; yv = rs.resid.values[m]
            coef, *_ = np.linalg.lstsq(np.c_[np.ones(len(X)), X], yv, rcond=None)
            self.yearly[cat] = coef[1:]        # intercept is absorbed by the level
        eff = self._effects_matrix(feats, ids)
        resid = y - hw_one_step(y, alpha, beta, gamma, effects=eff)
        mad = np.median(np.abs(resid[tm] - np.median(resid[tm], axis=0)), axis=0)
        sigma = np.maximum(1.4826 * mad, 0.12)
        yh = hw_one_step(y, alpha, beta, gamma, effects=eff, clip_sigma=self.clip_mult * sigma)
        resid = y - yh
        mad = np.median(np.abs(resid[tm] - np.median(resid[tm], axis=0)), axis=0)
        sigma = np.maximum(1.4826 * mad, 0.12)
        # bias factor: a log-space forecast is a geometric mean; scale it to the arithmetic mean on train
        raw_mu = np.expm1(yh).clip(min=0.05)
        bias = np.clip(u[tm].mean(0) / raw_mu[tm].mean(0), 0.5, 3.0)
        mu = raw_mu * bias
        # dispersion k per series from train residuals on the count scale: var = mu + mu^2/k
        rc = (u - mu)[tm]; m_tr = mu[tm]
        excess = np.maximum((rc ** 2 - m_tr).mean(0), 1e-6)
        k = np.clip((m_tr ** 2).mean(0) / excess, 0.5, 50.0)
        self.params = pd.DataFrame({"id": ids, "alpha": alpha, "beta": beta, "gamma": gamma, "sigma": sigma, "k": k,
                                    "bias": bias, "train_mse": best_mse})
        # standardize the frozen-origin tests on train so thresholds are comparable across scales
        _, Lf, Bf, Sf = hw_run(y, alpha, beta, gamma, eff, clip_sigma=self.clip_mult * sigma)
        _, Ls, Bs, Ss = hw_run(y, np.full(N, a_s), np.full(N, b_s), np.full(N, g_s), eff, clip_sigma=self.clip_mult * sigma)
        self.scale_sd = {}
        for w, states in ((3, (Lf, Bf, Sf)), (7, (Ls, Bs, Ss)), (14, (Ls, Bs, Ss)), (28, (Ls, Bs, Ss))):
            zw, _ = frozen_origin_test(u, *states, eff, w, k, bias)
            v = zw[tm]; v = v[np.isfinite(v)]
            self.scale_sd[f"{w}d"] = float(max(1.4826 * np.median(np.abs(v - np.median(v))), 0.3))
        return self

    def score(self, feats: pd.DataFrame, exclude: pd.DataFrame | None = None) -> pd.DataFrame:
        """Long frame [id, date, yhat_log, yhat_units, z_1d, z_3d, z_7d, z_14d, z_28d, z_resid, z_scale].

        `exclude`: optional long frame [id, date] of days the models must not learn from (pass-1 alerts)."""
        Y = _pivot(feats, "log_units"); U = _pivot(feats, "units"); ids = list(Y.columns)
        mask = None
        if exclude is not None and len(exclude):
            ex = exclude.assign(v=False).pivot(index="date", columns="id", values="v").reindex(index=Y.index, columns=ids)
            mask = ex.fillna(True).values.astype(bool)
        p = self.params.set_index("id").loc[ids]
        eff = self._effects_matrix(feats, ids)
        y = Y.values; u = U.values.astype(float)
        cs = self.clip_mult * p.sigma.values
        yh_fast, Lf, Bf, Sf = hw_run(y, p.alpha.values, p.beta.values, p.gamma.values, eff, clip_sigma=cs, update_mask=mask)
        a, b, g = self.slow_params
        N = len(ids)
        k = p.k.values; bias = p.bias.values
        mu_f = (np.expm1(yh_fast) * bias).clip(min=0.05); var_f = mu_f + mu_f ** 2 / k
        r_f = u - mu_f
        # 1-day test: exact negative-binomial tail mapped to a normal quantile, so heavy count tails are not over-read
        pnb = k / (k + mu_f)
        upper = nbinom.sf(u - 1, k, pnb)       # P(X >= u)
        lower = nbinom.cdf(u, k, pnb)          # P(X <= u)
        z1 = np.where(u >= mu_f, norm.isf(np.clip(upper, 1e-12, 1)), -norm.isf(np.clip(lower, 1e-12, 1)))
        z3, d3 = frozen_origin_test(u, Lf, Bf, Sf, eff, 3, k, bias); z3 = z3 / self.scale_sd.get("3d", 1.0)
        np.nan_to_num(z3, copy=False, nan=0.0)
        _, Ls, Bs, Ss = hw_run(y, np.full(N, a), np.full(N, b), np.full(N, g), eff, clip_sigma=cs, update_mask=mask)
        z7, d7 = frozen_origin_test(u, Ls, Bs, Ss, eff, 7, k, bias); z7 = z7 / self.scale_sd.get("7d", 1.0)
        z14, d14 = frozen_origin_test(u, Ls, Bs, Ss, eff, 14, k, bias); z14 = z14 / self.scale_sd.get("14d", 1.0)
        z28, d28 = frozen_origin_test(u, Ls, Bs, Ss, eff, 28, k, bias); z28 = z28 / self.scale_sd.get("28d", 1.0)
        d1 = u / mu_f - 1.0
        for arr in (z7, z14, z28, d3, d7, d14, d28):
            np.nan_to_num(arr, copy=False, nan=0.0)
        T = len(Y.index)
        out = pd.DataFrame({"id": np.tile(ids, T), "date": np.repeat(Y.index.values, N),
                            "yhat_log": yh_fast.ravel(), "yhat_units": mu_f.ravel(), "k": np.tile(k, T),
                            "z_1d": z1.ravel(), "z_3d": z3.ravel(), "z_7d": z7.ravel(), "z_14d": z14.ravel(), "z_28d": z28.ravel(),
                            "dev_1d": d1.ravel(), "dev_3d": d3.ravel(), "dev_7d": d7.ravel(), "dev_14d": d14.ravel(), "dev_28d": d28.ravel()})
        return self.pick(out)

    SCALES = ("1d", "3d", "7d", "14d", "28d")

    # significance alone is not enough on long windows: a +10% month on a high-volume series is real but not
    # actionable. Each scale also needs at least this relative deviation over its window.
    min_effect: dict = field(default_factory=lambda: {"1d": 0.0, "3d": 0.30, "7d": 0.25, "14d": 0.30, "28d": 0.35})

    def pick(self, scores: pd.DataFrame) -> pd.DataFrame:
        """Choose the time scale whose test is furthest past its own threshold; report its signed z."""
        ratios = np.stack([scores[f"z_{s}"].abs().values / self.thresholds[s]
                           * (scores[f"dev_{s}"].abs().values >= self.min_effect[s]) for s in self.SCALES])
        idx = ratios.argmax(0)
        zs = np.stack([scores[f"z_{s}"].values for s in self.SCALES])
        scores = scores.copy()
        scores["z_resid"] = np.take_along_axis(zs, idx[None], 0)[0]
        scores["z_scale"] = np.array(self.SCALES)[idx]
        scores["z_ratio"] = np.take_along_axis(ratios, idx[None], 0)[0]
        return scores

    def flag(self, scores: pd.DataFrame) -> np.ndarray:
        return (scores["z_ratio"] > 1.0).values


@dataclass
class IsolationForestDetector:
    name: str = "iforest"
    model: IsolationForest | None = None
    threshold: float = 0.6
    cols: list[str] = field(default_factory=lambda: list(FEATURE_COLS) + ["log_units_adj"])
    # continuous features are standardized within each series on train, so "unusual" means unusual for that
    # item at that store rather than unusual for the pooled population of 280 series of very different volume
    std_cols: tuple = ("log_units", "z28", "ratio_7", "lag7_ratio", "price_pct", "drift_ratio", "log_units_adj")
    mu: pd.DataFrame | None = None
    sd: pd.DataFrame | None = None

    def _transform(self, feats: pd.DataFrame) -> np.ndarray:
        X = feats[self.cols].astype(float).fillna(0.0).copy()
        if self.mu is not None:
            m = self.mu.reindex(feats.id).values; s = self.sd.reindex(feats.id).values
            X[list(self.std_cols)] = (X[list(self.std_cols)].values - np.nan_to_num(m)) / np.where(np.isfinite(s), s, 1.0)
        return X.values

    def fit(self, feats: pd.DataFrame, train_end: pd.Timestamp, seed: int = config.SEED, max_rows: int = 60000):
        tr = feats[(feats.date < train_end) & feats.rm28.notna()]
        g = tr.groupby("id")[list(self.std_cols)]
        self.mu = g.mean(); self.sd = g.std().clip(lower=1e-3)
        if len(tr) > max_rows:
            tr = tr.sample(max_rows, random_state=seed)
        self.model = IsolationForest(n_estimators=200, max_samples=1024, contamination="auto", random_state=seed, n_jobs=-1)
        self.model.fit(self._transform(tr))
        return self

    def score(self, feats: pd.DataFrame) -> np.ndarray:
        return -self.model.score_samples(self._transform(feats))     # higher = more anomalous

    def flag(self, score: np.ndarray) -> np.ndarray:
        return score > self.threshold


@dataclass
class AutoencoderDetector:
    name: str = "autoencoder"
    model: MLPRegressor | None = None
    mu: pd.Series | None = None
    sd: pd.Series | None = None
    threshold: float = 0.5
    window: int = config.WINDOW

    col: str = "log_units_adj"

    def _windows(self, feats: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
        Y = _pivot(feats, self.col); ids = list(Y.columns)
        y = (Y.values - self.mu.loc[ids].values) / self.sd.loc[ids].values
        W = self.window
        T, N = y.shape
        win = np.lib.stride_tricks.sliding_window_view(y, W, axis=0)    # (T-W+1, N, W)
        idx = pd.DataFrame({"id": np.tile(ids, T - W + 1), "date": np.repeat(Y.index.values[W - 1:], N)})
        return win.reshape(-1, W), idx

    def fit(self, feats: pd.DataFrame, train_end: pd.Timestamp, seed: int = config.SEED, max_rows: int = 40000):
        tr = feats[feats.date < train_end]
        self.mu = tr.groupby("id")[self.col].mean(); self.sd = tr.groupby("id")[self.col].std().clip(lower=0.2)
        X, idx = self._windows(feats)
        mask = (idx.date < train_end).values
        X = X[mask]
        rng = np.random.default_rng(seed)
        if len(X) > max_rows:
            X = X[rng.choice(len(X), max_rows, replace=False)]
        self.model = MLPRegressor(hidden_layer_sizes=(16, 4, 16), activation="tanh", max_iter=80, early_stopping=True,
                                  random_state=seed, learning_rate_init=0.003)
        self.model.fit(X, X)
        return self

    def score(self, feats: pd.DataFrame) -> pd.DataFrame:
        """Long frame [id, date, ae_score]; first window-1 days of each series get NaN."""
        X, idx = self._windows(feats)
        rec = self.model.predict(X)
        err = ((rec - X)[:, -7:] ** 2).mean(1)
        idx["ae_score"] = err
        return idx

    def flag(self, score: np.ndarray) -> np.ndarray:
        return score > self.threshold


def classify_type(row: pd.Series) -> str:
    if row.zero_streak >= 2 and row.zero_run_logp < -9.2:
        return "stockout"
    if row.price_ratio >= 3 or row.price_ratio <= 1 / 3:
        return "price_error"
    if row.z_resid > 0:
        return "spike"
    # gradual decline: a long-window test fires while the shortfall is not concentrated in the last week
    if row.z_scale in ("14d", "28d") and abs(row.z_3d) < 2.0 and abs(row.z_7d) < 0.85 * abs(row.z_resid):
        return "drift"
    return "drop"


@dataclass
class Ensemble:
    residual: ResidualDetector = field(default_factory=ResidualDetector)
    iforest: IsolationForestDetector = field(default_factory=IsolationForestDetector)
    autoenc: AutoencoderDetector = field(default_factory=AutoencoderDetector)
    vote_min: int = config.VOTE_MIN
    min_abs_z: float = 2.0           # significance floor: the forecast must disagree at 2 sigma
    scale_share: dict = field(default_factory=lambda: {"1d": 0.05, "3d": 0.30, "7d": 0.25, "14d": 0.20, "28d": 0.20})
    strong_ratio: float = 1.25       # the calibrated residual test at 1.25x its threshold alone counts as two votes
    big_effect: float = 0.40         # a significant 7d+ test with >= 40% deviation over its window needs no corroboration
    max_exclude_days: int = 28       # longest run of alerted days withheld from state updates (longer than any planted anomaly)
    min_abs_z_uncorroborated: float = 2.5   # alerts without the calibrated residual vote need a stronger residual
    min_volume: float = 4.0          # materiality floor: series selling < 4 units/day roll up to category level
    zero_run_logp: float = -9.2      # stockout rule: P(this many zero days | expected sales) < 1e-4
    fa_budget: float = config.PER_DETECTOR_FA
    calibration: dict = field(default_factory=dict)

    def _adjust(self, feats: pd.DataFrame) -> pd.DataFrame:
        feats = feats.copy()
        feats["log_units_adj"] = feats.log_units.values - self.residual.effects_long(feats)
        return feats

    def fit(self, feats: pd.DataFrame, train_end: pd.Timestamp, calib_start: pd.Timestamp) -> "Ensemble":
        self.residual.fit(feats, train_end)
        feats = self._adjust(feats)
        self.iforest.fit(feats, train_end)
        self.autoenc.fit(feats, train_end)
        sc = self._raw_scores(feats)
        cal = sc[(sc.date >= calib_start) & (sc.date < train_end) & (sc.rm28 >= self.min_volume)]
        # thresholds are set on pass-1 scores; pass-2 scores on clean days are at least as tight
        q = 1 - self.fa_budget
        self.residual.thresholds = {s: float(max(cal[f"z_{s}"].abs().quantile(1 - self.fa_budget * self.scale_share[s]), 2.5))
                                    for s in ResidualDetector.SCALES}
        self.iforest.threshold = float(cal.if_score.quantile(q))
        self.autoenc.threshold = float(cal.ae_score.quantile(q))
        cal = self.residual.pick(cal)
        cal = self._vote(cal)
        self.calibration = {
            "calib_start": str(calib_start.date()), "train_end": str(train_end.date()), "rows": int(len(cal)),
            "thresholds": {"residual_by_scale": self.residual.thresholds, "iforest": self.iforest.threshold,
                            "autoencoder": self.autoenc.threshold},
            "flag_rate": {"residual": float(cal.f_resid.mean()), "iforest": float(cal.f_if.mean()),
                           "autoencoder": float(cal.f_ae.mean()), "ensemble": float(cal.is_alert.mean())},
            "ensemble_fa_per_1k": float(cal.is_alert.mean() * 1000),
        }
        return self

    def _raw_scores(self, feats: pd.DataFrame, exclude: pd.DataFrame | None = None) -> pd.DataFrame:
        if "log_units_adj" not in feats.columns:
            feats = self._adjust(feats)
        r = self.residual.score(feats, exclude=exclude)
        a = self.autoenc.score(feats)
        base = feats[["id", "date", "store_id", "state_id", "item_id", "dept_id", "cat_id", "units", "sell_price",
                      "price_ref", "rm28", "event_name", "log_units_adj"] + FEATURE_COLS].copy()
        base["if_score"] = self.iforest.score(feats)
        sc = base.merge(r, on=["id", "date"], how="left").merge(a, on=["id", "date"], how="left")
        sc = sc.sort_values(["id", "date"]).reset_index(drop=True)
        # log-probability of the current zero-sales run under the fitted negative binomial
        lp0 = -sc.k * np.log1p(sc.yhat_units / sc.k)
        zero = (sc.units == 0)
        run = (~zero).groupby(sc.id).cumsum()
        sc["zero_run_logp"] = np.where(zero, lp0, 0.0)
        sc["zero_run_logp"] = sc.groupby([sc.id, run]).zero_run_logp.cumsum()
        return sc

    def _vote(self, sc: pd.DataFrame) -> pd.DataFrame:
        sc = sc.copy()
        sc["f_resid"] = self.residual.flag(sc)
        sc["f_if"] = self.iforest.flag(sc.if_score.fillna(0).values)
        sc["f_ae"] = self.autoenc.flag(sc.ae_score.fillna(0).values)
        sc["votes"] = sc[["f_resid", "f_if", "f_ae"]].sum(1).astype(int)
        # the calibrated residual test stands on its own when it is far past its threshold, or when a sustained
        # (7d+) test is significant AND the effect is large; otherwise two detectors must agree
        dev_picked = np.abs(np.take_along_axis(sc[[f"dev_{s}" for s in ResidualDetector.SCALES]].values,
                                               np.array([ResidualDetector.SCALES.index(s) for s in sc.z_scale])[:, None], 1)[:, 0])
        sustained = sc.f_resid & sc.z_scale.isin(["7d", "14d", "28d"]) & (dev_picked >= self.big_effect)
        strong = (sc.z_ratio >= self.strong_ratio) | sustained
        sc.loc[strong & (sc.votes == 1), "votes"] = 2
        # deterministic rules short-circuit the vote: a price file 3x off, or 3+ zero days on a series that sells
        sc["f_rule"] = (sc.price_ratio >= 3) | (sc.price_ratio <= 1 / 3) | \
                       ((sc.zero_streak >= 2) & (sc.zero_run_logp < self.zero_run_logp))
        floor = np.where(sc.f_resid, self.min_abs_z, self.min_abs_z_uncorroborated)
        statistical = (sc.votes >= self.vote_min) & (sc.z_resid.abs() >= floor) & (sc.rm28 >= self.min_volume)
        sc["is_alert"] = statistical | sc.f_rule
        sc["direction"] = np.where(sc.z_resid >= 0, "up", "down")
        return sc

    def two_pass(self, feats: pd.DataFrame) -> pd.DataFrame:
        """Pass 1 finds alert days; pass 2 re-scores with those days excluded from state updates, so an
        anomaly cannot drag the forecast and manufacture a second alert when sales return to normal."""
        if "log_units_adj" not in feats.columns:
            feats = self._adjust(feats)
        sc1 = self._vote(self.residual.pick(self._raw_scores(feats)))
        # only suppressed-demand days (negative anomalies) are excluded: sales were censored, demand was not.
        # Positive shifts are still learned (clipped), and a run of excluded days is capped so a genuine new
        # level is eventually accepted instead of being flagged forever.
        neg = sc1[sc1.is_alert & (sc1.z_resid < 0)].sort_values(["id", "date"])
        if neg.empty:
            return sc1
        gap = neg.groupby("id").date.diff().dt.days.fillna(99)
        run = (gap > 1).groupby(neg.id).cumsum()
        pos_in_run = neg.groupby([neg.id, run]).cumcount()
        ex = neg.loc[pos_in_run < self.max_exclude_days, ["id", "date"]]
        return self._vote(self.residual.pick(self._raw_scores(feats, exclude=ex)))

    def score(self, feats: pd.DataFrame, start: pd.Timestamp | None = None, end: pd.Timestamp | None = None) -> pd.DataFrame:
        """Full scoring frame for [start, end]. Needs full history in feats for state, slices afterwards."""
        sc = self.two_pass(feats)
        if start is not None:
            sc = sc[sc.date >= start]
        if end is not None:
            sc = sc[sc.date <= end]
        sc = sc.copy()
        sc["anomaly_type"] = ""
        m = sc.is_alert
        sc.loc[m, "anomaly_type"] = sc[m].apply(classify_type, axis=1)
        delta = (sc.yhat_units - sc.units).abs()
        horizon = sc.anomaly_type.map(HORIZON_DAYS).fillna(1)
        sc["revenue_at_risk"] = (delta * sc.price_ref * horizon).round(2)
        return sc.reset_index(drop=True)
