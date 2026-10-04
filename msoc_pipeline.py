#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
msoc_pipeline.py - open-source (Python) implementation of MSOC_ML_Pipeline.m

Classification of Player Load Intensity (PLI) in NCAA-DI men's soccer from
internal-load, external-load and wellness metrics, with leave-one-athlete-out
cross-validation of seven models (SVM, AdaBoost, ordinal logistic, multinomial
logistic, ordinal mixed model, linear mixed model, support vector regression).

This script follows the MATLAB script step by step and writes the same tables
(CSV) to MSOC_results_python/.  See README.md, section "Python implementation",
for which components follow the same definitions as the MATLAB script and which
use different algorithms or solvers. Numerical equivalence with the MATLAB
results has not been established.  The results reported in the manuscript are
those of the MATLAB script; this implementation has been run end to end on the
simulated workbook (MSOC_Data_SYNTHETIC.xlsx) but has not been run on the
original data.

Known differences from the MATLAB script (see README):
  * AdaBoost: scikit-learn SAMME (labelled AdaBoost_SAMME in the outputs)
    instead of MATLAB AdaBoostM2 (pseudo-loss boosting).
  * SVM/SVR solvers and the Bayesian optimizer differ in implementation.
The proportional-odds assumption (section 9) is tested with the Brant test,
as in the MATLAB script (brant_wald); the bootstrap p-values use NumPy's random
stream and therefore differ slightly from MATLAB's.

Requirements: Python >= 3.9, numpy, pandas, scipy, scikit-learn, statsmodels,
scikit-optimize, openpyxl.   pip install numpy pandas scipy scikit-learn statsmodels scikit-optimize openpyxl

Usage:
    python msoc_pipeline.py                 # full analysis (about 1-2 h)
    python msoc_pipeline.py --quick         # dry run: 12 tuning evaluations, primary strategy only
    python msoc_pipeline.py --data "MSOC Data.xlsx" --out MSOC_results_python
"""
import argparse, os, sys, json, time, platform
import numpy as np, pandas as pd
from scipy import stats, optimize
from scipy.special import expit
from itertools import combinations
import itertools

# (no global warning filter: statsmodels convergence warnings are expected for some folds and are left visible)

# ----------------------------------------------------------------------------- configuration
CFG = dict(
    data_file='MSOC Data.xlsx', ml_sheet='ML Sheet (2)', polar_sheet='ALL POLAR DATA', out_dir='MSOC_results_python',
    seed=20260925,
    cut_rule='meanHalfSD',        # 'meanHalfSD' or 'fixed'
    cut_sample='analytic',        # 'analytic' or 'allPlayers'
    fixed_cut_points=(75.0, 150.0),
    min_sessions_per_athlete=10,
    exclude_zero_pli=True,
    bayes_evals=40,               # total tuning evaluations per fold and model, incl. the fixed initial design
    n_gh=25,                      # Gauss-Hermite nodes (ordinal mixed model)
    lmm_interact=True,
    mrmr_bins=10,
    quick=False,
)
FEAT_VAR = ['sRPE', 'Fatigue', 'Stress', 'Soreness', 'HRpct', 'TD']
FEAT_LABEL = ['sRPE', 'Wellness-Fatigue', 'Wellness-Stress', 'Wellness-Soreness', '%HRmax', 'Total distance']
ML_VARS = ['Prim_V_Sec', 'Ath_ID', 'Prac_o_Gam', 'Return', 'sRPE', 'Well1_Fat', 'Well2_Stress', 'Well3_Sore',
           'P_HR_avg_%', 'P_Tot_Dis', 'P_Training_load_score']
POLAR_VARS = ['Ath_ID', 'Duration', 'Time_HR_zone_1', 'Time_HR_zone_2', 'Time_HR_zone_3', 'Time_HR_zone_4',
              'Time_HR_zone_5', 'P_Training_load_score', 'P_Cardio_load']
MODEL_NAMES = ['SVM', 'AdaBoost_SAMME', 'OrdinalLogistic', 'Multinomial', 'CLMM', 'LMM_binned', 'SVR_binned']  # AdaBoost_SAMME = MATLAB AdaBoostM2 column
CLASS_NAMES = ['Low', 'Medium', 'High']
K = 3
STRATEGIES = [dict(name='Primary: within-fold MRMR top 3', mode='mrmr', k=3, features=None, seed_offset=0),
              dict(name='Sensitivity: no heart rate (sRPE + TD)', mode='fixed', k=2, features=[0, 5], seed_offset=0)]


# ----------------------------------------------------------------------------- helpers
def to_minutes(col):
    """Minutes from an Excel time column read by pandas.
    Time-like cells (timedelta, datetime.time, datetime, 'h:mm:ss' strings) are converted directly.
    Numeric cells are Excel serial times, i.e. fractions of a day, and are multiplied by 1440;
    the factor is applied only to numeric cells, never to values already converted from time objects."""
    out = np.full(len(col), np.nan)
    numeric = np.zeros(len(col), bool)
    for i, v in enumerate(col):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        if isinstance(v, pd.Timedelta):
            out[i] = v.total_seconds() / 60
        elif hasattr(v, 'hour') and hasattr(v, 'minute'):          # datetime.time or datetime
            out[i] = v.hour * 60 + v.minute + v.second / 60
        elif isinstance(v, str) and ':' in v:
            parts = [float(x) for x in v.strip().split(':')]
            out[i] = parts[0] * 60 + parts[1] + (parts[2] / 60 if len(parts) > 2 else 0)
        else:
            out[i] = float(v); numeric[i] = True
    if numeric.any():
        vals = out[numeric]
        if np.nanmax(vals) <= 1.5:      # Excel serial time: fraction of a day (durations up to 36 h)
            out[numeric] = vals * 1440
        # otherwise the numeric cells are already minutes
    return out


def pli_category(x, cut):
    c = np.full(len(x), 2, int)
    c[np.asarray(x) <= cut[0]] = 1
    c[np.asarray(x) >= cut[1]] = 3
    return c


def rmcorr(x, y, g):
    _, gi = np.unique(g, return_inverse=True)
    xc = x - np.bincount(gi, x) [gi] / np.bincount(gi)[gi]
    yc = y - np.bincount(gi, y)[gi] / np.bincount(gi)[gi]
    r = np.sum(xc * yc) / np.sqrt(np.sum(xc ** 2) * np.sum(yc ** 2))
    df = len(x) - gi.max() - 1 - 1
    t = r * np.sqrt(df / (1 - r ** 2))
    return r, 2 * stats.t.cdf(-abs(t), df), df


def holm(p):
    p = np.asarray(p, float); m = len(p); order = np.argsort(p)
    adj = np.minimum(1, np.maximum.accumulate((m - np.arange(m)) * p[order]))
    out = np.empty(m); out[order] = adj
    return out


def mutual_info(a, b):
    P = pd.crosstab(a, b).values.astype(float); P /= P.sum()
    PaPb = P.sum(1, keepdims=True) * P.sum(0, keepdims=True)
    nz = P > 0
    return float(np.sum(P[nz] * np.log(P[nz] / PaPb[nz])))


def mrmr_miq(X, y, q):
    """MRMR ranking with the mutual-information-quotient criterion (Ding & Peng, 2005).
    Continuous predictors are discretized into q equal-frequency bins whose edges are the
    (1..q-1)/q quantiles of the data passed in (MATLAB 'quantile' == numpy method 'hazen')."""
    n, p = X.shape
    B = np.zeros_like(X, dtype=int)
    for j in range(p):
        x = X[:, j]
        if len(np.unique(x)) <= q:
            B[:, j] = np.unique(x, return_inverse=True)[1] + 1
        else:
            e = np.unique(np.quantile(x, np.arange(1, q) / q, method='hazen'))
            B[:, j] = 1 + (x[:, None] > e[None, :]).sum(1)
    yc = np.unique(y, return_inverse=True)[1]
    rel = np.array([mutual_info(B[:, j], yc) for j in range(p)])
    order, score = [int(np.argmax(rel))], [float(rel.max())]
    remaining = [j for j in range(p) if j != order[0]]
    while remaining:
        miq = [rel[j] / max(np.mean([mutual_info(B[:, j], B[:, s]) for s in order]), np.finfo(float).eps) for j in remaining]
        i = int(np.argmax(miq)); order.append(remaining[i]); score.append(float(miq[i])); remaining.pop(i)
    return np.array(order), np.array(score)


def class_metrics(yt, yp, ath):
    C = np.zeros((K, K), int)
    for t, p in zip(yt, yp):
        C[t - 1, p - 1] += 1
    n = C.sum()
    acc_ath = np.array([np.mean(yp[ath == a] == yt[ath == a]) for a in np.unique(ath)])
    sens = np.diag(C) / C.sum(1); prec = np.diag(C) / np.maximum(C.sum(0), 1)
    spec = np.array([(n - C[k].sum() - C[:, k].sum() + C[k, k]) / (n - C[k].sum()) for k in range(K)])
    f1 = 2 * prec * sens / np.maximum(prec + sens, np.finfo(float).eps)
    return dict(C=C, acc=np.trace(C) / n, accAth=acc_ath, accMean=acc_ath.mean(), accSD=acc_ath.std(ddof=1),
                balAcc=sens.mean(), macroF1=f1.mean(), sens=sens, spec=spec, prec=prec, f1=f1, extremeErr=C[0, K - 1] + C[K - 1, 0])


def reg_metrics(y, yhat):
    e = y - yhat
    return float(np.sqrt(np.mean(e ** 2))), float(np.mean(np.abs(e))), float(1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2))


def mcnemar_exact(yt, p1, p2):
    c1, c2 = p1 == yt, p2 == yt
    b, c = int(np.sum(c1 & ~c2)), int(np.sum(~c1 & c2))
    p = 1.0 if b + c == 0 else min(1.0, 2 * stats.binom.cdf(min(b, c), b + c, 0.5))
    return b, c, p


# ----------------------------------------------------------------------------- models
from sklearn.svm import SVC, SVR
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from statsmodels.miscmodels.ordinal_model import OrderedModel
from statsmodels.discrete.discrete_model import MNLogit
import statsmodels.api as sm
import statsmodels.formula.api as smf


def fit_one(kind, hp, X, y):
    if kind == 'SVM':      # MATLAB kernel exp(-||x-z||^2 / KS^2)  ->  gamma = 1 / KS^2
        return make_pipeline(StandardScaler(), SVC(kernel='rbf', C=hp['C'], gamma=1.0 / hp['KS'] ** 2,
                                                   decision_function_shape='ovo')).fit(X, y)
    if kind == 'ADA':      # SAMME AdaBoost with depth-limited trees (MATLAB AdaBoostM2 uses the pseudo-loss variant)
        base = DecisionTreeClassifier(max_leaf_nodes=int(hp['MaxSplits']) + 1, random_state=0)
        return AdaBoostClassifier(estimator=base, n_estimators=int(hp['NumCycles']), learning_rate=hp['LearnRate'],
                                  random_state=0).fit(X, y)
    if kind == 'SVR':
        return make_pipeline(StandardScaler(), SVR(kernel='rbf', C=hp['C'], gamma=1.0 / hp['KS'] ** 2, epsilon=hp['Eps'])).fit(X, y)
    raise ValueError(kind)


def inner_loss(kind, hp, X, y, g):
    err = 0.0
    for a in np.unique(g):
        te = g == a
        m = fit_one(kind, hp, X[~te], y[~te]); yh = m.predict(X[te])
        err += np.sum((yh - y[te]) ** 2) if kind == 'SVR' else np.sum(yh != y[te])
    L = err / len(y)
    return float(np.log1p(L)) if kind == 'SVR' else float(L)


def tune_and_fit(kind, X, y, g, cfg, rng_seed):
    """Bayesian optimization (scikit-optimize gp_minimize, expected improvement) over the same ranges as the
    MATLAB script, starting from the same fixed initial design; returns model, best hyperparameters, inner loss."""
    from skopt import gp_minimize
    from skopt.space import Real, Integer
    if kind == 'SVM':
        space = [Real(1e-3, 1e3, prior='log-uniform', name='C'), Real(1e-3, 1e3, prior='log-uniform', name='KS')]
        x0 = [[c, k] for c in (0.1, 10, 1000) for k in (0.3, 3, 30)]
        names = ['C', 'KS']
    elif kind == 'ADA':
        space = [Integer(10, 500, prior='log-uniform', name='NumCycles'), Real(1e-3, 1.0, prior='log-uniform', name='LearnRate'),
                 Integer(1, 20, prior='log-uniform', name='MaxSplits')]
        x0 = [[a, b, c] for a in (30, 300) for b in (0.03, 0.3) for c in (2, 10)]
        names = ['NumCycles', 'LearnRate', 'MaxSplits']
    else:
        s = stats.iqr(y) / 1.349
        space = [Real(1e-3, 1e3, prior='log-uniform', name='C'), Real(1e-3, 1e3, prior='log-uniform', name='KS'),
                 Real(1e-3 * s, 1e2 * s, prior='log-uniform', name='Eps')]
        x0 = [[a, b, c] for a in (1, 100) for b in (1, 10) for c in (0.01 * s, 0.1 * s)]
        names = ['C', 'KS', 'Eps']
    obj = lambda v: inner_loss(kind, dict(zip(names, v)), X, y, g)
    n_calls = max(cfg['bayes_evals'], len(x0))
    res = gp_minimize(obj, space, x0=x0, n_calls=n_calls, n_initial_points=0, acq_func='EI', random_state=rng_seed, verbose=False)
    best = dict(zip(names, res.x)); best_loss = float(res.fun)
    return fit_one(kind, best, X, y), best, best_loss


def cluster_compare(cA, cB, cl, n_boot, n_perm, rng):
    """Cluster bootstrap 95% CI for the difference in pooled accuracy (A - B, percentage points) and a
    cluster-level sign-flip p-value (exact enumeration when there are <= 12 clusters)."""
    _, gi = np.unique(cl, return_inverse=True); nc = gi.max() + 1
    sA = np.bincount(gi, cA.astype(float), nc); sB = np.bincount(gi, cB.astype(float), nc); nn = np.bincount(gi, None, nc).astype(float)
    draws = rng.integers(0, nc, size=(n_boot, nc))
    diffs = 100 * (sA[draws].sum(1) - sB[draws].sum(1)) / nn[draws].sum(1)
    ci = np.percentile(diffs, [2.5, 97.5])
    d = sA - sB; obs = d.sum()
    if nc <= 12:
        signs = np.array(list(itertools.product([-1, 1], repeat=nc))); dist = signs @ d
    else:
        dist = (rng.choice([-1, 1], size=(n_perm, nc)) @ d)
    return ci, float(np.mean(np.abs(dist) >= abs(obs) - 1e-9))


def num_hessian(f, x, rel=1e-4):
    x = np.asarray(x, float); n = len(x); H = np.zeros((n, n)); h = rel * np.maximum(1, np.abs(x)); f0 = f(x)
    for i in range(n):
        ei = np.zeros(n); ei[i] = h[i]
        H[i, i] = (f(x + ei) - 2 * f0 + f(x - ei)) / h[i] ** 2
        for j in range(i + 1, n):
            ej = np.zeros(n); ej[j] = h[j]
            H[i, j] = H[j, i] = (f(x + ei + ej) - f(x + ei - ej) - f(x - ei + ej) + f(x - ei - ej)) / (4 * h[i] * h[j])
    return H


def cumlogit_np_nll(par, X, y):
    """Cumulative-logit model with threshold-specific slopes: logit P(Y<=k) = theta_k - x*beta_k."""
    n, p = X.shape
    theta = par[:K - 1]; B = par[K - 1:].reshape(K - 1, p).T
    C = expit(theta[None, :] - X @ B); C = np.column_stack([C, np.ones(n)])
    P = np.column_stack([C[:, 0], np.diff(C, axis=1)])
    Py = P[np.arange(n), y - 1]
    return -np.sum(np.log(np.maximum(Py, 1e-12))), P


def cumlogit_np_fit(X, y, beta0, theta0):
    """Non-parallel-slopes alternative of the proportional-odds test, maximised from the proportional-odds solution."""
    par0 = np.r_[theta0, np.tile(beta0, K - 1)]
    f = lambda par: cumlogit_np_nll(par, X, y)[0]
    r = optimize.minimize(f, par0, method='BFGS', options=dict(gtol=1e-8, maxiter=4000))
    par = r.x if r.fun <= f(par0) else par0
    g = optimize.approx_fprime(par, f, 1e-6 * np.maximum(1, np.abs(par)))
    _, P = cumlogit_np_nll(par, X, y)
    return dict(logLik=-f(par), converged=bool(r.success or np.max(np.abs(g)) < 1e-4), max_grad=float(np.max(np.abs(g))), n_negative=int((P < 0).any(1).sum()),
                theta=par[:K - 1], beta=par[K - 1:].reshape(K - 1, X.shape[1]).T)


def brant_wald(Z, y):
    """Brant (1990) test of proportional odds: binary logits for Y > k (k = 1..K-1), Wald test of equal
    slopes with Brant's covariance Cov(b_j, b_l) = A_j X'W_jl X A_l, W_jl = pi_l - pi_j pi_l (l >= j).
    Returns the omnibus statistic, the slope differences (first cut point minus each later one), their
    covariance, and the binary-logit coefficients and standard errors ((p+1) x (K-1))."""
    n, p = Z.shape; X = np.column_stack([np.ones(n), Z]); q = p + 1; J = K - 1
    B = np.zeros((q, J)); PI = np.zeros((n, J)); A = []
    for j in range(J):
        r = sm.GLM((np.asarray(y) > j + 1).astype(float), X, family=sm.families.Binomial()).fit(tol=1e-12, maxiter=200)
        B[:, j] = r.params; PI[:, j] = expit(X @ r.params)
        A.append(np.linalg.inv(X.T @ (X * (PI[:, j] * (1 - PI[:, j]))[:, None])))
    V = np.zeros((J * q, J * q))
    for j in range(J):
        for l in range(J):
            a, c = min(j, l), max(j, l)
            w = PI[:, c] - PI[:, a] * PI[:, c]
            V[j * q:(j + 1) * q, l * q:(l + 1) * q] = A[j] @ (X.T @ (X * w[:, None])) @ A[l]
    SE = np.sqrt(np.diag(V)).reshape(J, q).T
    idx = np.concatenate([np.arange(j * q + 1, (j + 1) * q) for j in range(J)])
    bs = B[1:, :].T.reshape(-1); Vs = V[np.ix_(idx, idx)]
    Dm = np.zeros(((J - 1) * p, J * p))
    for j in range(J - 1):
        Dm[j * p:(j + 1) * p, :p] = np.eye(p); Dm[j * p:(j + 1) * p, (j + 1) * p:(j + 2) * p] = -np.eye(p)
    d = Dm @ bs; Vd = Dm @ Vs @ Dm.T
    return float(d @ np.linalg.solve(Vd, d)), d, Vd, B, SE


# ---- ordinal models -----------------------------------------------------------
def ordinal_fit(Z, y):
    """Proportional-odds model; returns statsmodels result. Convention: logit P(Y<=k) = theta_k - x*beta."""
    return OrderedModel(np.asarray(y) - 1, Z, distr='logit').fit(method='bfgs', disp=False, maxiter=500)


def ordinal_predict(res, Z):
    return np.argmax(res.model.predict(res.params, exog=Z), axis=1) + 1


def multinomial_fit(Z, y):
    return MNLogit(np.asarray(y) - 1, sm.add_constant(Z)).fit(method='bfgs', disp=False, maxiter=500)


def multinomial_predict(res, Z):
    return np.argmax(res.predict(sm.add_constant(Z, has_constant='add')), axis=1) + 1


def gh_nodes(n):
    x, w = np.polynomial.hermite.hermgauss(n)
    return x, w


def clmm_nll(par, X, y, g, z, w, use_re):
    p = X.shape[1]
    theta = np.cumsum(np.r_[par[0], np.exp(par[1:K - 1])])
    beta = par[K - 1:K - 1 + p]
    eta = X @ beta
    if use_re:
        ls = min(max(par[-1], -12), 3)
        u = np.sqrt(2) * np.exp(ls) * z[None, :]; logw = np.log(w)[None, :] - 0.5 * np.log(np.pi)
    else:
        u = np.zeros((1, 1)); logw = np.zeros((1, 1))
    thU = np.r_[theta, np.inf]; thL = np.r_[-np.inf, theta]
    PU = expit(thU[y - 1][:, None] - eta[:, None] - u); PL = expit(thL[y - 1][:, None] - eta[:, None] - u)
    lp = np.log(np.maximum(PU - PL, 1e-300))
    ll = 0.0
    for i in np.unique(g):
        s = lp[g == i].sum(0) + logw[0]
        mx = s.max(); ll += mx + np.log(np.sum(np.exp(s - mx)))
    return -ll if np.isfinite(ll) else 1e10


def clmm_fit(X, y, g, n_gh, use_re, beta0, theta0):
    p = X.shape[1]
    par0 = np.r_[theta0[0], np.log(np.maximum(np.diff(theta0), 1e-3)), beta0]
    if use_re:
        par0 = np.r_[par0, np.log(0.5)]
    z, w = gh_nodes(n_gh)
    f = lambda par: clmm_nll(par, X, y, g, z, w, use_re)
    r = optimize.minimize(f, par0, method='BFGS', options=dict(gtol=1e-8, maxiter=2000))
    par = r.x if r.fun <= f(par0) else par0
    if use_re:
        par[-1] = min(max(par[-1], -12), 3)
    fit = dict(theta=np.cumsum(np.r_[par[0], np.exp(par[1:K - 1])]), beta=par[K - 1:K - 1 + p],
               sigma=float(np.exp(par[-1])) if use_re else 0.0, logLik=-f(par), use_re=use_re, z=z, w=w)
    return fit


def clmm_predict(fit, X):
    eta = X @ fit['beta']; th = np.r_[-np.inf, fit['theta'], np.inf]
    if fit['use_re']:
        u = np.sqrt(2) * fit['sigma'] * fit['z'][None, :]; wq = fit['w'] / np.sqrt(np.pi)
    else:
        u = np.zeros((1, 1)); wq = np.ones(1)
    P = np.zeros((X.shape[0], K))
    for k in range(K):
        Pk = expit(th[k + 1] - eta[:, None] - u) - expit(th[k] - eta[:, None] - u)
        P[:, k] = Pk @ wq
    return P


def lmm_formula(vn, interact):
    terms = list(vn)
    if interact and len(vn) > 1:
        terms += ['%s:%s' % (a, b) for a, b in combinations(vn, 2)]
    return 'PLI ~ ' + ' + '.join(terms)


def lmm_fit_predict(Ztr, ytr, gtr, Zte, vn, interact):
    """Linear mixed model with a random athlete intercept (REML); population-level prediction for new athletes."""
    Ttr = pd.DataFrame(Ztr, columns=vn); Ttr['PLI'] = ytr; Ttr['Ath'] = gtr
    md = smf.mixedlm(lmm_formula(vn, interact), Ttr, groups=Ttr['Ath']).fit(reml=True)
    Tte = pd.DataFrame(Zte, columns=vn)
    return md.predict(Tte).values, md


# ----------------------------------------------------------------------------- main
def main(cfg):
    t0 = time.time()
    rng = np.random.RandomState(cfg['seed'])
    os.makedirs(cfg['out_dir'], exist_ok=True)
    log = open(os.path.join(cfg['out_dir'], 'run_log.txt'), 'w')
    def say(*a):
        s = ' '.join(str(x) for x in a); print(s); log.write(s + '\n'); log.flush()
    say('MSOC pipeline (Python) | seed = %d | tuning evaluations = %d | quick = %s' % (cfg['seed'], cfg['bayes_evals'], cfg['quick']))

    # ---- 1 import and row-order checks
    R = pd.read_excel(cfg['data_file'], sheet_name=cfg['ml_sheet'])
    missing = [v for v in ML_VARS if v not in R.columns]
    assert not missing, 'missing columns in %s: %s' % (cfg['ml_sheet'], missing)
    R = R[ML_VARS].apply(pd.to_numeric, errors='coerce')
    ids = R['Ath_ID'].values; n_rows = len(R)
    per_session = int(np.argmax(ids[1:] == ids[0]) + 1)
    assert n_rows % per_session == 0 and np.array_equal(ids, np.tile(ids[:per_session], n_rows // per_session)), 'roster blocks differ'
    R['Session'] = np.arange(n_rows) // per_session + 1
    say('Sheet "%s": %d rows = %d sessions x %d rostered athletes' % (cfg['ml_sheet'], n_rows, n_rows // per_session, per_session))
    P = pd.read_excel(cfg['data_file'], sheet_name=cfg['polar_sheet'])
    assert len(P) == n_rows and np.array_equal(pd.to_numeric(P['Ath_ID'], errors='coerce').values, ids), 'ALL POLAR DATA not aligned'
    pliP = pd.to_numeric(P['P_Training_load_score'], errors='coerce').values
    both = ~np.isnan(pliP) & ~np.isnan(R['P_Training_load_score'].values)
    assert np.allclose(pliP[both], R['P_Training_load_score'].values[both]), 'PLI differs between sheets'
    R['DurMin'] = to_minutes(P['Duration'].values)
    zone = np.column_stack([to_minutes(P['Time_HR_zone_%d' % z].values) for z in range(1, 6)])
    R['TRIMP'] = zone @ np.arange(1, 6)
    R['Cardio'] = pd.to_numeric(P['P_Cardio_load'], errors='coerce').values

    # ---- 2 analytic sample and exclusion flow
    complete = R[ML_VARS].notna().all(1).values
    primary = complete & (R['Prim_V_Sec'].values == 1)
    # Diagnostic (log only): the completeness test requires all 11 columns; report whether it differs
    # from requiring only the seven analysis variables (the criterion as described in the manuscript).
    analysis_vars = ['sRPE', 'Well1_Fat', 'Well2_Stress', 'Well3_Sore', 'P_HR_avg_%', 'P_Tot_Dis', 'P_Training_load_score']
    complete_analysis = R[analysis_vars].notna().all(1).values
    say('Rows with the seven analysis variables present: %d; of these, %d lack Prim_V_Sec, Ath_ID, Prac_o_Gam or Return (excluded by the 11-column test).'
        % (complete_analysis.sum(), (complete_analysis & ~complete).sum()))
    flow = [('Rows in sheet (sessions x rostered athletes)', n_rows, len(np.unique(ids))),
            ('Complete rows (all analysis variables present)', int(complete.sum()), len(np.unique(ids[complete]))),
            ('Complete rows, primary athletes', int(primary.sum()), len(np.unique(ids[primary])))]
    keep = primary.copy()
    counts = pd.Series(ids[keep]).value_counts()
    low_ids = sorted(counts[counts < cfg['min_sessions_per_athlete']].index)
    keep &= ~np.isin(ids, low_ids)
    flow.append(('Excluded primary athletes with < %d complete sessions (ID %s)' % (cfg['min_sessions_per_athlete'], ', '.join(str(int(i)) for i in low_ids)),
                 int(keep.sum()), len(np.unique(ids[keep]))))
    if cfg['exclude_zero_pli']:
        zero = keep & (R['P_Training_load_score'].values == 0)
        keep &= ~zero
        flow.append(('Excluded sessions with PLI = 0 (export artifact; n = %d)' % zero.sum(), int(keep.sum()), len(np.unique(ids[keep]))))
        if zero.any():
            zp = R.loc[zero, ['Ath_ID', 'sRPE', 'P_HR_avg_%', 'P_Tot_Dis', 'P_Training_load_score']]
            say('\nExcluded PLI = 0 rows (athlete, sRPE, %HRmax, TD, PLI):'); say(zp.to_string(index=False))
            assert bool(((zp['sRPE'] > 0) & (zp['P_HR_avg_%'] > 0) & (zp['P_Tot_Dis'] > 0)).all()), \
                'A PLI = 0 row does not have non-zero sRPE, heart rate and distance; revise the exclusion rule.'
    flow_tbl = pd.DataFrame(flow, columns=['Step', 'Rows', 'Athletes']); say(flow_tbl.to_string(index=False))
    flow_tbl.to_csv(os.path.join(cfg['out_dir'], 'Table_S1_exclusion_flow.csv'), index=False)
    D = R[keep].reset_index(drop=True)
    n = len(D); ath = D['Ath_ID'].values.astype(int); ath_ids = np.unique(ath); n_ath = len(ath_ids)
    pli = D['P_Training_load_score'].values.astype(float)
    Xc = D[['sRPE', 'Well1_Fat', 'Well2_Stress', 'Well3_Sore', 'P_HR_avg_%', 'P_Tot_Dis']].values.astype(float)

    # ---- 3 PLI categories
    if cfg['cut_rule'] == 'meanHalfSD':
        ref = pli if cfg['cut_sample'] == 'analytic' else R['P_Training_load_score'].values[complete]
        cut = (ref.mean() - 0.5 * ref.std(ddof=1), ref.mean() + 0.5 * ref.std(ddof=1))
        note = 'mean +/- 0.5 SD of %s sample (n = %d; mean = %.2f, SD = %.2f)' % (cfg['cut_sample'], len(ref), ref.mean(), ref.std(ddof=1))
    else:
        cut = tuple(cfg['fixed_cut_points']); note = 'fixed values'
    ycat = pli_category(pli, cut)
    say('\nPLI cut points: %.2f and %.2f AU (%s)' % (cut[0], cut[1], note))
    say('Classes: ' + '  '.join('%s = %d (%.1f%%)' % (CLASS_NAMES[c - 1], (ycat == c).sum(), 100 * (ycat == c).mean()) for c in (1, 2, 3)))
    pd.DataFrame({'Boundary': ['Low/Medium', 'Medium/High'], 'PLI_AU': cut, 'Rule': note}).to_csv(os.path.join(cfg['out_dir'], 'Table_S0_PLI_cut_points.csv'), index=False)

    # ---- 3b sensitivity of the cut points to the held-out athlete (deterministic; nothing downstream uses it)
    cp_rows = []
    for a in ath_ids:
        v = pli[ath != a]; cp = (v.mean() - 0.5 * v.std(ddof=1), v.mean() + 0.5 * v.std(ddof=1))
        cat_f = pli_category(pli, cp)
        cp_rows.append([a, cp[0], cp[1], int((cat_f != ycat).sum()), int((cat_f[ath == a] != ycat[ath == a]).sum())])
    cp_tbl = pd.DataFrame(cp_rows, columns=['HeldOutAthlete', 'Cut_low_AU', 'Cut_high_AU', 'Sessions_changed_all', 'Sessions_changed_heldout'])
    say('\nCut-point sensitivity (each athlete omitted): low %.2f-%.2f AU, high %.2f-%.2f AU; held-out sessions whose category would change: %d of %d'
        % (cp_tbl.Cut_low_AU.min(), cp_tbl.Cut_low_AU.max(), cp_tbl.Cut_high_AU.min(), cp_tbl.Cut_high_AU.max(), cp_tbl.Sessions_changed_heldout.sum(), n))
    cp_tbl.to_csv(os.path.join(cfg['out_dir'], 'Table_S0b_cut_point_sensitivity.csv'), index=False)

    # ---- 4 descriptives
    descX = np.column_stack([Xc, pli])
    desc = pd.DataFrame({'Variable': FEAT_LABEL + ['PLI (AU)'], 'Mean': descX.mean(0), 'SD': descX.std(0, ddof=1), 'Min': descX.min(0), 'Max': descX.max(0)})
    say('\nAnalytic sample: %d athletes, %d athlete-sessions (%d practice, %d match)' % (n_ath, n, (D['Prac_o_Gam'] == 3).sum(), (D['Prac_o_Gam'] == 4).sum()))
    say(desc.to_string(index=False)); desc.to_csv(os.path.join(cfg['out_dir'], 'Table_1_descriptives.csv'), index=False)

    # ---- 5 Spearman and rmcorr
    rho, pv = stats.spearmanr(np.column_stack([pli, Xc]))
    rm = [rmcorr(Xc[:, j], pli, ath) for j in range(6)]
    corr = pd.DataFrame({'Variable': FEAT_LABEL, 'Spearman_rho': rho[1:, 0], 'Spearman_p': pv[1:, 0], 'Spearman_p_Holm': holm(pv[1:, 0]),
                         'rmcorr_r': [r[0] for r in rm], 'rmcorr_df': [r[2] for r in rm], 'rmcorr_p': [r[1] for r in rm], 'rmcorr_p_Holm': holm([r[1] for r in rm])})
    say('\nAssociations with continuous PLI:'); say(corr.to_string(index=False))
    corr.to_csv(os.path.join(cfg['out_dir'], 'Table_2_correlations_with_PLI.csv'), index=False)
    pd.DataFrame(rho, index=['PLI'] + FEAT_LABEL, columns=['PLI'] + FEAT_LABEL).to_csv(os.path.join(cfg['out_dir'], 'Table_S2_spearman_matrix.csv'))

    # ---- 6 PLI construct analysis
    C6 = pd.DataFrame({'PLI': pli, 'Dur': D['DurMin'].values, 'HR': Xc[:, 4], 'TRIMP': D['TRIMP'].values, 'TD': Xc[:, 5], 'sRPE': Xc[:, 0], 'Cardio': D['Cardio'].values})
    forms = ['PLI ~ Dur', 'PLI ~ HR', 'PLI ~ Dur*HR', 'PLI ~ Dur:HR', 'PLI ~ TRIMP', 'PLI ~ Cardio', 'TD ~ Dur', 'sRPE ~ Dur']   # Dur*HR = main effects + interaction; Dur:HR = product only
    constr = pd.DataFrame({'Model': forms, 'R2': [smf.ols(f, C6).fit().rsquared for f in forms]})
    say('\nPLI construct analysis:'); say(constr.to_string(index=False)); constr.to_csv(os.path.join(cfg['out_dir'], 'Table_S3_PLI_construct.csv'), index=False)

    # ---- 7 descriptive MRMR
    order, score = mrmr_miq(Xc, ycat, cfg['mrmr_bins'])
    mr = pd.DataFrame({'Rank': np.arange(1, 7), 'Variable': [FEAT_LABEL[i] for i in order], 'MRMR_score': score})
    say('\nDescriptive MRMR ranking (full sample; not used for models):'); say(mr.to_string(index=False))
    mr.to_csv(os.path.join(cfg['out_dir'], 'Table_S4_MRMR_full_sample.csv'), index=False)

    # ---- 8 leave-one-athlete-out cross-validation
    strategies = STRATEGIES[:1] if cfg['quick'] else STRATEGIES
    perf, percls, mcn, sel_rows, hp_rows, cont_rows, pred_all, cont_all, sel_all = [], [], [], [], [], [], {}, {}, {}
    for s, st in enumerate(strategies, 1):
        say('\n=== LOAO strategy %d/%d: %s ===' % (s, len(strategies), st['name']))
        pred = np.full((n, len(MODEL_NAMES)), np.nan); pred_cont = np.full((n, 2), np.nan); sel_log = np.zeros((n_ath, 6), bool)
        for f, a in enumerate(ath_ids, 1):
            fold_seed = cfg['seed'] + 100 * s + f + st['seed_offset']
            te = ath == a; tr = ~te
            if st['mode'] == 'mrmr':
                idx, _ = mrmr_miq(Xc[tr], ycat[tr], cfg['mrmr_bins']); sel = np.sort(idx[:st['k']])
            else:
                sel = np.array(st['features'])
            sel_log[f - 1, sel] = True
            Xtr, Xte, ytr, gtr = Xc[tr][:, sel], Xc[te][:, sel], ycat[tr], ath[tr]
            mu, sd = Xtr.mean(0), Xtr.std(0, ddof=1); sd[sd == 0] = 1
            Ztr, Zte = (Xtr - mu) / sd, (Xte - mu) / sd
            # 1 SVM
            m, hp, L = tune_and_fit('SVM', Xtr, ytr, gtr, cfg, fold_seed); pred[te, 0] = m.predict(Xte)
            hp_rows.append([st['name'], f, a, 'SVM', '; '.join('%s=%.4g' % kv for kv in hp.items()), L])
            # 2 AdaBoost
            m, hp, L = tune_and_fit('ADA', Xtr, ytr, gtr, cfg, fold_seed); pred[te, 1] = m.predict(Xte)
            hp_rows.append([st['name'], f, a, 'AdaBoostM2', '; '.join('%s=%.4g' % kv for kv in hp.items()), L])
            # 3 ordinal
            om = ordinal_fit(Ztr, ytr); pred[te, 2] = ordinal_predict(om, Zte)
            # 4 multinomial
            mm = multinomial_fit(Ztr, ytr); pred[te, 3] = multinomial_predict(mm, Zte)
            # 5 CLMM (marginal prediction for the unseen athlete)
            pp = Ztr.shape[1]; prm = np.asarray(om.params); beta0 = prm[:pp]
            theta0 = np.cumsum(np.r_[prm[pp], np.exp(prm[pp + 1:])])   # statsmodels stores increments as logs
            _, gi = np.unique(gtr, return_inverse=True)
            fc = clmm_fit(Ztr, ytr, gi, cfg['n_gh'], True, beta0, theta0); pred[te, 4] = np.argmax(clmm_predict(fc, Zte), 1) + 1
            # 6 LMM on continuous PLI
            vn = [FEAT_VAR[i] for i in sel]
            pred_cont[te, 0], _ = lmm_fit_predict(Ztr, pli[tr], gtr, Zte, vn, cfg['lmm_interact']); pred[te, 5] = pli_category(pred_cont[te, 0], cut)
            # 7 SVR on z-scored PLI
            muY, sdY = pli[tr].mean(), pli[tr].std(ddof=1)
            m, hp, L = tune_and_fit('SVR', Xtr, (pli[tr] - muY) / sdY, gtr, cfg, fold_seed)
            pred_cont[te, 1] = muY + sdY * m.predict(Xte); pred[te, 6] = pli_category(pred_cont[te, 1], cut)
            hp_rows.append([st['name'], f, a, 'SVR', '; '.join('%s=%.4g' % kv for kv in hp.items()), L])
            sel_rows.append([st['name'], f, a, ', '.join(FEAT_LABEL[i] for i in sel)])
            say('  fold %2d/%d  athlete %2d  n = %2d  predictors: %-45s acc: %s' % (f, n_ath, a, te.sum(), ', '.join(FEAT_LABEL[i] for i in sel),
                ' '.join('%.2f' % np.mean(pred[te, m_] == ycat[te]) for m_ in range(len(MODEL_NAMES)))))
        pred = pred.astype(int); pred_all[s] = pred; cont_all[s] = pred_cont.copy(); sel_all[s] = sel_log.copy()
        M = [class_metrics(ycat, pred[:, m_], ath) for m_ in range(len(MODEL_NAMES))]
        for m_, Mm in enumerate(M):
            perf.append([st['name'], MODEL_NAMES[m_], 100 * Mm['accMean'], 100 * Mm['accSD'], 100 * Mm['acc'], 100 * Mm['balAcc'], Mm['macroF1'], Mm['extremeErr']])
            for c in range(K):
                percls.append([st['name'], MODEL_NAMES[m_], CLASS_NAMES[c], Mm['sens'][c], Mm['spec'][c], Mm['prec'][c], Mm['f1'][c]])
        for m_ in range(2):
            cont_rows.append([st['name'], MODEL_NAMES[5 + m_], *reg_metrics(pli, pred_cont[:, m_])])
        for a_, b_ in combinations(range(len(MODEL_NAMES)), 2):
            mcn.append([st['name'], MODEL_NAMES[a_], MODEL_NAMES[b_], *mcnemar_exact(ycat, pred[:, a_], pred[:, b_])])
        say('  Selection frequency: ' + '; '.join('%s %d/%d' % (FEAT_LABEL[j], sel_log[:, j].sum(), n_ath) for j in range(6)))
    cols = ['Strategy', 'Model', 'Acc_mean_across_athletes_pct', 'Acc_SD_across_athletes_pct', 'Acc_pooled_pct', 'Balanced_acc_pct', 'Macro_F1', 'Low_High_confusions']
    perf_tbl = pd.DataFrame(perf, columns=cols); say('\n=== LOAO performance (Table 3) ==='); say(perf_tbl.to_string(index=False))
    perf_tbl.to_csv(os.path.join(cfg['out_dir'], 'Table_3_LOAO_performance.csv'), index=False)
    pd.DataFrame(percls, columns=['Strategy', 'Model', 'Class', 'Sensitivity', 'Specificity', 'Precision', 'F1']).to_csv(os.path.join(cfg['out_dir'], 'Table_S5_LOAO_per_class.csv'), index=False)
    pd.DataFrame(cont_rows, columns=['Strategy', 'Model', 'RMSE_AU', 'MAE_AU', 'R2']).to_csv(os.path.join(cfg['out_dir'], 'Table_S6_LOAO_continuous_PLI.csv'), index=False)
    pd.DataFrame(mcn, columns=['Strategy', 'ModelA', 'ModelB', 'A_right_B_wrong', 'A_wrong_B_right', 'Exact_McNemar_p']).to_csv(os.path.join(cfg['out_dir'], 'Table_S7_McNemar.csv'), index=False)
    freq = pd.DataFrame({'Variable': FEAT_LABEL})
    for s in sel_all:
        freq['S%d_folds_selected' % s] = sel_all[s].sum(0)
    freq.to_csv(os.path.join(cfg['out_dir'], 'Table_S8_selection_frequency.csv'), index=False)
    pd.DataFrame(sel_rows, columns=['Strategy', 'Fold', 'HeldOutAthlete', 'Predictors']).to_csv(os.path.join(cfg['out_dir'], 'Table_S9_selected_predictors_by_fold.csv'), index=False)
    pd.DataFrame(hp_rows, columns=['Strategy', 'Fold', 'HeldOutAthlete', 'Model', 'Hyperparameters', 'Inner_LOAO_loss']).to_csv(os.path.join(cfg['out_dir'], 'Table_S10_hyperparameters_by_fold.csv'), index=False)
    pt = pd.DataFrame({'Athlete': ath, 'Session': D['Session'].values, 'PLI': pli, 'TrueClass': ycat})
    for s, pred in pred_all.items():
        for m_, name in enumerate(MODEL_NAMES):
            pt['S%d_%s' % (s, name)] = pred[:, m_]
    pt.to_csv(os.path.join(cfg['out_dir'], 'Table_S11_session_predictions.csv'), index=False)

    # ---- 8b functional-form sensitivity: LMM with main effects only, same folds and predictors as strategy 1
    pred_add = np.full(n, np.nan); pred_add_cont = np.full(n, np.nan)
    for f, a in enumerate(ath_ids, 1):
        te = ath == a; tr = ~te
        sel = np.flatnonzero(sel_all[1][f - 1])
        Xtr, Xte = Xc[tr][:, sel], Xc[te][:, sel]
        mu, sd = Xtr.mean(0), Xtr.std(0, ddof=1); sd[sd == 0] = 1
        Ztr, Zte = (Xtr - mu) / sd, (Xte - mu) / sd
        vn = [FEAT_VAR[i] for i in sel]
        pred_add_cont[te], _ = lmm_fit_predict(Ztr, pli[tr], ath[tr], Zte, vn, False)
        pred_add[te] = pli_category(pred_add_cont[te], cut)
    pred_add = pred_add.astype(int)
    Madd = class_metrics(ycat, pred_add, ath); Mint = class_metrics(ycat, pred_all[1][:, 5], ath)
    rm_a, ma_a, r2_a = reg_metrics(pli, pred_add_cont); rm_i, ma_i, r2_i = reg_metrics(pli, cont_all[1][:, 0])
    nb_a, nc_a, p_a = mcnemar_exact(ycat, pred_all[1][:, 5], pred_add)
    add_tbl = pd.DataFrame([['LMM_interactions', 100 * Mint['accMean'], 100 * Mint['accSD'], 100 * Mint['acc'], 100 * Mint['balAcc'], Mint['macroF1'], Mint['extremeErr'], rm_i, ma_i, r2_i, nb_a, nc_a, p_a],
                            ['LMM_additive', 100 * Madd['accMean'], 100 * Madd['accSD'], 100 * Madd['acc'], 100 * Madd['balAcc'], Madd['macroF1'], Madd['extremeErr'], rm_a, ma_a, r2_a, nb_a, nc_a, p_a]],
                           columns=['Model', 'Acc_mean_across_athletes_pct', 'Acc_SD_across_athletes_pct', 'Acc_pooled_pct', 'Balanced_acc_pct', 'Macro_F1', 'Low_High_confusions',
                                    'RMSE_AU', 'MAE_AU', 'R2', 'McNemar_interactions_right_additive_wrong', 'McNemar_interactions_wrong_additive_right', 'McNemar_p'])
    say('\n=== Functional-form sensitivity: LMM with vs without interaction terms (primary strategy) ==='); say(add_tbl.to_string(index=False))
    add_tbl.to_csv(os.path.join(cfg['out_dir'], 'Table_S6b_LMM_functional_form.csv'), index=False)

    # ---- 8c distance of misclassified sessions from the nearest cut point (primary strategy)
    bnd = []
    for m_, name in enumerate(MODEL_NAMES):
        mis = pred_all[1][:, m_] != ycat
        dist = np.minimum(np.abs(pli[mis] - cut[0]), np.abs(pli[mis] - cut[1]))
        bnd.append([name, int(mis.sum()), int((dist <= 20).sum()), 100 * float(np.mean(dist <= 20)) if mis.any() else np.nan,
                    float(np.median(dist)) if mis.any() else np.nan, float(dist.max()) if mis.any() else np.nan])
    bnd_tbl = pd.DataFrame(bnd, columns=['Model', 'Misclassified', 'Within_20_AU_of_cut_point', 'Pct_within_20_AU', 'Median_distance_AU', 'Max_distance_AU'])
    say('\n=== Misclassified sessions: distance to the nearest cut point (primary strategy) ==='); say(bnd_tbl.to_string(index=False))
    bnd_tbl.to_csv(os.path.join(cfg['out_dir'], 'Table_S16_boundary_distance.csv'), index=False)

    # ---- 8d cluster bootstrap (training days; athletes) and sign-flip tests for pairwise accuracy differences
    brng = np.random.default_rng(cfg['seed'] + 777); n_boot, n_perm = 10000, 20000
    cb = []
    for s, pred in pred_all.items():
        for a_, b_ in combinations(range(len(MODEL_NAMES)), 2):
            cA = pred[:, a_] == ycat; cB = pred[:, b_] == ycat
            ciD, pD = cluster_compare(cA, cB, D['Session'].values, n_boot, n_perm, brng)
            ciA, pA = cluster_compare(cA, cB, ath, n_boot, n_perm, brng)
            cb.append([strategies[s - 1]['name'], MODEL_NAMES[a_], MODEL_NAMES[b_], 100 * (cA.mean() - cB.mean()), ciD[0], ciD[1], pD, ciA[0], ciA[1], pA])
    cb_tbl = pd.DataFrame(cb, columns=['Strategy', 'ModelA', 'ModelB', 'Diff_pooled_acc_pct', 'Day_boot_CI_low', 'Day_boot_CI_high', 'Day_signflip_p',
                                       'Athlete_boot_CI_low', 'Athlete_boot_CI_high', 'Athlete_signflip_p'])
    say('\n=== Pairwise accuracy differences: cluster bootstrap (days, athletes) and sign-flip tests ==='); say(cb_tbl.round(3).to_string(index=False))
    cb_tbl.to_csv(os.path.join(cfg['out_dir'], 'Table_S7b_cluster_bootstrap.csv'), index=False)

    # ---- 8e inner-split MRMR agreement with the outer-fold selection (deterministic)
    inner_rows = []; agree = 0
    for f, a in enumerate(ath_ids, 1):
        tr = ath != a; sel = np.flatnonzero(sel_all[1][f - 1])
        for j in ath_ids[ath_ids != a]:
            inner = tr & (ath != j)
            idx, _ = mrmr_miq(Xc[inner], ycat[inner], cfg['mrmr_bins']); sel_j = np.sort(idx[:3])
            same = bool(np.array_equal(sel_j, sel)); agree += same
            inner_rows.append([f, a, j, ', '.join(FEAT_LABEL[i] for i in sel_j), same])
    say('\nInner-split MRMR agreement with the outer-fold selection: %d of %d inner splits' % (agree, len(inner_rows)))
    pd.DataFrame(inner_rows, columns=['OuterFold', 'HeldOutAthlete', 'InnerValidationAthlete', 'InnerMRMR_top3', 'SameAsOuterFold']).to_csv(
        os.path.join(cfg['out_dir'], 'Table_S9b_inner_MRMR_agreement.csv'), index=False)

    # ---- 8f ordinal model with all six candidate predictors under LOAO (deterministic)
    pred_aug = np.full(n, np.nan)
    for a in ath_ids:
        te = ath == a; tr = ~te
        mu, sd = Xc[tr].mean(0), Xc[tr].std(0, ddof=1); sd[sd == 0] = 1
        om6 = ordinal_fit((Xc[tr] - mu) / sd, ycat[tr]); pred_aug[te] = ordinal_predict(om6, (Xc[te] - mu) / sd)
    pred_aug = pred_aug.astype(int)
    Maug = class_metrics(ycat, pred_aug, ath); M3 = class_metrics(ycat, pred_all[1][:, 2], ath)
    nbG, ncG, pG = mcnemar_exact(ycat, pred_all[1][:, 2], pred_aug)
    ciG, pGd = cluster_compare(pred_all[1][:, 2] == ycat, pred_aug == ycat, D['Session'].values, n_boot, n_perm, brng)
    aug_tbl = pd.DataFrame([['Ordinal_3_predictors', 100 * M3['accMean'], 100 * M3['accSD'], 100 * M3['acc'], 100 * M3['balAcc'], M3['macroF1'], M3['extremeErr'], nbG, ncG, pG, ciG[0], ciG[1], pGd],
                            ['Ordinal_6_predictors', 100 * Maug['accMean'], 100 * Maug['accSD'], 100 * Maug['acc'], 100 * Maug['balAcc'], Maug['macroF1'], Maug['extremeErr'], nbG, ncG, pG, ciG[0], ciG[1], pGd]],
                           columns=['Model', 'Acc_mean_across_athletes_pct', 'Acc_SD_across_athletes_pct', 'Acc_pooled_pct', 'Balanced_acc_pct', 'Macro_F1', 'Low_High_confusions',
                                    'McNemar_3_right_6_wrong', 'McNemar_3_wrong_6_right', 'McNemar_p', 'Day_boot_CI_low', 'Day_boot_CI_high', 'Day_signflip_p'])
    say('\n=== Ordinal model with three vs all six candidate predictors (LOAO) ==='); say(aug_tbl.round(3).to_string(index=False))
    aug_tbl.to_csv(os.path.join(cfg['out_dir'], 'Table_S17_ordinal_all_six_predictors.csv'), index=False)

    # ---- 8g leave-one-day-out evaluation of the deterministic models (unseen days, seen athletes)
    days = np.unique(D['Session'].values); lodo_names = ['OrdinalLogistic', 'Multinomial', 'CLMM', 'LMM_binned']
    pred_lodo = np.full((n, 4), np.nan); pred_lodo_cont = np.full(n, np.nan); sel_lodo = np.zeros((len(days), 6), bool)
    for di, d in enumerate(days):
        te = D['Session'].values == d; tr = ~te
        idx, _ = mrmr_miq(Xc[tr], ycat[tr], cfg['mrmr_bins']); sel = np.sort(idx[:3]); sel_lodo[di, sel] = True
        Xtr, Xte = Xc[tr][:, sel], Xc[te][:, sel]
        mu, sd = Xtr.mean(0), Xtr.std(0, ddof=1); sd[sd == 0] = 1
        Ztr, Zte = (Xtr - mu) / sd, (Xte - mu) / sd
        om = ordinal_fit(Ztr, ycat[tr]); pred_lodo[te, 0] = ordinal_predict(om, Zte)
        mm = multinomial_fit(Ztr, ycat[tr]); pred_lodo[te, 1] = multinomial_predict(mm, Zte)
        pp = Ztr.shape[1]; prm = np.asarray(om.params); beta0 = prm[:pp]; theta0 = np.cumsum(np.r_[prm[pp], np.exp(prm[pp + 1:])])
        _, gi = np.unique(ath[tr], return_inverse=True)
        fc = clmm_fit(Ztr, ycat[tr], gi, cfg['n_gh'], True, beta0, theta0); pred_lodo[te, 2] = np.argmax(clmm_predict(fc, Zte), 1) + 1
        vn = [FEAT_VAR[i] for i in sel]
        pred_lodo_cont[te], _ = lmm_fit_predict(Ztr, pli[tr], ath[tr], Zte, vn, cfg['lmm_interact']); pred_lodo[te, 3] = pli_category(pred_lodo_cont[te], cut)
    pred_lodo = pred_lodo.astype(int)
    lodo = []
    for m_, name in enumerate(lodo_names):
        Mm = class_metrics(ycat, pred_lodo[:, m_], D['Session'].values)      # accMean/SD across DAYS
        lodo.append([name, 100 * Mm['acc'], 100 * Mm['accMean'], 100 * Mm['accSD'], 100 * Mm['balAcc'], Mm['macroF1'], Mm['extremeErr']])
    lodo_tbl = pd.DataFrame(lodo, columns=['Model', 'Acc_pooled_pct', 'Acc_mean_across_days_pct', 'Acc_SD_across_days_pct', 'Balanced_acc_pct', 'Macro_F1', 'Low_High_confusions'])
    rmL, maL, r2L = reg_metrics(pli, pred_lodo_cont)
    lodo_tbl['RMSE_AU'] = [np.nan] * 3 + [rmL]; lodo_tbl['MAE_AU'] = [np.nan] * 3 + [maL]; lodo_tbl['R2'] = [np.nan] * 3 + [r2L]
    say('\n=== Leave-one-day-out (%d days), deterministic models ===' % len(days)); say(lodo_tbl.round(3).to_string(index=False))
    say('Leave-one-day-out MRMR selection frequency (days): ' + '; '.join('%s %d/%d' % (FEAT_LABEL[j], sel_lodo[:, j].sum(), len(days)) for j in range(6)))
    lodo_tbl.to_csv(os.path.join(cfg['out_dir'], 'Table_S18_leave_one_day_out.csv'), index=False)

    # ---- 9 inference models on the full sample (sRPE, %maxHR, TD)
    inf = [0, 4, 5]; Zall = (Xc[:, inf] - Xc[:, inf].mean(0)) / Xc[:, inf].std(0, ddof=1); names = [FEAT_LABEL[i] for i in inf]
    om = ordinal_fit(Zall, ycat); mm = multinomial_fit(Zall, ycat)
    pp = len(inf); prm = np.asarray(om.params); beta = prm[:pp]; theta = np.cumsum(np.r_[prm[pp], np.exp(prm[pp + 1:])])
    # Standard errors of the cumulative thresholds by the delta method (statsmodels parameterises the
    # second threshold as theta_1 + exp(t)).
    V = np.asarray(om.cov_params()); Jt = np.zeros((K - 1, len(prm))); Jt[0, pp] = 1; Jt[1, pp] = 1; Jt[1, pp + 1] = np.exp(prm[pp + 1])
    se_theta = np.sqrt(np.diag(Jt @ V @ Jt.T)); p_theta = 2 * stats.norm.sf(np.abs(theta / se_theta))
    ord_tbl = pd.DataFrame({'Term': ['theta_1 (Low|Medium)', 'theta_2 (Medium|High)'] + names, 'Estimate': np.r_[theta, beta],
                            'SE': np.r_[se_theta, np.asarray(om.bse)[:pp]], 'p': np.r_[p_theta, np.asarray(om.pvalues)[:pp]]})
    say('\nOrdinal logistic regression (z-scored predictors; beta > 0 = higher category):'); say(ord_tbl.to_string(index=False))
    # Brant test of proportional odds (as in the MATLAB script, Section 9a). The likelihood-ratio test against
    # the non-parallel-slopes cumulative-logit model is printed for information only (that model can give
    # negative fitted probabilities, so the test is not valid).
    fnp = cumlogit_np_fit(Zall, ycat, beta, theta)
    say('(Information only, not reported) LR test against the non-parallel-slopes cumulative-logit model: LR = %.2f; converged %s, max |gradient| %.2e; sessions with a negative fitted probability: %d'
        % (2 * (fnp['logLik'] - om.llf), fnp['converged'], fnp['max_grad'], fnp['n_negative']))
    W_obs, d_obs, Vd_obs, B_bin, SE_bin = brant_wald(Zall, ycat)
    bin_tbl = pd.DataFrame({'Predictor': names, 'Ordinal_beta': beta, 'Beta_Y_gt_Low': B_bin[1:, 0], 'SE_Y_gt_Low': SE_bin[1:, 0],
                            'Beta_Y_gt_Medium': B_bin[1:, 1], 'SE_Y_gt_Medium': SE_bin[1:, 1]})
    say('Binary logistic regressions at each cut point (Brant test):'); say(bin_tbl.round(4).to_string(index=False))
    n_boot_po = 200 if cfg['quick'] else 2000
    prng = np.random.default_rng(cfg['seed'])
    Cfit = expit(theta[None, :] - (Zall @ beta)[:, None])               # P(Y <= k), k = 1..K-1
    Wb = []
    for b_ in range(n_boot_po):
        ys = 1 + (prng.random(len(ycat))[:, None] > Cfit).sum(1)
        if len(np.unique(ys)) < K:
            continue
        Wq, dq, Vq, _, _ = brant_wald(Zall, ys); Wb.append(np.r_[Wq, dq ** 2 / np.diag(Vq)])
    Wb = np.array(Wb)
    po_stat = np.r_[W_obs, d_obs ** 2 / np.diag(Vd_obs)]; po_df = np.r_[(K - 2) * len(inf), np.ones(len(inf))]
    brant_tbl = pd.DataFrame({'Test': ['Omnibus (all predictors)'] + names, 'Wald_chi2': po_stat, 'df': po_df.astype(int),
                              'p_asymptotic': stats.chi2.sf(po_stat, po_df), 'p_bootstrap': (1 + (Wb >= po_stat).sum(0)) / (len(Wb) + 1)})
    say('Brant test of proportional odds (H0: equal slopes at both cut points; %d bootstrap data sets):' % len(Wb)); say(brant_tbl.round(4).to_string(index=False))
    brant_tbl.to_csv(os.path.join(cfg['out_dir'], 'Table_S12b_brant_test.csv'), index=False)
    ord_tbl.to_csv(os.path.join(cfg['out_dir'], 'Table_S12_ordinal_logistic.csv'), index=False)
    _, g_all = np.unique(ath, return_inverse=True)
    f0 = clmm_fit(Zall, ycat, g_all, cfg['n_gh'], False, beta, theta); f1 = clmm_fit(Zall, ycat, g_all, cfg['n_gh'], True, beta, theta)
    say('Check: custom ordinal log-likelihood %.3f vs statsmodels %.3f' % (f0['logLik'], om.llf))
    re_lr = max(0.0, 2 * (f1['logLik'] - f0['logLik'])); re_p = 0.5 * (1 - stats.chi2.cdf(re_lr, 1))
    # Standard errors from the numerical Hessian of the CLMM negative log-likelihood (delta method for the
    # cumulative thresholds and for sigma = exp(log-sigma)); the variance component's p is the LR test.
    z_gh, w_gh = gh_nodes(cfg['n_gh'])
    par1 = np.r_[f1['theta'][0], np.log(np.maximum(np.diff(f1['theta']), 1e-8)), f1['beta'], np.log(max(f1['sigma'], 1e-8))]
    H1 = num_hessian(lambda par: clmm_nll(par, Zall, ycat, g_all, z_gh, w_gh, True), par1)
    try:
        V1 = np.linalg.inv(H1)
    except np.linalg.LinAlgError:
        V1 = np.linalg.pinv(H1)
    Jc = np.zeros((K - 1 + pp + 1, len(par1))); Jc[0, 0] = 1; Jc[1, 0] = 1; Jc[1, 1] = np.exp(par1[1])
    for i in range(pp):
        Jc[K - 1 + i, K - 1 + i] = 1
    Jc[-1, -1] = f1['sigma']
    se1 = np.sqrt(np.maximum(np.diag(Jc @ V1 @ Jc.T), 0))
    est1 = np.r_[f1['theta'], f1['beta'], f1['sigma']]
    z1 = est1 / np.where(se1 > 0, se1, np.nan); pv1 = 2 * stats.norm.sf(np.abs(z1)); pv1[-1] = re_p
    clmm_tbl = pd.DataFrame({'Term': ['theta_1 (Low|Medium)', 'theta_2 (Medium|High)'] + names + ['SD athlete intercept'], 'Estimate': est1, 'SE': se1, 'z': z1, 'p': pv1})
    say('\nOrdinal mixed model (random athlete intercept):'); say(clmm_tbl.to_string(index=False))
    say('Random-intercept LR test: LR = %.2f, p = %.3f (boundary-corrected)' % (re_lr, re_p))
    clmm_tbl.to_csv(os.path.join(cfg['out_dir'], 'Table_S13_ordinal_mixed_model.csv'), index=False)
    # LMM with crossed athlete and session random intercepts (variance components with a single group)
    T = pd.DataFrame(Zall, columns=[FEAT_VAR[i] for i in inf]); T['PLI'] = pli; T['Ath'] = ath; T['Session'] = D['Session'].values; T['one'] = 1
    vc = {'Ath': '0 + C(Ath)', 'Session': '0 + C(Session)'}
    lme_null = smf.mixedlm('PLI ~ 1', T, groups='one', vc_formula=vc).fit(reml=True)
    lme_full = smf.mixedlm(lmm_formula([FEAT_VAR[i] for i in inf], cfg['lmm_interact']), T, groups='one', vc_formula=vc).fit(reml=True)
    def icc(m):
        va, vs, ve = float(m.vcomp[0]), float(m.vcomp[1]), float(m.scale)
        return va, vs, ve, va / (va + vs + ve), vs / (va + vs + ve)
    icc_tbl = pd.DataFrame([['Null (intercept only)', *icc(lme_null)], ['With predictors', *icc(lme_full)]],
                           columns=['Model', 'Var_athlete', 'Var_session', 'Var_residual', 'ICC_athlete', 'ICC_session'])
    say('\nLinear mixed model, continuous PLI (crossed athlete and session intercepts):'); say(str(lme_full.summary())); say(icc_tbl.to_string(index=False))
    icc_tbl.to_csv(os.path.join(cfg['out_dir'], 'Table_S14_LMM_variance_components.csv'), index=False)
    fe = pd.DataFrame({'Name': lme_full.fe_params.index, 'Estimate': lme_full.fe_params.values, 'SE': lme_full.bse_fe.values, 'pValue': lme_full.pvalues[:len(lme_full.fe_params)].values})
    fe.to_csv(os.path.join(cfg['out_dir'], 'Table_S15_LMM_fixed_effects.csv'), index=False)
    json.dump({k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()}, open(os.path.join(cfg['out_dir'], 'config.json'), 'w'), indent=1)
    import sklearn, statsmodels, skopt, scipy
    with open(os.path.join(cfg['out_dir'], 'software_versions.txt'), 'w') as fh:
        fh.write('Python %s (%s)\nnumpy %s\npandas %s\nscipy %s\nscikit-learn %s\nstatsmodels %s\nscikit-optimize %s\n'
                 % (platform.python_version(), platform.platform(), np.__version__, pd.__version__, scipy.__version__, sklearn.__version__, statsmodels.__version__, skopt.__version__))
    say('\nDone in %.1f min. Outputs in %s' % ((time.time() - t0) / 60, cfg['out_dir']))
    log.close()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=CFG['data_file']); ap.add_argument('--out', default=CFG['out_dir'])
    ap.add_argument('--quick', action='store_true', help='dry run: 12 tuning evaluations, primary strategy only')
    ap.add_argument('--evals', type=int, default=None)
    a = ap.parse_args()
    cfg = dict(CFG); cfg['data_file'] = a.data; cfg['out_dir'] = a.out; cfg['quick'] = a.quick
    if a.quick:
        cfg['bayes_evals'] = 12
    if a.evals:
        cfg['bayes_evals'] = a.evals
    main(cfg)
