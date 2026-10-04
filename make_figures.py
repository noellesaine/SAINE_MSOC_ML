# -*- coding: utf-8 -*-
"""Regenerate manuscript Figures 2-6 and Supplemental Figures S1-S5 from the pipeline outputs
(MSOC_results.mat and Table_S11_session_predictions.csv), with labels matching the manuscript.
Requires the outputs of a MATLAB run in ./results (it cannot run from the Python pipeline's CSVs alone).
Requirements: numpy, pandas, scipy, matplotlib."""
import numpy as np, pandas as pd, scipy.io as sio, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures'); SUPP = FIG; os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11})

m = sio.loadmat(os.path.join(RES, 'MSOC_results.mat'), squeeze_me=True, struct_as_record=False)
pred = pd.read_csv(os.path.join(RES, 'Table_S11_session_predictions.csv'))
ytrue = pred.TrueClass.values
labels = ['PLI', 'sRPE', 'SRWM-Fatigue', 'SRWM-Stress', 'SRWM-Soreness', '%maxHR', 'TD']
classes = ['Low', 'Medium', 'High']
cuts = np.array(m['cfg'].cutPoints, dtype=float)

def stars(p):
    return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''

# ---------------------------------------------------------------- Figure 2
rho, pv = m['rhoAll'], m['pAll']
n = len(labels)
fig, ax = plt.subplots(figsize=(8, 7), dpi=300)
mask = np.tril(np.ones((n, n), bool))
data = np.where(mask, rho, np.nan)
cmap = LinearSegmentedColormap.from_list('gray_div', ['#1a1a1a', '#ffffff'])
im = ax.imshow(data, cmap='gray', vmin=-1, vmax=1)
for i in range(n):
    for j in range(i + 1):
        s = '' if i == j else stars(pv[i, j])
        ax.text(j, i, '%.2f%s' % (rho[i, j], s), ha='center', va='center', fontsize=10.5,
                color='white' if rho[i, j] < -0.2 else 'black')
ax.set_xticks(range(n)); ax.set_yticks(range(n))
ax.set_xticklabels(labels, rotation=45, ha='right'); ax.set_yticklabels(labels)
ax.tick_params(length=0)
for sp in ax.spines.values():
    sp.set_visible(False)
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03); cb.set_label('Spearman ρ')
ax.set_title('Spearman rank correlations (n = 372 athlete-sessions)', fontsize=12)
fig.savefig(os.path.join(FIG, 'Figure2_Spearman_matrix.png'), dpi=300, facecolor='white', bbox_inches='tight'); plt.close(fig)

# ---------------------------------------------------------------- Figure 3
mr = pd.read_csv(os.path.join(RES, 'Table_S4_MRMR_full_sample.csv'))
name_map = {'Total distance': 'TD', '%HRmax': '%maxHR', 'sRPE': 'sRPE', 'Wellness-Fatigue': 'SRWM-Fatigue',
            'Wellness-Stress': 'SRWM-Stress', 'Wellness-Soreness': 'SRWM-Soreness'}
fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
ax.bar(range(len(mr)), mr.MRMR_score, color=['#4d4d4d'] + ['#9e9e9e'] * (len(mr) - 1), edgecolor='black')
ax.set_xticks(range(len(mr))); ax.set_xticklabels([name_map[v] for v in mr.Variable], rotation=30, ha='right')
ax.set_ylabel('MRMR score'); ax.grid(axis='y', alpha=0.3)
ax.set_title('MRMR ranking, full analytic sample (descriptive)', fontsize=12)
ax.text(0.98, 0.95, 'rank 1: relevance, I(x; y) in nats\nranks 2–6: mutual-information quotient',
        transform=ax.transAxes, ha='right', va='top', fontsize=9)
for sp in ('top', 'right'):
    ax.spines[sp].set_visible(False)
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'Figure3_MRMR_ranking.png'), dpi=300, facecolor='white'); plt.close(fig)

# ---------------------------------------------------------------- confusion matrices
def confusion(ypred, title, path):
    C = np.zeros((3, 3), int)
    for t, p in zip(ytrue, ypred):
        C[t - 1, p - 1] += 1
    sens = np.diag(C) / C.sum(1); prec = np.diag(C) / C.sum(0)
    fig = plt.figure(figsize=(7.2, 6.2), dpi=300)
    gs = fig.add_gridspec(2, 2, width_ratios=[3, 0.7], height_ratios=[3, 0.7], wspace=0.06, hspace=0.06)
    ax = fig.add_subplot(gs[0, 0]); axr = fig.add_subplot(gs[0, 1]); axb = fig.add_subplot(gs[1, 0])
    blues = LinearSegmentedColormap.from_list('b', ['#ffffff', '#1f5fa8'])
    ax.imshow(C, cmap=blues, vmin=0, vmax=C.max())
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(C[i, j]), ha='center', va='center', fontsize=13,
                    color='white' if C[i, j] > 0.6 * C.max() else 'black')
    ax.set_xticks(range(3)); ax.set_yticks(range(3)); ax.set_xticklabels([]); ax.set_yticklabels(classes)
    ax.set_ylabel('True class'); ax.tick_params(length=0)
    ax.set_title(title, fontsize=12)
    # right: sensitivity
    axr.imshow(sens[:, None], cmap=blues, vmin=0, vmax=1)
    for i in range(3):
        axr.text(0, i, '%.1f%%' % (100 * sens[i]), ha='center', va='center', fontsize=11,
                 color='white' if sens[i] > 0.6 else 'black')
    axr.set_xticks([0]); axr.set_xticklabels(['Sensitivity'], fontsize=10); axr.xaxis.tick_top(); axr.set_yticks([]); axr.tick_params(length=0)
    # bottom: precision
    axb.imshow(prec[None, :], cmap=blues, vmin=0, vmax=1)
    for j in range(3):
        axb.text(j, 0, '%.1f%%' % (100 * prec[j]), ha='center', va='center', fontsize=11,
                 color='white' if prec[j] > 0.6 else 'black')
    axb.set_xticks(range(3)); axb.set_xticklabels(classes); axb.set_yticks([0]); axb.set_yticklabels(['Precision'], fontsize=10)
    axb.set_xlabel('Predicted class'); axb.tick_params(length=0)
    for a in (ax, axr, axb):
        for sp in a.spines.values():
            sp.set_edgecolor('#444444')
    fig.savefig(path, dpi=300, facecolor='white', bbox_inches='tight'); plt.close(fig)
    return C

names = {'SVM': 'SVM', 'AdaBoostM2': 'AdaBoostM2', 'OrdinalLogistic': 'Ordinal logistic regression',
         'Multinomial': 'Multinomial logistic regression', 'CLMM': 'Ordinal mixed model (CLMM)',
         'LMM_binned': 'Linear mixed model (binned)', 'SVR_binned': 'Support vector regression (binned)'}
out = {'SVM': os.path.join(FIG, 'Figure4_SVM_confusion.png'), 'AdaBoostM2': os.path.join(FIG, 'Figure5_AdaBoostM2_confusion.png'),
       'OrdinalLogistic': os.path.join(SUPP, 'FigureS1_OrdinalLogistic_confusion.png'), 'Multinomial': os.path.join(SUPP, 'FigureS2_Multinomial_confusion.png'),
       'CLMM': os.path.join(SUPP, 'FigureS3_CLMM_confusion.png'), 'LMM_binned': os.path.join(SUPP, 'FigureS4_LMM_binned_confusion.png'),
       'SVR_binned': os.path.join(SUPP, 'FigureS5_SVR_binned_confusion.png')}
for k, path in out.items():
    C = confusion(pred['S1_' + k].values, names[k] + ', leave-one-athlete-out CV', path)
    print(k, C.tolist(), 'acc %.3f' % (np.trace(C) / C.sum()))

# ---------------------------------------------------------------- Figure 6
predCont = np.asarray(m['cv'][0].predCont, float)
pli = pred.PLI.values
cols = ['#3b2f8f', '#1b9e9e', '#d9a400']
fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), dpi=300)
for k, (ax, ttl) in enumerate(zip(axes, ['Linear mixed model (population-level)', 'Support vector regression'])):
    xp = predCont[:, k]
    for c in range(3):
        sel = ytrue == c + 1
        ax.scatter(xp[sel], pli[sel], s=16, color=cols[c], alpha=0.7, label=classes[c], edgecolor='none')
    lim = [0, max(pli.max(), xp.max()) * 1.05]
    ax.plot(lim, lim, 'k--', lw=1)
    for cpt in cuts:
        ax.axvline(cpt, ls=':', color='gray', lw=1); ax.axhline(cpt, ls=':', color='gray', lw=1)
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect('equal'); ax.grid(alpha=0.3)
    ax.set_xlabel('Predicted PLI (AU)'); ax.set_ylabel('Observed PLI (AU)'); ax.set_title(ttl, fontsize=12)
    if k == 0:
        ax.legend(loc='upper left', frameon=True, title='Observed category')
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'Figure6_continuous_predictions.png'), dpi=300, facecolor='white'); plt.close(fig)
print('done')
