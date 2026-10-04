# -*- coding: utf-8 -*-
"""Draws Figure 1 (leave-one-athlete-out cross-validation pipeline schematic) to figures/Figure1_pipeline.png.
The figure contains no data; it documents the procedure implemented in MSOC_ML_Pipeline.m Section 8.
Requirements: matplotlib."""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures'); os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10.5})
fig = plt.figure(figsize=(10.3, 7.9), dpi=300)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis('off')

def box(x, y, w, h, text, fc='white', ec='black', lw=1.1, fs=10.5, weight='normal', ha='center', ls='-', pad=0.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=%.2f,rounding_size=1.2' % pad, fc=fc, ec=ec, lw=lw, ls=ls))
    tx = x + w / 2 if ha == 'center' else x + 1.2
    ax.text(tx, y + h / 2, text, ha=ha, va='center', fontsize=fs, weight=weight, linespacing=1.35)

def arrow(x0, y0, x1, y1, lw=1.2):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>', mutation_scale=14, lw=lw, color='black', shrinkA=0, shrinkB=0))

# top: data set
box(11, 88, 78, 9.5,
    'Analytic data set: 10 primary athletes, 372 athlete-sessions\n'
    'Candidate predictors: sRPE, SRWM-Fatigue, SRWM-Stress, SRWM-Soreness, %maxHR, TD\n'
    'Outcome: continuous PLI and PLI category (Low / Medium / High; mean ± 0.5 SD cut points,\n'
    'estimated once from the analytic sample and held fixed; fold-specific recomputation changes 3 of 372 categories)',
    fc='#f2f2f2', fs=10)
# outer loop frame
ax.add_patch(FancyBboxPatch((2.5, 15), 95, 69.5, boxstyle='round,pad=0.6,rounding_size=1.5', fc='none', ec='black', lw=1.4, ls=(0, (6, 4))))
ax.text(5, 82.2, 'Outer loop: leave-one-athlete-out cross-validation (10 folds)', fontsize=11, weight='bold', va='center')
arrow(56, 88, 56, 84.8)
# split arrows
ax.plot([32.5, 32.5, 80.25, 80.25], [78.2, 79.6, 79.6, 78.2], color='black', lw=1.2)
ax.plot([56, 56], [84.8, 79.6], color='black', lw=1.2)
arrow(32.5, 79.6, 32.5, 76.6); arrow(80.25, 79.6, 80.25, 76.6)
# training partition
box(5.5, 68.5, 54, 8, 'Training partition: the other 9 athletes\n(331–350 sessions)', fc='#dfe8f4', fs=12, weight='bold')
# held-out
box(65.5, 68.5, 29.5, 8, 'Held-out athlete (22–41 sessions)\nnever used for selection, tuning,\nstandardisation or fitting', fc='#f9e3d6', fs=10.5)
# steps
arrow(32.5, 68.5, 32.5, 65.8)
box(5.5, 57.8, 54, 8, '1  Feature selection: MRMR (mutual-information quotient) on the\n'
    '    training athletes only → top-3 predictors for this fold; applied once\n'
    '    per fold (not repeated inside the inner loop); same set for every model', ha='left', fs=9.6)
arrow(32.5, 57.8, 32.5, 55.1)
box(5.5, 49.6, 54, 5.5, '2  Standardisation: z-scores from the training-partition mean and SD\n'
    '    (SVR outcome: PLI standardised with the training-partition mean and SD)', ha='left', fs=9.6)
arrow(32.5, 49.6, 32.5, 46.9)
box(5.5, 36.4, 54, 10.5, '3  Hyperparameter tuning (SVM, AdaBoostM2, SVR)\n'
    '    Bayesian optimisation: fixed 8–9-point initial design, then\n'
    '    expected-improvement-plus, 40 evaluations in total; objective =\n'
    '    inner leave-one-athlete-out error over the 9 training athletes', ha='left', fs=9.6)
arrow(32.5, 36.4, 32.5, 33.7)
box(5.5, 19.5, 54, 14.2, '4  Fit seven models on all training sessions\n'
    '    SVM (RBF kernel, one-vs-one)  ·  AdaBoostM2\n'
    '    ordinal logistic regression  ·  multinomial logistic regression\n'
    '    cumulative-link mixed model (random athlete intercept)\n'
    '    linear mixed model (LMM) on continuous PLI\n'
    '    support vector regression (SVR) on continuous PLI', ha='left', fs=9.6)
# right column: apply
arrow(80.25, 68.5, 80.25, 51.5)
ax.text(81.75, 60, 'held-out\nsessions', fontsize=10, style='italic', va='center')
box(65.5, 26, 29.5, 25.5, 'Apply the fold-specific predictor set,\nstandardisation and fitted models\nto the held-out athlete\n\n'
    '→ predicted PLI category (all models)\n→ predicted continuous PLI (LMM, SVR),\n    binned with the same fixed cut points', fc='#f9e3d6', fs=10.5)
arrow(59.5, 30.5, 65.5, 30.5)
ax.text(62.5, 32.4, 'fitted\nmodels', fontsize=9, style='italic', ha='center', va='bottom', linespacing=1.1)
# bottom
arrow(80.25, 26, 80.25, 13.2)
box(5, 2.3, 90, 10.4,
    'Pooled out-of-sample predictions (372 sessions from 10 folds)\n'
    'Classification: accuracy (mean ± SD across athletes, and pooled), balanced accuracy, macro-F1,\n'
    'per-class sensitivity and specificity, Low↔High confusions. Between models: discordant counts (McNemar),\n'
    'cluster-bootstrap 95% CIs and cluster-level tests (clusters = training days, athletes). Continuous PLI (LMM, SVR): RMSE, MAE, R².', fc='#f2f2f2', fs=9.4)
fig.savefig(os.path.join(OUT, 'Figure1_pipeline.png'), dpi=300, facecolor='white')
print('wrote', os.path.join(OUT, 'Figure1_pipeline.png'))
