#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_synthetic_workbook.py - writes MSOC_Data_SYNTHETIC.xlsx, a SIMULATED workbook with the
structure of the study's monitoring workbook (same sheets, columns and stacked row layout),
so that MSOC_ML_Pipeline.m and msoc_pipeline.py can be run end to end without the original data.

All values are generated at random from simple rules (session duration x heart-rate intensity
drives the load score) and bear no relation to any athlete. Results obtained from this workbook
are meaningless and must not be reported; the file exists only to exercise the code.

Usage:  python make_synthetic_workbook.py            # writes MSOC_Data_SYNTHETIC.xlsx
"""
import numpy as np, pandas as pd

rng = np.random.default_rng(20260928)
N_ATH, N_DAYS, N_PRIMARY = 26, 69, 11
ids = np.sort(rng.choice(np.arange(1, 60), N_ATH, replace=False))          # de-identified codes
prim = np.zeros(N_ATH, int) + 2; prim[rng.choice(N_ATH, N_PRIMARY, replace=False)] = 1
single_session_athlete = np.flatnonzero(prim == 1)[0]                      # mimics the athlete excluded for < 10 sessions

day_type = np.array([3] * 42 + [4] * 15 + [0] * 12); rng.shuffle(day_type)  # 3 practice, 4 match, 0 no session
base_hr = rng.normal(63, 3, N_ATH); base_fat = rng.integers(2, 6, N_ATH); base_str = rng.integers(1, 5, N_ATH); base_sore = rng.integers(2, 6, N_ATH)

rows_ml, rows_polar = [], []
for d in range(N_DAYS):
    kind = day_type[d]
    dur_day = rng.uniform(60, 110) if kind == 3 else rng.uniform(92, 100) if kind == 4 else np.nan
    intensity_day = rng.normal(0, 3)
    for a in range(N_ATH):
        present = kind != 0 and rng.random() < (0.85 if prim[a] == 1 else 0.5)
        if a == single_session_athlete:
            present = kind != 0 and d == 10
        rec = dict(Prim_V_Sec=prim[a], Ath_ID=int(ids[a]), Prac_o_Gam=(kind if kind else np.nan), Return=0,
                   sRPE=np.nan, Well1_Fat=np.nan, Well2_Stress=np.nan, Well3_Sore=np.nan,
                   **{'P_HR_avg_%': np.nan}, P_Tot_Dis=np.nan, P_Training_load_score=np.nan)
        pol = dict(Ath_ID=int(ids[a]), Duration=np.nan, **{'Time_HR_zone_%d' % z: np.nan for z in range(1, 6)},
                   P_Training_load_score=np.nan, P_Cardio_load=np.nan)
        if present:
            dur = dur_day * rng.uniform(0.55, 1.0) if kind == 4 else dur_day * rng.uniform(0.9, 1.0)
            hr = np.clip(base_hr[a] + intensity_day + (8 if kind == 4 else 0) + rng.normal(0, 4), 40, 90)
            td = dur * (rng.uniform(55, 90) + (25 if kind == 4 else 0)) * rng.uniform(0.9, 1.1)
            rpe = int(np.clip(round(1 + (hr - 45) / 6 + rng.normal(0, 1)), 1, 10))
            pli = dur * np.exp(0.07 * (hr - 60)) * 0.9 * np.exp(rng.normal(0, 0.15))
            zones = np.exp(-0.5 * ((np.arange(1, 6) - (hr - 45) / 10) / 1.2) ** 2); zones = dur * zones / zones.sum()
            rec.update(sRPE=round(rpe * dur), **{'P_HR_avg_%': round(hr)}, P_Tot_Dis=round(td), P_Training_load_score=round(pli))
            if rng.random() > 0.10:                                       # wellness missing on 10 % of days
                rec.update(Well1_Fat=int(np.clip(base_fat[a] + rng.integers(-1, 3), 1, 10)),
                           Well2_Stress=int(np.clip(base_str[a] + rng.integers(-1, 3), 1, 10)),
                           Well3_Sore=int(np.clip(base_sore[a] + rng.integers(-1, 3), 1, 10)))
            pol.update(Duration=dur / 1440, **{'Time_HR_zone_%d' % z: zones[z - 1] / 1440 for z in range(1, 6)},
                       P_Training_load_score=rec['P_Training_load_score'], P_Cardio_load=round(pli * 0.95 * np.exp(rng.normal(0, 0.05))))
        rows_ml.append(rec); rows_polar.append(pol)

ml = pd.DataFrame(rows_ml); polar = pd.DataFrame(rows_polar)
# three "export artifact" rows: PLI = 0 with non-zero heart rate, distance and sRPE
ok = np.flatnonzero((ml.Prim_V_Sec == 1) & ml.P_Training_load_score.notna())
for i in rng.choice(ok, 3, replace=False):
    ml.loc[i, 'P_Training_load_score'] = 0; polar.loc[i, 'P_Training_load_score'] = 0
with pd.ExcelWriter('MSOC_Data_SYNTHETIC.xlsx') as xw:
    ml.to_excel(xw, sheet_name='ML Sheet (2)', index=False)
    polar.to_excel(xw, sheet_name='ALL POLAR DATA', index=False)
print('wrote MSOC_Data_SYNTHETIC.xlsx: %d rows (%d athletes x %d days); complete rows %d'
      % (len(ml), N_ATH, N_DAYS, int(ml.notna().all(1).sum())))
