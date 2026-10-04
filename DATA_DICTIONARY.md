# Data dictionary: `MSOC Data.xlsx`

The workbook is the season monitoring record of one NCAA Division I men's soccer team (10-week competitive season). Both sheets used by the analysis have one row per athlete-session, stacked by session: the same roster block of 26 athletes repeats for each session block of the season (1,794 rows in total; blocks without monitoring data contain no complete rows). Row order is identical in the two sheets; the scripts verify this and stop otherwise. Athlete identifiers are de-identified study codes.

## Sheet `ML Sheet (2)` (analysis variables)

| Column | Type / units | Source | Definition | Use in the analysis |
|---|---|---|---|---|
| `Prim_V_Sec` | integer, 1 = primary, other = secondary | roster classification | Primary athletes played ≥ 45 min in ≥ 8 of the 15 matches. | Row filter: primary athletes only. |
| `Ath_ID` | integer | de-identified code | Athlete identifier. | Grouping variable (cross-validation folds, random effects, repeated-measures correlation). |
| `Prac_o_Gam` | integer, 3 = practice, 4 = match | schedule | Session type. | Descriptive only (counts of practices and matches). |
| `Return` | integer flag | roster | Return-to-play flag recorded in the monitoring workbook. | Not used as a variable; read only for the completeness test (see below). |
| `sRPE` | AU (RPE × minutes) | Borg CR-10 rating collected ~15 min after the session × session duration | Session rating of perceived exertion. | Candidate predictor. |
| `Well1_Fat` | integer 1–10 | morning electronic survey: "How fatigued are you? (in general)" | Self-reported fatigue (1 = lowest, 10 = highest). | Candidate predictor (SRWM-Fatigue). |
| `Well2_Stress` | integer 1–10 | "How stressed are you? (in general)" | Self-reported stress. | Candidate predictor (SRWM-Stress). |
| `Well3_Sore` | integer 1–10 | "How sore are you? (in general)" | Self-reported soreness. | Candidate predictor (SRWM-Soreness). |
| `P_HR_avg_%` | percent | Polar Team Pro chest sensor | Session-average heart rate as a percentage of the athlete's maximal heart rate (%maxHR). | Candidate predictor. |
| `P_Tot_Dis` | metres | Polar Team Pro 10 Hz GNSS | Total distance covered in the session (TD). | Candidate predictor. |
| `P_Training_load_score` | AU | Polar Team Pro proprietary algorithm (heart rate, energy expenditure, mechanical impact, duration) | Player load intensity (PLI). | Outcome: continuous, and categorized as Low ≤ 59.8 < Medium < 160.1 ≤ High (mean ± 0.5 SD of the analytic sample; values rounded here, the script uses the full-precision cut points 59.777… and 160.126… AU, see `results/Table_S0_PLI_cut_points.csv`). |

A row is "complete" when every column listed above is present (the scripts test all 11 columns of the sheet, i.e. the seven analysis variables `sRPE`, `Well1_Fat`, `Well2_Stress`, `Well3_Sore`, `P_HR_avg_%`, `P_Tot_Dis` and `P_Training_load_score` together with `Prim_V_Sec`, `Ath_ID`, `Prac_o_Gam` and `Return`; the scripts print the number of rows, if any, that have the seven analysis variables but lack one of the other four, and the manuscript's description of the criterion as "complete data for all analysis variables" assumes that number is zero). The analytic sample is then formed in this order: (1) keep primary athletes (`Prim_V_Sec` = 1); (2) exclude any primary athlete with fewer than 10 complete sessions (n = 1, an athlete with a single complete session); (3) exclude sessions with PLI = 0 despite non-zero heart rate, distance and sRPE, treated as device-export artifacts (n = 3; the scripts print these rows and stop if any of them does not have non-zero sRPE, heart rate and distance).

## Sheet `ALL POLAR DATA` (device export; used only for the PLI construct analysis)

| Column | Type / units | Definition | Use |
|---|---|---|---|
| `Ath_ID` | integer | Athlete identifier (must equal `ML Sheet (2)` row for row). | Alignment check. |
| `Duration` | Excel time (h:mm:ss) | Session duration recorded by the device. | Session duration in minutes for the construct regressions. |
| `Time_HR_zone_1` … `Time_HR_zone_5` | Excel time | Time spent in heart-rate zones 1–5 (50–60%, 60–70%, 70–80%, 80–90%, 90–100% of maximal heart rate). | Zone-weighted heart-rate load = Σ (minutes in zone k × k), k = 1…5 (Edwards 1993). |
| `P_Training_load_score` | AU | Same PLI as in `ML Sheet (2)` (checked for equality). | Alignment check. |
| `P_Cardio_load` | AU | Device cardio load. | Construct regression PLI ~ cardio load. |

## Derived variables (created by the scripts, not stored in the workbook)

| Variable | Definition |
|---|---|
| `Session` | Session index (one per roster block) derived from the row position within the stacked roster blocks; used as the crossed random intercept "session (training day)". |
| `DurMin`, `TRIMP`, `Cardio` | Duration in minutes; zone-weighted heart-rate load; device cardio load (from `ALL POLAR DATA`). |
| PLI category | 1 = Low, 2 = Medium, 3 = High, with the fixed cut points above. |
| z-scored predictors | Within each cross-validation fold, (x − mean of training partition) / SD of training partition; on the full sample, standardized over all 372 sessions. |

## Analytic sample produced by the scripts

372 athlete-sessions (321 practices, 51 matches) from 10 athletes; median 40 sessions per athlete (range 22–41). See `results/Table_S1_exclusion_flow.csv`.
