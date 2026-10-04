%% MSOC_ML_Pipeline.m  (v2.4, 2026-10-03)
% =========================================================================
% Classification of Player Load Intensity (PLI) in NCAA-DI men's soccer
% from internal-load, external-load and wellness metrics.
%
% ONE script that reproduces every number, table and figure in the
% manuscript, from the raw workbook to the final outputs.
%
% Version history
%   v1  2026-09-25  first full run (R2022b Update 1, seed 20260925, 270 min)
%                   -- superseded; not reported
%   v2.4 2026-10-03 - Section 9a: the proportional-odds assumption is tested
%                     with the Brant test (separate binary logits at each
%                     cut point; Wald test of equal slopes, omnibus df = 3
%                     and per predictor df = 1; asymptotic and parametric-
%                     bootstrap p-values; Table S12b). The likelihood-ratio
%                     test against the non-parallel-slopes cumulative-logit
%                     model is no longer reported, because that model gives
%                     negative fitted probabilities for 17 sessions; its fit
%                     is still printed for information. The bootstrap uses
%                     its own random stream (RandStream, seed cfg.seed), so
%                     no other output changes. Same code and results as the
%                     standalone script MSOC_brant_test.m (run 3 Oct 2026).
%   v2.3 2026-09-29 - Section 9a: the proportional-odds test now uses an
%                     explicitly maximised non-parallel-slopes cumulative-
%                     logit likelihood (cumlogitNPFit) with a convergence
%                     check; mnrfit's alternative fit (which stopped at its
%                     iteration limit) is retained in the log for comparison.
%                   - Section 8d: cluster bootstrap (training days; athletes)
%                     and cluster-level sign-flip tests for every pairwise
%                     accuracy difference (Table S7b); McNemar counts kept
%                     as descriptive statistics.
%                   - Section 3b: cut points recomputed with each athlete
%                     omitted (Table S0b). Section 6: product-only model
%                     added. Section 8e: inner-split MRMR agreement check
%                     (Table S9b). Section 8f: ordinal model with all six
%                     candidate predictors under LOAO (Table S17).
%                     Section 8g: leave-one-day-out evaluation of the four
%                     deterministic models (Table S18). Section 2: the
%                     PLI = 0 rows are checked to have non-zero sRPE, heart
%                     rate and distance. All additions are deterministic
%                     post-processing or seeded (bootstrap); no reported
%                     number from v2.1 changes except the proportional-odds
%                     statistic if the converged fit differs from mnrfit's.
%   v2.2 2026-09-28 - Section 8c added: distance of every misclassified
%                     session from the nearest category cut point (the
%                     "30 of 35 SVM errors within 20 AU" statement in the
%                     Results). Deterministic; computed from the Section 8
%                     predictions, so no reported number changes.
%                   - completeness diagnostic written to the log (Section 2).
%                   - continuous-prediction figure renamed Figure6 (it is
%                     Figure 6 in the manuscript, not a supplemental figure).
%                   - comments on the proportional-odds test corrected: the
%                     alternative model is the non-parallel-slopes
%                     cumulative-logit model, not the multinomial model.
%   v2.1 2026-09-26 - Section 8b added: leave-one-athlete-out evaluation of the
%                     linear mixed model with main effects only (no interaction
%                     terms), as a sensitivity analysis of the functional form.
%                     Deterministic; every other output is unchanged.
%   v2  2026-09-26  - hyperparameter search starts from a fixed log-spaced
%                     design (bayesopt 'InitialX'), 40 evaluations per fold,
%                     and the inner-loop objective at the chosen point is
%                     logged (Table S10). In v1, one SVM fold (athlete 39)
%                     selected KernelScale = 0.11 and predicted "Medium" for
%                     8 of 9 High sessions.
%                   - multinomial (non-parallel-slopes) logistic regression
%                     added to the LOAO comparison, because the
%                     likelihood-ratio proportional-odds test then used
%                     suggested non-parallel slopes (LR = 8.36); that test
%                     is not valid here and was replaced in v2.4 by the
%                     Brant test (p = 0.14), see Section 9a.
%                   - the "fixed sRPE + %HRmax + TD" strategy is removed:
%                     MRMR selected exactly those three predictors in 10/10
%                     folds, so it duplicated the primary strategy and
%                     differed only in the optimizer's random start. An
%                     optional seed replicate of the primary strategy
%                     (cfg.seedReplicate) quantifies that tuning variability.
%                   - figures: classes ordered Low/Medium/High in confusion
%                     charts (v1 sorted them alphabetically), masked upper
%                     triangle in Figure 2, legend in the continuous-
%                     prediction figure.
%                   - optional parallel Bayesian optimization
%                     (cfg.useParallel) for exploratory runs.
%
% Requirements
%   MATLAB R2022b or later. The results reported in the manuscript were
%   produced with MATLAB R2026a Update 5, Statistics and Machine Learning
%   Toolbox 26.1 and Optimization Toolbox 26.1 (run of 26-Sep-2026, script
%   v2.1, seed 20260925, 17 min, serial; reproduced byte for byte by the
%   v2.3 run of 01-Oct-2026, whose aggregate outputs are in results/).
%   v2.2-v2.4 add post-processing, checks and the Brant test (see version
%   history); the reported Brant results come from MSOC_brant_test.m,
%   which contains the same code as Section 9a.
%   Statistics and Machine Learning Toolbox (fitcecoc, fitcensemble,
%     fitrsvm, bayesopt, mnrfit, fitlme, corr)
%   Optimization Toolbox is optional (fminunc for the ordinal mixed model;
%     fminsearch from base MATLAB is used otherwise)
%   Parallel Computing Toolbox is optional (parallel bayesopt evaluations)
%
% Input
%   'MSOC Data.xlsx' (same folder as this script)
%     sheet 'ML Sheet (2)'   : session-level analysis variables
%     sheet 'ALL POLAR DATA' : Polar Team Pro export in the same row order
%                              (duration and HR-zone times; used only for the
%                              PLI construct analysis in Section 6)
%
% Output (folder cfg.outDir)
%   CSV tables, PNG figures (300 dpi), MSOC_results.mat, run_log.txt,
%   software_versions.txt
%
% Analysis overview
%   1  Import and row-order checks
%   2  Analytic sample (primary athletes, exclusion flow)
%   3  PLI categories: mean +/- 0.5 SD cut points (computed once, then fixed)
%   3b Sensitivity of the cut points to the held-out athlete
%   4  Descriptive statistics
%   5  Associations with PLI: Spearman rank correlation and
%      repeated-measures correlation (rmcorr; Bakdash & Marusich, 2017)
%   6  PLI construct analysis: how much of the proprietary PLI is explained
%      by heart-rate intensity x duration
%   7  Descriptive MRMR ranking on the full analytic sample (not used for
%      model building). MRMR uses the mutual information quotient (Ding &
%      Peng, 2005) on predictors discretized into cfg.mrmrBins
%      equal-frequency bins; implemented below (mrmrMIQ) rather than with
%      fscmrmr, whose scores after the first feature were numerically zero
%      in this data set, so later ranks were arbitrary ties.
%   8  Leave-one-athlete-out (LOAO) cross-validation. In every outer fold,
%      using only the training athletes:
%        - MRMR selects the predictors (or a fixed set for sensitivity runs)
%        - the SAME predictors are given to all seven models:
%            SVM (RBF, one-vs-one), AdaBoostM2, ordinal logistic regression
%            (proportional odds), multinomial logistic regression
%            (non-parallel slopes), ordinal mixed model with a random
%            athlete intercept (CLMM), linear mixed model on continuous PLI
%            (LMM, then binned), support vector regression on continuous
%            PLI (SVR, then binned)
%        - SVM, AdaBoostM2 and SVR hyperparameters are tuned by Bayesian
%          optimization (fixed initial design + expected-improvement-plus)
%          with an inner leave-one-athlete-out loop
%        - the held-out athlete is never used for selection, tuning,
%          standardization or fitting
%   8b Sensitivity: the LMM refitted with main effects only under the same
%      folds and predictors (functional form)
%   8c Distance of every misclassified session from the nearest cut point
%   8d Cluster bootstrap and cluster-level sign-flip tests for pairwise
%      accuracy differences (clusters = training days, and athletes)
%   8e Inner-split MRMR agreement check (was selection stable within the
%      inner tuning loop?)
%   8f Ordinal model with all six candidate predictors under LOAO
%   8g Leave-one-day-out evaluation of the deterministic models
%   9  Inference models on the full analytic sample: ordinal logistic
%      regression with a proportional-odds test (alternative: non-parallel-
%      slopes cumulative-logit model), CLMM, and LMM with crossed athlete
%      and session random intercepts
%  10  Figures
%
% Coefficient sign convention for ordinal models: logit P(Y <= k) =
% theta_k - x*beta, so beta > 0 means higher odds of a higher PLI category.
% (MATLAB's mnrfit uses logit P(Y <= k) = alpha_k + x*b, i.e. beta = -b.)
% =========================================================================

close all; clear; clc;

%% 0  CONFIGURATION
cfg = struct();
cfg.version    = 'v2.4 (2026-10-03)';
cfg.dataFile   = 'MSOC Data.xlsx';
cfg.mlSheet    = 'ML Sheet (2)';
cfg.polarSheet = 'ALL POLAR DATA';
cfg.outDir     = fullfile(pwd, 'MSOC_results');

cfg.seed       = 20260925;       % global RNG seed (reported in manuscript)
% PLI categories: Low <= mean - 0.5 SD < Medium < mean + 0.5 SD <= High
% (stratification by half standard deviations; Bacon & Mauger, 2017)
cfg.cutRule    = 'meanHalfSD';   % 'meanHalfSD' or 'fixed'
cfg.cutSample  = 'analytic';     % 'analytic' (primary athletes after exclusions)
                                 % or 'allPlayers' (all complete rows)
cfg.fixedCutPoints = [75 150];   % used only if cutRule = 'fixed' (previous analyses)

% Exclusion rules (all reported in the exclusion-flow table)
cfg.minSessionsPerAthlete = 10;  % primary athletes with fewer complete
                                 % sessions cannot form a meaningful LOAO fold
cfg.excludeZeroPLI = true;       % PLI = 0 with nonzero HR, sRPE and distance
                                 % indicates a Polar export artifact

% Model settings
cfg.bayesEvals   = 40;           % Bayesian-optimization evaluations per fold and
                                 % model, INCLUDING the fixed initial design
                                 % (9 points for SVM, 8 for AdaBoostM2, 8 for SVR)
cfg.nGH          = 25;           % Gauss-Hermite nodes for the ordinal mixed model
cfg.lmmInteract  = true;         % LMM includes all two-way interactions
cfg.mrmrBins     = 10;           % equal-frequency bins for MRMR mutual information
cfg.useParallel  = false;        % true = parallel bayesopt evaluations (needs the
                                 % Parallel Computing Toolbox; roughly 3-4x faster on
                                 % a 4-8 core laptop). Parallel evaluations complete
                                 % asynchronously, so a parallel run is NOT bit-for-bit
                                 % reproducible from the seed; use false for the run
                                 % reported in the manuscript.
cfg.seedReplicate = false;       % true = repeat the primary strategy with a
                                 % second optimizer seed (tuning variability)
cfg.quickTest    = false;        % true = 12 evaluations, primary strategy only
                                 % (dry run; do NOT report these numbers)

% Candidate predictors (column order of Xcand below)
featVar   = {'sRPE','Fatigue','Stress','Soreness','HRpct','TD'};
featLabel = {'sRPE','Wellness-Fatigue','Wellness-Stress','Wellness-Soreness', ...
             '%HRmax','Total distance'};

% Feature strategies evaluated with identical outer folds.
% seedOffset changes only the optimizer's random start (fold seed =
% cfg.seed + 100*s + f + seedOffset).
cfg.strategies = struct( ...
    'name',       {'Primary: within-fold MRMR top 3', ...
                   'Sensitivity: no heart rate (sRPE + TD)'}, ...
    'mode',       {'mrmr', 'fixed'}, ...
    'k',          {3, 2}, ...
    'features',   {[], [1 6]}, ...
    'seedOffset', {0, 0});
if cfg.seedReplicate
    cfg.strategies(end+1) = struct( ...
        'name', 'Replicate: primary strategy, second optimizer seed', ...
        'mode', 'mrmr', 'k', 3, 'features', [], 'seedOffset', 5000);
end

if cfg.quickTest
    cfg.bayesEvals = 12;
    cfg.strategies = cfg.strategies(1);
end

modelNames = {'SVM','AdaBoostM2','OrdinalLogistic','Multinomial','CLMM','LMM_binned','SVR_binned'};
classNames = {'Low','Medium','High'};
K = 3;

if isMATLABReleaseOlderThan('R2022b')
    warning('MSOC:release', 'Written and tested for R2022b or later.');
end
cfg.hasFminunc = license('test','Optimization_Toolbox') && ~isempty(which('fminunc'));

if ~exist(cfg.outDir, 'dir'), mkdir(cfg.outDir); end
diary(fullfile(cfg.outDir, 'run_log.txt')); diary on;
rng(cfg.seed, 'twister');
tStart = tic;

fprintf('MSOC ML pipeline %s  |  %s  |  MATLAB %s\n', cfg.version, string(datetime('now')), version);
fprintf('Seed = %d | Bayesian evaluations per fold = %d | parallel = %d | fminunc available = %d\n\n', ...
    cfg.seed, cfg.bayesEvals, cfg.useParallel, cfg.hasFminunc);

%% 1  IMPORT AND ROW-ORDER CHECKS
mlVars = {'Prim_V_Sec','Ath_ID','Prac_o_Gam','Return','sRPE','Well1_Fat', ...
          'Well2_Stress','Well3_Sore','P_HR_avg_%','P_Tot_Dis','P_Training_load_score'};
R = readSheet(cfg.dataFile, cfg.mlSheet, mlVars);
for v = 1:numel(mlVars), R.(mlVars{v}) = toDouble(R.(mlVars{v})); end
nRows = height(R);

% The sheet is stacked by session: the same roster block repeats for every
% session. Derive the session index and verify the stacking.
ids = R.Ath_ID;
nPerSession = find(ids(2:end) == ids(1), 1);        % roster size
assert(~isempty(nPerSession) && mod(nRows, nPerSession) == 0, ...
    'MSOC:layout', 'Rows are not an integer number of roster blocks.');
nSessions = nRows / nPerSession;
assert(isequal(ids, repmat(ids(1:nPerSession), nSessions, 1)), ...
    'MSOC:layout', 'Athlete order differs between session blocks.');
R.Session = ceil((1:nRows)' / nPerSession);
fprintf('Sheet "%s": %d rows = %d sessions x %d rostered athletes\n', ...
    cfg.mlSheet, nRows, nSessions, nPerSession);

% Polar export (duration and HR-zone times) for the construct analysis
polarVars = {'Ath_ID','Duration','Time_HR_zone_1','Time_HR_zone_2', ...
             'Time_HR_zone_3','Time_HR_zone_4','Time_HR_zone_5', ...
             'P_Training_load_score','P_Cardio_load'};
P = readSheet(cfg.dataFile, cfg.polarSheet, polarVars);
assert(height(P) == nRows && isequal(toDouble(P.Ath_ID), ids), ...
    'MSOC:align', 'ALL POLAR DATA is not row-aligned with the ML sheet.');
pliP = toDouble(P.P_Training_load_score);
both = ~isnan(pliP) & ~isnan(R.P_Training_load_score);
assert(all(pliP(both) == R.P_Training_load_score(both)), ...
    'MSOC:align', 'PLI differs between sheets.');
R.DurMin = toMinutes(P.Duration);
zoneMin = zeros(nRows, 5);
for z = 1:5, zoneMin(:, z) = toMinutes(P.(sprintf('Time_HR_zone_%d', z))); end
R.TRIMP  = zoneMin * (1:5)';                        % Edwards zone-weighted TRIMP
R.Cardio = toDouble(P.P_Cardio_load);

%% 2  ANALYTIC SAMPLE AND EXCLUSION FLOW
complete = all(~isnan(R{:, mlVars}), 2);
primary  = complete & R.Prim_V_Sec == 1;

% Diagnostic (log only): the completeness test above requires all 11 columns
% of the sheet. The manuscript describes the criterion as "all analysis
% variables present"; the line below shows whether the two differ, i.e.
% whether any row with the seven analysis variables present lacks one of
% Prim_V_Sec, Ath_ID, Prac_o_Gam or Return.
analysisVars = {'sRPE','Well1_Fat','Well2_Stress','Well3_Sore', ...
                'P_HR_avg_%','P_Tot_Dis','P_Training_load_score'};
completeAnalysis = all(~isnan(R{:, analysisVars}), 2);
fprintf('Rows with the seven analysis variables present: %d; of these, %d lack Prim_V_Sec, Ath_ID, Prac_o_Gam or Return (excluded by the 11-column test).\n', ...
    sum(completeAnalysis), sum(completeAnalysis & ~complete));

flowStep = {'Rows in sheet (sessions x rostered athletes)'; ...
            'Complete rows (all analysis variables present)'; ...
            'Complete rows, primary athletes'};
flowN    = [nRows; sum(complete); sum(primary)];
flowAth  = [numel(unique(ids)); numel(unique(ids(complete))); numel(unique(ids(primary)))];

keep = primary;
[g, gid] = findgroups(ids(keep));
nPerAth  = splitapply(@numel, ids(keep), g);
lowIDs   = gid(nPerAth < cfg.minSessionsPerAthlete);
keep     = keep & ~ismember(ids, lowIDs);
flowStep{end+1,1} = sprintf('Excluded primary athletes with < %d complete sessions (ID %s)', ...
    cfg.minSessionsPerAthlete, strjoin(string(lowIDs'), ', '));
flowN(end+1,1)   = sum(keep);
flowAth(end+1,1) = numel(unique(ids(keep)));

if cfg.excludeZeroPLI
    zeroPLI = keep & R.P_Training_load_score == 0;
    keep    = keep & ~zeroPLI;
    flowStep{end+1,1} = sprintf('Excluded sessions with PLI = 0 (export artifact; n = %d)', sum(zeroPLI));
    flowN(end+1,1)   = sum(keep);
    flowAth(end+1,1) = numel(unique(ids(keep)));
    if any(zeroPLI)
        zp = R(zeroPLI, {'Ath_ID','sRPE','P_HR_avg_%','P_Tot_Dis','P_Training_load_score'});
        fprintf('\nExcluded PLI = 0 rows (athlete, sRPE, %%HRmax, TD, PLI):\n'); disp(zp);
        assert(all(zp.sRPE > 0 & zp.('P_HR_avg_%') > 0 & zp.P_Tot_Dis > 0), 'MSOC:zeroPLI', ...
            'A PLI = 0 row does not have non-zero sRPE, heart rate and distance; revise the exclusion rule.');
    end
end
flowTbl = table(flowStep, flowN, flowAth, 'VariableNames', {'Step','Rows','Athletes'});
disp(flowTbl);
writetable(flowTbl, fullfile(cfg.outDir, 'Table_S1_exclusion_flow.csv'));

D = R(keep, :);
n    = height(D);
ath  = D.Ath_ID;
athIDs = unique(ath);
nAth = numel(athIDs);
pli  = D.P_Training_load_score;
Xcand = [D.sRPE, D.Well1_Fat, D.Well2_Stress, D.Well3_Sore, D.('P_HR_avg_%'), D.P_Tot_Dis];

% Data-integrity notes written to the log (no exclusions applied here)
lowPLI = find(pli < 10);
if ~isempty(lowPLI)
    fprintf('\nNote: %d session(s) with PLI < 10 AU retained (athlete, session, PLI, duration min):\n', numel(lowPLI));
    disp([ath(lowPLI), D.Session(lowPLI), pli(lowPLI), D.DurMin(lowPLI)]);
end
fprintf('Wellness item ranges: Fatigue %d-%d, Stress %d-%d, Soreness %d-%d\n', ...
    min(D.Well1_Fat), max(D.Well1_Fat), min(D.Well2_Stress), max(D.Well2_Stress), ...
    min(D.Well3_Sore), max(D.Well3_Sore));

%% 3  PLI CATEGORIES
% Cut points are computed ONCE from the chosen reference sample, before any
% cross-validation, and then held fixed. They define the outcome and are not
% model parameters.
switch cfg.cutRule
    case 'meanHalfSD'
        switch cfg.cutSample
            case 'analytic',   refPLI = pli;                               % primary athletes, after exclusions
            case 'allPlayers', refPLI = R.P_Training_load_score(complete); % all complete rows
        end
        cfg.cutPoints = mean(refPLI) + [-0.5 0.5] * std(refPLI);
        cutNote = sprintf('mean +/- 0.5 SD of %s sample (n = %d; mean = %.2f, SD = %.2f)', ...
            cfg.cutSample, numel(refPLI), mean(refPLI), std(refPLI));
    case 'fixed'
        cfg.cutPoints = cfg.fixedCutPoints;
        cutNote = 'fixed values';
end
ycat = pliCategory(pli, cfg.cutPoints);
catN = accumarray(ycat, 1, [K 1]);
fprintf('\nPLI cut points: %.2f and %.2f AU (%s)\n', cfg.cutPoints, cutNote);
fprintf('Low <= %.2f < Medium < %.2f <= High: ', cfg.cutPoints);
for c = 1:K
    fprintf('%s = %d (%.1f%%)  ', classNames{c}, catN(c), 100*catN(c)/n);
end
fprintf('\n');
cutTbl = table({'Low/Medium'; 'Medium/High'}, cfg.cutPoints(:), repmat({cutNote}, 2, 1), ...
    'VariableNames', {'Boundary','PLI_AU','Rule'});
writetable(cutTbl, fullfile(cfg.outDir, 'Table_S0_PLI_cut_points.csv'));

%% 3b  SENSITIVITY OF THE CUT POINTS TO THE HELD-OUT ATHLETE
% The cut points are estimated from all analytic sessions, including those of
% each athlete who is later held out. Recompute them with each athlete omitted
% and count the held-out sessions whose category would change under the
% fold-specific cut points. Deterministic; nothing downstream uses these values.
cpRows = zeros(nAth, 5);
for f = 1:nAth
    v    = pli(ath ~= athIDs(f));
    cpF  = mean(v) + [-0.5 0.5] * std(v);
    catF = pliCategory(pli, cpF);
    cpRows(f, :) = [athIDs(f), cpF, sum(catF ~= ycat), ...
                    sum(catF(ath == athIDs(f)) ~= ycat(ath == athIDs(f)))];
end
cutSensTbl = array2table(cpRows, 'VariableNames', {'HeldOutAthlete','Cut_low_AU', ...
    'Cut_high_AU','Sessions_changed_all','Sessions_changed_heldout'});
fprintf(['\nCut-point sensitivity (each athlete omitted): low %.2f-%.2f AU, high %.2f-%.2f AU; ' ...
         'held-out sessions whose category would change: %d of %d\n'], ...
    min(cpRows(:,2)), max(cpRows(:,2)), min(cpRows(:,3)), max(cpRows(:,3)), sum(cpRows(:,5)), n);
writetable(cutSensTbl, fullfile(cfg.outDir, 'Table_S0b_cut_point_sensitivity.csv'));

%% 4  DESCRIPTIVE STATISTICS (analytic sample)
descVars = [featLabel, {'PLI (AU)'}];
descX    = [Xcand, pli];
descTbl  = table(descVars', mean(descX)', std(descX)', min(descX)', max(descX)', ...
    'VariableNames', {'Variable','Mean','SD','Min','Max'});
sessPerAth = accumarray(findgroups(ath), 1);
fprintf('\nAnalytic sample: %d athletes, %d athlete-sessions (%d practice, %d match); ', ...
    nAth, n, sum(D.Prac_o_Gam == 3), sum(D.Prac_o_Gam == 4));
fprintf('sessions per athlete: median %g (range %d-%d)\n', ...
    median(sessPerAth), min(sessPerAth), max(sessPerAth));
disp(descTbl);
writetable(descTbl, fullfile(cfg.outDir, 'Table_1_descriptives.csv'));

%% 5  ASSOCIATIONS WITH PLI: SPEARMAN AND REPEATED-MEASURES CORRELATION
% Spearman: rank-based, appropriate for the ordinal wellness and CR-10-based
% variables. rmcorr: within-athlete association that accounts for repeated
% sessions nested in athletes (df = N - athletes - 1).
[rhoAll, pAll] = corr([pli, Xcand], 'Type', 'Spearman');
rmR = zeros(6,1); rmP = zeros(6,1); rmDf = zeros(6,1);
for j = 1:6
    [rmR(j), rmP(j), rmDf(j)] = rmcorr(Xcand(:, j), pli, ath);
end
corrTbl = table(featLabel', rhoAll(2:end, 1), pAll(2:end, 1), holm(pAll(2:end, 1)), ...
    rmR, rmDf, rmP, holm(rmP), ...
    'VariableNames', {'Variable','Spearman_rho','Spearman_p','Spearman_p_Holm', ...
                      'rmcorr_r','rmcorr_df','rmcorr_p','rmcorr_p_Holm'});
fprintf('\nAssociations with continuous PLI:\n'); disp(corrTbl);
writetable(corrTbl, fullfile(cfg.outDir, 'Table_2_correlations_with_PLI.csv'));
spearTbl = array2table(rhoAll, 'VariableNames', matlab.lang.makeValidName(['PLI', featLabel]), ...
    'RowNames', ['PLI', featLabel]);
writetable(spearTbl, fullfile(cfg.outDir, 'Table_S2_spearman_matrix.csv'), 'WriteRowNames', true);

%% 6  PLI CONSTRUCT ANALYSIS (how the proprietary score relates to its inputs)
% Polar describes the Team Pro training-load score as combining heart rate,
% energy expenditure, mechanical impact and duration. These regressions show
% how much of PLI is recovered from HR intensity and duration, which is
% needed to interpret the predictive accuracy of %HRmax-based models.
C6 = table(pli, D.DurMin, D.('P_HR_avg_%'), D.TRIMP, D.P_Tot_Dis, D.sRPE, D.Cardio, ...
    'VariableNames', {'PLI','Dur','HR','TRIMP','TD','sRPE','Cardio'});
% 'PLI ~ Dur*HR' = main effects + interaction; 'PLI ~ Dur:HR' = product only.
constructForms = {'PLI ~ Dur', 'PLI ~ HR', 'PLI ~ Dur*HR', 'PLI ~ Dur:HR', 'PLI ~ TRIMP', ...
                  'PLI ~ Cardio', 'TD ~ Dur', 'sRPE ~ Dur'};
constructR2 = zeros(numel(constructForms), 1);
for j = 1:numel(constructForms)
    mdl = fitlm(C6, constructForms{j});
    constructR2(j) = mdl.Rsquared.Ordinary;
end
constructTbl = table(constructForms', constructR2, 'VariableNames', {'Model','R2'});
fprintf('\nPLI construct analysis (OLS, analytic sample; Dur = session duration, min):\n');
disp(constructTbl);
writetable(constructTbl, fullfile(cfg.outDir, 'Table_S3_PLI_construct.csv'));

%% 7  DESCRIPTIVE MRMR RANKING ON THE FULL ANALYTIC SAMPLE (NOT used for models)
[mrmrIdx, mrmrScore] = mrmrMIQ(Xcand, ycat, cfg.mrmrBins);   % scores in rank order
mrmrTbl = table((1:6)', featLabel(mrmrIdx)', mrmrScore', ...
    'VariableNames', {'Rank','Variable','MRMR_score'});
fprintf('\nDescriptive MRMR ranking (full sample; not used for model building):\n');
disp(mrmrTbl);
writetable(mrmrTbl, fullfile(cfg.outDir, 'Table_S4_MRMR_full_sample.csv'));

%% 8  LEAVE-ONE-ATHLETE-OUT CROSS-VALIDATION
nMod   = numel(modelNames);
nStrat = numel(cfg.strategies);
perfRows = {};                 % collected into Table 3
perClassRows = {};
mcnRows = {};
selRows = {};
hpRows  = {};
contRows = {};
cv = struct();

for s = 1:nStrat
    st = cfg.strategies(s);
    fprintf('\n=== LOAO strategy %d/%d: %s ===\n', s, nStrat, st.name);
    pred     = nan(n, nMod);
    predCont = nan(n, 2);      % continuous predictions: LMM, SVR
    selLog   = false(nAth, 6);

    for f = 1:nAth
        rng(cfg.seed + 100*s + f + st.seedOffset, 'twister');    % fold-level reproducibility
        te = ath == athIDs(f);
        tr = ~te;

        % ---- predictors: chosen from TRAINING athletes only ----
        if strcmp(st.mode, 'mrmr')
            idx = mrmrMIQ(Xcand(tr, :), ycat(tr), cfg.mrmrBins);
            sel = sort(idx(1:st.k));
        else
            sel = st.features;
        end
        selLog(f, sel) = true;
        Xtr = Xcand(tr, sel);  Xte = Xcand(te, sel);
        ytr = ycat(tr);        gtr = ath(tr);

        % ---- z-score with training parameters (regression-type models) ----
        mu = mean(Xtr);  sd = std(Xtr);  sd(sd == 0) = 1;
        Ztr = (Xtr - mu) ./ sd;
        Zte = (Xte - mu) ./ sd;

        % 1 SVM (RBF, one-vs-one; standardization inside fitcecoc)
        [mdl, hp, innerL] = tuneAndFit('SVM', Xtr, ytr, gtr, cfg);
        pred(te, 1) = predict(mdl, Xte);
        hpRows(end+1, :) = {st.name, f, athIDs(f), 'SVM', hpString(hp), innerL}; %#ok<SAGROW>

        % 2 AdaBoostM2
        [mdl, hp, innerL] = tuneAndFit('ADA', Xtr, ytr, gtr, cfg);
        pred(te, 2) = predict(mdl, Xte);
        hpRows(end+1, :) = {st.name, f, athIDs(f), 'AdaBoostM2', hpString(hp), innerL}; %#ok<SAGROW>

        % 3 Ordinal logistic regression (proportional odds)
        B = mnrfit(Ztr, ytr, 'model', 'ordinal');
        [~, pred(te, 3)] = max(mnrval(B, Zte, 'model', 'ordinal'), [], 2);

        % 4 Multinomial logistic regression (nominal, baseline-category logits;
        %   the model that does not impose the proportional-odds constraint).
        %   Note: the likelihood-ratio test in Section 9a uses a different
        %   alternative, the non-parallel-slopes cumulative-logit model.
        Bn = mnrfit(Ztr, ytr, 'model', 'nominal');
        [~, pred(te, 4)] = max(mnrval(Bn, Zte, 'model', 'nominal'), [], 2);

        % 5 Ordinal mixed model, random athlete intercept (marginal prediction
        %   for the unseen athlete: integrates over the athlete distribution)
        [~, ~, gi] = unique(gtr);
        fitC = clmmFit(Ztr, ytr, gi, cfg.nGH, true, cfg.hasFminunc, -B(K:end), B(1:K-1));
        [~, pred(te, 5)] = max(clmmPredict(fitC, Zte), [], 2);

        % 6 Linear mixed model on continuous PLI, random athlete intercept
        %   (population-level prediction for the unseen athlete)
        vn  = featVar(sel);
        Ttr = array2table(Ztr, 'VariableNames', vn);
        Ttr.PLI = pli(tr);  Ttr.Ath = categorical(gtr);
        Tte = array2table(Zte, 'VariableNames', vn);
        % Ath is required by predict but ignored when 'Conditional' is false;
        % a training level avoids an undefined-level error.
        Tte.PLI = pli(te);  Tte.Ath = repmat(Ttr.Ath(1), sum(te), 1);
        lme = fitlme(Ttr, lmmFormula(vn, cfg.lmmInteract), 'FitMethod', 'REML');
        predCont(te, 1) = predict(lme, Tte, 'Conditional', false);
        pred(te, 6) = pliCategory(predCont(te, 1), cfg.cutPoints);

        % 7 Support vector regression on continuous PLI
        %   (response z-scored with training mean/SD so the C and epsilon
        %   search ranges are scale-free; predictions are back-transformed)
        muY = mean(pli(tr));  sdY = std(pli(tr));
        [mdl, hp, innerL] = tuneAndFit('SVR', Xtr, (pli(tr) - muY) / sdY, gtr, cfg);
        predCont(te, 2) = muY + sdY * predict(mdl, Xte);
        pred(te, 7) = pliCategory(predCont(te, 2), cfg.cutPoints);
        hpRows(end+1, :) = {st.name, f, athIDs(f), 'SVR', hpString(hp), innerL}; %#ok<SAGROW>

        selRows(end+1, :) = {st.name, f, athIDs(f), strjoin(featLabel(sel), ', ')}; %#ok<SAGROW>
        fprintf('  fold %2d/%d  athlete %2d  n = %2d  predictors: %-45s acc: %s\n', ...
            f, nAth, athIDs(f), sum(te), strjoin(featLabel(sel), ', '), ...
            sprintf('%.2f ', mean(pred(te, :) == ycat(te))));
    end

    % ---- performance ----
    M = cell(nMod, 1);
    for m = 1:nMod
        M{m} = classMetrics(ycat, pred(:, m), ath, K);
        perfRows(end+1, :) = {st.name, modelNames{m}, 100*M{m}.accMean, 100*M{m}.accSD, ...
            100*M{m}.acc, 100*M{m}.balAcc, M{m}.macroF1, M{m}.extremeErr}; %#ok<SAGROW>
        for c = 1:K
            perClassRows(end+1, :) = {st.name, modelNames{m}, classNames{c}, ...
                M{m}.sens(c), M{m}.spec(c), M{m}.prec(c), M{m}.f1(c)}; %#ok<SAGROW>
        end
    end
    for m = 1:2
        [rm, ma, r2] = regMetrics(pli, predCont(:, m));
        contRows(end+1, :) = {st.name, modelNames{5+m}, rm, ma, r2}; %#ok<SAGROW>
    end
    for a = 1:nMod-1
        for b = a+1:nMod
            [nb, nc, p] = mcnemarExact(ycat, pred(:, a), pred(:, b));
            mcnRows(end+1, :) = {st.name, modelNames{a}, modelNames{b}, nb, nc, p}; %#ok<SAGROW>
        end
    end
    freq = sum(selLog, 1);
    fprintf('  Selection frequency:');
    for j = 1:6
        fprintf(' %s %d/%d;', featLabel{j}, freq(j), nAth);
    end
    fprintf('\n');

    cv(s).name = st.name;  cv(s).pred = pred;  cv(s).predCont = predCont;
    cv(s).metrics = M;     cv(s).selLog = selLog;
end

perfTbl = cell2table(perfRows, 'VariableNames', {'Strategy','Model', ...
    'Acc_mean_across_athletes_pct','Acc_SD_across_athletes_pct','Acc_pooled_pct', ...
    'Balanced_acc_pct','Macro_F1','Low_High_confusions'});
perClassTbl = cell2table(perClassRows, 'VariableNames', {'Strategy','Model','Class', ...
    'Sensitivity','Specificity','Precision','F1'});
contTbl = cell2table(contRows, 'VariableNames', {'Strategy','Model','RMSE_AU','MAE_AU','R2'});
mcnTbl  = cell2table(mcnRows, 'VariableNames', {'Strategy','ModelA','ModelB', ...
    'A_right_B_wrong','A_wrong_B_right','Exact_McNemar_p'});
selTbl  = cell2table(selRows, 'VariableNames', {'Strategy','Fold','HeldOutAthlete','Predictors'});
hpTbl   = cell2table(hpRows, 'VariableNames', {'Strategy','Fold','HeldOutAthlete','Model', ...
    'Hyperparameters','Inner_LOAO_loss'});
freqTbl = table(featLabel', 'VariableNames', {'Variable'});
for s = 1:nStrat
    freqTbl.(matlab.lang.makeValidName(sprintf('S%d_folds_selected', s))) = sum(cv(s).selLog, 1)';
end

fprintf('\n=== LOAO performance (Table 3) ===\n');   disp(perfTbl);
fprintf('\n=== Continuous PLI prediction (LOAO) ===\n'); disp(contTbl);
fprintf('\n=== Paired exact McNemar tests ===\n');     disp(mcnTbl);
fprintf('\n=== Within-fold selection frequency ===\n'); disp(freqTbl);
fprintf('\n=== Hyperparameters and inner-loop loss by fold ===\n'); disp(hpTbl);

writetable(perfTbl,     fullfile(cfg.outDir, 'Table_3_LOAO_performance.csv'));
writetable(perClassTbl, fullfile(cfg.outDir, 'Table_S5_LOAO_per_class.csv'));
writetable(contTbl,     fullfile(cfg.outDir, 'Table_S6_LOAO_continuous_PLI.csv'));
writetable(mcnTbl,      fullfile(cfg.outDir, 'Table_S7_McNemar.csv'));
writetable(freqTbl,     fullfile(cfg.outDir, 'Table_S8_selection_frequency.csv'));
writetable(selTbl,      fullfile(cfg.outDir, 'Table_S9_selected_predictors_by_fold.csv'));
writetable(hpTbl,       fullfile(cfg.outDir, 'Table_S10_hyperparameters_by_fold.csv'));

% Session-level predictions (audit trail)
predTbl = table(ath, D.Session, pli, ycat, 'VariableNames', {'Athlete','Session','PLI','TrueClass'});
for s = 1:nStrat
    for m = 1:nMod
        predTbl.(sprintf('S%d_%s', s, modelNames{m})) = cv(s).pred(:, m);
    end
end
writetable(predTbl, fullfile(cfg.outDir, 'Table_S11_session_predictions.csv'));

%% 8b  SENSITIVITY: LINEAR MIXED MODEL WITH MAIN EFFECTS ONLY (functional form)
% Same outer folds and per-fold predictors as the primary strategy. The
% additive model omits the two-way interaction terms that let the LMM
% represent the multiplicative (duration x heart-rate intensity) structure of
% PLI. fitlme does not draw from the random stream, so all results above are
% unaffected by this block.
predAdd = nan(n, 1);  predAddCont = nan(n, 1);
for f = 1:nAth
    te = ath == athIDs(f);  tr = ~te;
    sel = find(cv(1).selLog(f, :));
    Xtr = Xcand(tr, sel);  Xte = Xcand(te, sel);
    mu = mean(Xtr);  sd = std(Xtr);  sd(sd == 0) = 1;
    Ztr = (Xtr - mu) ./ sd;  Zte = (Xte - mu) ./ sd;
    vn  = featVar(sel);
    Ttr = array2table(Ztr, 'VariableNames', vn);  Ttr.PLI = pli(tr);  Ttr.Ath = categorical(ath(tr));
    Tte = array2table(Zte, 'VariableNames', vn);  Tte.PLI = pli(te);  Tte.Ath = repmat(Ttr.Ath(1), sum(te), 1);
    lmeAdd = fitlme(Ttr, lmmFormula(vn, false), 'FitMethod', 'REML');
    predAddCont(te) = predict(lmeAdd, Tte, 'Conditional', false);
    predAdd(te)     = pliCategory(predAddCont(te), cfg.cutPoints);
end
Madd = classMetrics(ycat, predAdd, ath, K);
Mint = cv(1).metrics{6};                                   % LMM with interactions (primary strategy)
[rmA, maA, r2A] = regMetrics(pli, predAddCont);
[nbA, ncA, pA]  = mcnemarExact(ycat, cv(1).pred(:, 6), predAdd);
addTbl = table({'LMM_interactions'; 'LMM_additive'}, ...
    100*[Mint.accMean; Madd.accMean], 100*[Mint.accSD; Madd.accSD], 100*[Mint.acc; Madd.acc], ...
    100*[Mint.balAcc; Madd.balAcc], [Mint.macroF1; Madd.macroF1], [Mint.extremeErr; Madd.extremeErr], ...
    [contTbl.RMSE_AU(1); rmA], [contTbl.MAE_AU(1); maA], [contTbl.R2(1); r2A], ...
    [nbA; nbA], [ncA; ncA], [pA; pA], ...
    'VariableNames', {'Model', 'Acc_mean_across_athletes_pct', 'Acc_SD_across_athletes_pct', 'Acc_pooled_pct', ...
                      'Balanced_acc_pct', 'Macro_F1', 'Low_High_confusions', 'RMSE_AU', 'MAE_AU', 'R2', ...
                      'McNemar_interactions_right_additive_wrong', 'McNemar_interactions_wrong_additive_right', 'McNemar_p'});
fprintf('\n=== Functional-form sensitivity: LMM with vs without interaction terms (primary strategy) ===\n');
disp(addTbl);
fprintf('Exact McNemar, interactions vs additive: %d vs %d discordant, p = %.3f\n', nbA, ncA, pA);
writetable(addTbl, fullfile(cfg.outDir, 'Table_S6b_LMM_functional_form.csv'));

%% 8c  DISTANCE OF MISCLASSIFIED SESSIONS FROM THE NEAREST CUT POINT
% Supports the Results statement that most SVM misclassifications lie within
% 20 AU of a category boundary (count, percentage and median distance).
% Deterministic post-processing of the Section 8 predictions (primary
% strategy); no model is refitted.
bndRows = cell(0, 6);
for m = 1:nMod
    mis  = cv(1).pred(:, m) ~= ycat;
    dist = min(abs(pli(mis) - cfg.cutPoints(1)), abs(pli(mis) - cfg.cutPoints(2)));
    bndRows(end+1, :) = {modelNames{m}, sum(mis), sum(dist <= 20), ...
        100*mean(dist <= 20), median(dist), max([dist; NaN])}; %#ok<SAGROW>  (NaN if no errors)
end
bndTbl = cell2table(bndRows, 'VariableNames', {'Model','Misclassified', ...
    'Within_20_AU_of_cut_point','Pct_within_20_AU','Median_distance_AU','Max_distance_AU'});
fprintf('\n=== Misclassified sessions: distance to the nearest cut point (primary strategy) ===\n');
disp(bndTbl);
writetable(bndTbl, fullfile(cfg.outDir, 'Table_S16_boundary_distance.csv'));

%% 8d  CLUSTER BOOTSTRAP AND CLUSTER-LEVEL TESTS FOR PAIRWISE ACCURACY DIFFERENCES
% The exact McNemar test (Section 8) treats the 372 athlete-sessions as
% independent pairs, but sessions are clustered within training days (session
% ICC 0.53 given the predictors) and within athletes. For every pair of models:
%   - difference in pooled accuracy (percentage points);
%   - 95% percentile CI from a cluster bootstrap (cfg.nBoot resamples of
%     training days with replacement; separately of athletes);
%   - cluster-level sign-flip test: the per-cluster difference in the number
%     of correctly classified sessions is sign-permuted (Monte Carlo for days,
%     exact enumeration for the 10 athletes).
% Seeded separately so that Section 8 is unaffected.
cfg.nBoot = 10000;  cfg.nPerm = 20000;
rng(cfg.seed + 777, 'twister');
cbRows = {};
for s = 1:nStrat
    for a = 1:nMod-1
        for b = a+1:nMod
            cA = cv(s).pred(:, a) == ycat;  cB = cv(s).pred(:, b) == ycat;
            [ciD, pD] = clusterCompare(cA, cB, D.Session, cfg.nBoot, cfg.nPerm);
            [ciA, pA] = clusterCompare(cA, cB, ath,       cfg.nBoot, cfg.nPerm);
            cbRows(end+1, :) = {cfg.strategies(s).name, modelNames{a}, modelNames{b}, ...
                100*(mean(cA) - mean(cB)), ciD(1), ciD(2), pD, ciA(1), ciA(2), pA}; %#ok<SAGROW>
        end
    end
end
cbTbl = cell2table(cbRows, 'VariableNames', {'Strategy','ModelA','ModelB','Diff_pooled_acc_pct', ...
    'Day_boot_CI_low','Day_boot_CI_high','Day_signflip_p','Athlete_boot_CI_low','Athlete_boot_CI_high','Athlete_signflip_p'});
fprintf('\n=== Pairwise accuracy differences: cluster bootstrap (days, athletes) and sign-flip tests ===\n');
disp(cbTbl);
writetable(cbTbl, fullfile(cfg.outDir, 'Table_S7b_cluster_bootstrap.csv'));

%% 8e  INNER-SPLIT MRMR AGREEMENT (was selection stable within the tuning loop?)
% MRMR is applied once per outer fold on all training athletes; the inner
% leave-one-athlete-out tuning loop then uses that predictor set. Here MRMR is
% repeated on every inner training split (nine athletes minus one) and its
% top-3 set is compared with the outer-fold set. Deterministic.
agree = 0;  total = 0;  innerRows = {};
for f = 1:nAth
    tr  = ath ~= athIDs(f);
    sel = find(cv(1).selLog(f, :));
    trIDs = athIDs(athIDs ~= athIDs(f));
    for j = 1:numel(trIDs)
        inner = tr & ath ~= trIDs(j);
        idxJ  = mrmrMIQ(Xcand(inner, :), ycat(inner), cfg.mrmrBins);
        selJ  = sort(idxJ(1:3));
        same  = isequal(selJ(:)', sel(:)');
        agree = agree + same;  total = total + 1;
        innerRows(end+1, :) = {f, athIDs(f), trIDs(j), strjoin(featLabel(selJ), ', '), same}; %#ok<SAGROW>
    end
end
innerTbl = cell2table(innerRows, 'VariableNames', {'OuterFold','HeldOutAthlete', ...
    'InnerValidationAthlete','InnerMRMR_top3','SameAsOuterFold'});
fprintf('\nInner-split MRMR agreement with the outer-fold selection: %d of %d inner splits\n', agree, total);
writetable(innerTbl, fullfile(cfg.outDir, 'Table_S9b_inner_MRMR_agreement.csv'));

%% 8f  ORDINAL MODEL WITH ALL SIX CANDIDATE PREDICTORS UNDER LOAO
% Does adding the three wellness ratings to the ordinal model change its
% held-out accuracy? Same outer folds; deterministic.
predAug = nan(n, 1);
for f = 1:nAth
    te = ath == athIDs(f);  tr = ~te;
    mu = mean(Xcand(tr, :));  sd = std(Xcand(tr, :));  sd(sd == 0) = 1;
    B6 = mnrfit((Xcand(tr, :) - mu) ./ sd, ycat(tr), 'model', 'ordinal');
    [~, predAug(te)] = max(mnrval(B6, (Xcand(te, :) - mu) ./ sd, 'model', 'ordinal'), [], 2);
end
Maug = classMetrics(ycat, predAug, ath, K);
M3   = cv(1).metrics{3};                                  % ordinal, three predictors
[nbG, ncG, pG] = mcnemarExact(ycat, cv(1).pred(:, 3), predAug);
[ciG, pGd] = clusterCompare(cv(1).pred(:, 3) == ycat, predAug == ycat, D.Session, cfg.nBoot, cfg.nPerm);
augTbl = table({'Ordinal_3_predictors'; 'Ordinal_6_predictors'}, ...
    100*[M3.accMean; Maug.accMean], 100*[M3.accSD; Maug.accSD], 100*[M3.acc; Maug.acc], ...
    100*[M3.balAcc; Maug.balAcc], [M3.macroF1; Maug.macroF1], [M3.extremeErr; Maug.extremeErr], ...
    [nbG; nbG], [ncG; ncG], [pG; pG], [ciG(1); ciG(1)], [ciG(2); ciG(2)], [pGd; pGd], ...
    'VariableNames', {'Model','Acc_mean_across_athletes_pct','Acc_SD_across_athletes_pct', ...
    'Acc_pooled_pct','Balanced_acc_pct','Macro_F1','Low_High_confusions', ...
    'McNemar_3_right_6_wrong','McNemar_3_wrong_6_right','McNemar_p','Day_boot_CI_low','Day_boot_CI_high','Day_signflip_p'});
fprintf('\n=== Ordinal model with three vs all six candidate predictors (LOAO) ===\n');
disp(augTbl);
writetable(augTbl, fullfile(cfg.outDir, 'Table_S17_ordinal_all_six_predictors.csv'));

%% 8g  LEAVE-ONE-DAY-OUT EVALUATION OF THE DETERMINISTIC MODELS
% Complement to LOAO: each training day (all athletes' sessions of that day)
% is held out in turn, so the held-out sessions come from days unseen in
% training but from athletes seen in training. Within each fold: MRMR top 3 on
% the training days, standardisation, and the four models that need no tuning
% (ordinal, multinomial, CLMM with marginal prediction, LMM with population-
% level prediction). Deterministic.
days   = unique(D.Session);  nDays = numel(days);
lodoNames = {'OrdinalLogistic','Multinomial','CLMM','LMM_binned'};
predLODO = nan(n, 4);  predLODOcont = nan(n, 1);  selLODO = false(nDays, 6);
for d = 1:nDays
    te = D.Session == days(d);  tr = ~te;
    idx = mrmrMIQ(Xcand(tr, :), ycat(tr), cfg.mrmrBins);  sel = sort(idx(1:3));
    selLODO(d, sel) = true;
    Xtr = Xcand(tr, sel);  Xte = Xcand(te, sel);
    mu = mean(Xtr);  sd = std(Xtr);  sd(sd == 0) = 1;
    Ztr = (Xtr - mu) ./ sd;  Zte = (Xte - mu) ./ sd;
    Bo = mnrfit(Ztr, ycat(tr), 'model', 'ordinal');
    [~, predLODO(te, 1)] = max(mnrval(Bo, Zte, 'model', 'ordinal'), [], 2);
    Bn = mnrfit(Ztr, ycat(tr), 'model', 'nominal');
    [~, predLODO(te, 2)] = max(mnrval(Bn, Zte, 'model', 'nominal'), [], 2);
    [~, ~, gi] = unique(ath(tr));
    fitC = clmmFit(Ztr, ycat(tr), gi, cfg.nGH, true, cfg.hasFminunc, -Bo(K:end), Bo(1:K-1));
    [~, predLODO(te, 3)] = max(clmmPredict(fitC, Zte), [], 2);
    vn  = featVar(sel);
    Ttr = array2table(Ztr, 'VariableNames', vn);  Ttr.PLI = pli(tr);  Ttr.Ath = categorical(ath(tr));
    Tte = array2table(Zte, 'VariableNames', vn);  Tte.PLI = pli(te);  Tte.Ath = repmat(Ttr.Ath(1), sum(te), 1);
    lme = fitlme(Ttr, lmmFormula(vn, cfg.lmmInteract), 'FitMethod', 'REML');
    predLODOcont(te) = predict(lme, Tte, 'Conditional', false);
    predLODO(te, 4)  = pliCategory(predLODOcont(te), cfg.cutPoints);
end
lodoRows = {};
for m = 1:4
    Mm = classMetrics(ycat, predLODO(:, m), D.Session, K);   % accMean/SD here are across DAYS
    lodoRows(end+1, :) = {lodoNames{m}, 100*Mm.acc, 100*Mm.accMean, 100*Mm.accSD, ...
        100*Mm.balAcc, Mm.macroF1, Mm.extremeErr}; %#ok<SAGROW>
end
lodoTbl = cell2table(lodoRows, 'VariableNames', {'Model','Acc_pooled_pct', ...
    'Acc_mean_across_days_pct','Acc_SD_across_days_pct','Balanced_acc_pct','Macro_F1','Low_High_confusions'});
[rmL, maL, r2L] = regMetrics(pli, predLODOcont);
fprintf('\n=== Leave-one-day-out (%d days), deterministic models ===\n', nDays);
disp(lodoTbl);
fprintf('LMM continuous PLI under leave-one-day-out: RMSE %.1f AU, MAE %.1f AU, R2 %.3f\n', rmL, maL, r2L);
fprintf('Leave-one-day-out MRMR selection frequency (days): %s\n', ...
    strjoin(arrayfun(@(j) sprintf('%s %d/%d', featLabel{j}, sum(selLODO(:, j)), nDays), 1:6, 'UniformOutput', false), '; '));
lodoTbl.RMSE_AU = [NaN; NaN; NaN; rmL];  lodoTbl.MAE_AU = [NaN; NaN; NaN; maL];  lodoTbl.R2 = [NaN; NaN; NaN; r2L];
writetable(lodoTbl, fullfile(cfg.outDir, 'Table_S18_leave_one_day_out.csv'));

%% 9  INFERENCE MODELS ON THE FULL ANALYTIC SAMPLE (fixed sRPE, %HRmax, TD)
infSel = [1 5 6];
Zall = (Xcand(:, infSel) - mean(Xcand(:, infSel))) ./ std(Xcand(:, infSel));
infNames = featLabel(infSel);

% 9a Ordinal logistic regression and Brant test of proportional odds.
%    Brant (1990): a separate binary logistic regression is fitted at each
%    cut point (Y > Low, Y > Medium) and a Wald test, using Brant's
%    covariance between the two fits, tests whether the slopes are equal
%    (omnibus df = 3; per predictor df = 1). P-values from the chi-square
%    distribution and from a parametric bootstrap (cfg.nBootPO data sets
%    simulated from the fitted proportional-odds model; own random stream).
%    The likelihood-ratio test against the non-parallel-slopes cumulative-
%    logit model (v2.3) is printed for information only: that model gives
%    negative fitted probabilities for some sessions, so the test is not valid.
[B0, dev0, st0] = mnrfit(Zall, ycat, 'model', 'ordinal');
fitNP = cumlogitNPFit(Zall, ycat, cfg.hasFminunc, -B0(K:end), B0(1:K-1));
poLR_NP = 2 * (fitNP.logLik - (-dev0/2));          % ll(PO) = -deviance/2
fprintf(['\n(Information only, not reported) LR test against the non-parallel-slopes cumulative-logit model: ' ...
         'LR = %.2f; exitflag %d, max |gradient| %.2e; sessions with a negative fitted probability: %d\n'], ...
    poLR_NP, fitNP.exitflag, fitNP.maxGrad, fitNP.nNegative);
ordTbl = table([{'theta_1 (Low|Medium)'; 'theta_2 (Medium|High)'}; infNames'], ...
    [B0(1:K-1); -B0(K:end)], st0.se, st0.p, ...
    'VariableNames', {'Term','Estimate','SE','p'});
fprintf('\nOrdinal logistic regression (z-scored predictors; beta > 0 = higher category):\n');
disp(ordTbl);

cfg.nBootPO = 2000;
[Wobs, dObs, VdObs, Bbin, SEbin, warnObs] = brantWald(Zall, ycat, K);
if ~isempty(warnObs), fprintf('Note: glmfit warning in the Brant fits: %s\n', warnObs); end
binTbl = table(infNames', -B0(K:end), Bbin(2:end, 1), SEbin(2:end, 1), Bbin(2:end, 2), SEbin(2:end, 2), ...
    'VariableNames', {'Predictor','Ordinal_beta','Beta_Y_gt_Low','SE_Y_gt_Low', ...
                      'Beta_Y_gt_Medium','SE_Y_gt_Medium'});
fprintf('Binary logistic regressions at each cut point (Brant test):\n');
disp(binTbl);
rsPO = RandStream('mt19937ar', 'Seed', cfg.seed);  % same stream as rng(cfg.seed,'twister')
Pfit = mnrval(B0, Zall, 'model', 'ordinal');
cumP = cumsum(Pfit, 2);
Wb = nan(cfg.nBootPO, 1 + numel(infSel));  nSkipPO = 0;  nWarnPO = 0;
for b = 1:cfg.nBootPO
    ys = 1 + sum(rand(rsPO, n, 1) > cumP(:, 1:K-1), 2);
    if numel(unique(ys)) < K, nSkipPO = nSkipPO + 1; continue; end
    [Ws, ds, Vds, ~, ~, wmsg] = brantWald(Zall, ys, K);
    nWarnPO = nWarnPO + ~isempty(wmsg);
    Wb(b, :) = [Ws, (ds.^2 ./ diag(Vds))'];
end
Wb = Wb(~any(isnan(Wb), 2), :);
poStat = [Wobs, (dObs.^2 ./ diag(VdObs))'];
poDf   = [(K-2) * numel(infSel), ones(1, numel(infSel))];
poP    = 1 - chi2cdf(poStat, poDf);
poPb   = (1 + sum(Wb >= poStat, 1)) / (size(Wb, 1) + 1);
brantTbl = table([{'Omnibus (all predictors)'}; infNames'], poStat', poDf', poP', poPb', ...
    'VariableNames', {'Test','Wald_chi2','df','p_asymptotic','p_bootstrap'});
fprintf('Brant test of proportional odds (H0: equal slopes at both cut points):\n');
disp(brantTbl);
fprintf('Bootstrap: %d valid data sets (%d skipped: a category empty; %d with a glmfit warning)\n', ...
    size(Wb, 1), nSkipPO, nWarnPO);
writetable(brantTbl, fullfile(cfg.outDir, 'Table_S12b_brant_test.csv'));
writetable(ordTbl, fullfile(cfg.outDir, 'Table_S12_ordinal_logistic.csv'));

% 9b Ordinal mixed model (random athlete intercept) vs fixed-effects model
[~, ~, gAll] = unique(ath);
fitNoRE = clmmFit(Zall, ycat, gAll, cfg.nGH, false, cfg.hasFminunc, -B0(K:end), B0(1:K-1));
fitRE   = clmmFit(Zall, ycat, gAll, cfg.nGH, true,  cfg.hasFminunc, -B0(K:end), B0(1:K-1));
chk = max(abs([fitNoRE.theta; fitNoRE.beta] - [B0(1:K-1); -B0(K:end)]));
fprintf('\nCheck: custom ordinal likelihood vs mnrfit, max |difference| = %.2e (should be < 1e-3)\n', chk);
fprintf('Check: -2 log-likelihood custom = %.3f, mnrfit deviance = %.3f\n', -2*fitNoRE.logLik, dev0);
reLR = max(0, 2*(fitRE.logLik - fitNoRE.logLik));
reP  = 0.5 * (1 - chi2cdf(reLR, 1));          % boundary-corrected (50:50 chi-bar-square)
clmmTbl = table([{'theta_1 (Low|Medium)'; 'theta_2 (Medium|High)'}; infNames'; {'SD athlete intercept'}], ...
    [fitRE.theta; fitRE.beta; fitRE.sigma], [fitRE.seTheta; fitRE.seBeta; fitRE.seSigma], ...
    'VariableNames', {'Term','Estimate','SE'});
clmmTbl.z = clmmTbl.Estimate ./ clmmTbl.SE;
clmmTbl.p = 2 * normcdf(-abs(clmmTbl.z));
clmmTbl.p(end) = reP;                          % variance component: LR test
fprintf('\nOrdinal mixed model (random athlete intercept, %d-node Gauss-Hermite):\n', cfg.nGH);
disp(clmmTbl);
fprintf('Random-intercept LR test vs fixed-effects ordinal model: LR = %.2f, p = %.3f (boundary-corrected)\n', ...
    reLR, reP);
writetable(clmmTbl, fullfile(cfg.outDir, 'Table_S13_ordinal_mixed_model.csv'));

% 9c Linear mixed models on continuous PLI with crossed athlete and session
%    random intercepts; intraclass correlations
Tinf = array2table(Zall, 'VariableNames', featVar(infSel));
Tinf.PLI = pli;  Tinf.Ath = categorical(ath);  Tinf.Session = categorical(D.Session);
lmeNull = fitlme(Tinf, 'PLI ~ 1 + (1|Ath) + (1|Session)', 'FitMethod', 'REML');
lmeFull = fitlme(Tinf, [lmmFormula(featVar(infSel), cfg.lmmInteract) ' + (1|Session)'], 'FitMethod', 'REML');
[vNull, eNull] = reVariances(lmeNull);
[vFull, eFull] = reVariances(lmeFull);
iccTbl = table({'Null (intercept only)'; 'With predictors'}, ...
    [vNull(1); vFull(1)], [vNull(2); vFull(2)], [eNull; eFull], ...
    [vNull(1)/(sum(vNull)+eNull); vFull(1)/(sum(vFull)+eFull)], ...
    [vNull(2)/(sum(vNull)+eNull); vFull(2)/(sum(vFull)+eFull)], ...
    'VariableNames', {'Model','Var_athlete','Var_session','Var_residual','ICC_athlete','ICC_session'});
fprintf('\nLinear mixed model, continuous PLI (crossed athlete and session intercepts):\n');
disp(lmeFull);
disp(iccTbl);
writetable(iccTbl, fullfile(cfg.outDir, 'Table_S14_LMM_variance_components.csv'));
writetable(asTable(lmeFull.Coefficients), fullfile(cfg.outDir, 'Table_S15_LMM_fixed_effects.csv'));

%% 10  FIGURES (numbering follows the manuscript)
% Figure 1 (pipeline schematic) is drawn separately.

% Figure 2: Spearman correlation matrix (lower triangle; upper triangle masked)
lbl = ['PLI', featLabel];
nL  = numel(lbl);
fig = figure('Color', 'w', 'Units', 'inches', 'Position', [1 1 8 7]);
hIm = imagesc(rhoAll); set(hIm, 'AlphaData', tril(true(nL)));
set(gca, 'Color', 'w');
colormap(gray); cb = colorbar; cb.Label.String = 'Spearman \rho'; clim([-1 1]);
set(gca, 'XTick', 1:nL, 'XTickLabel', lbl, 'YTick', 1:nL, 'YTickLabel', lbl, ...
    'TickLength', [0 0], 'FontSize', 11);
for i = 1:nL
    for j = 1:i
        if i == j, star = ''; else, star = sigStars(pAll(i, j)); end
        text(j, i, sprintf('%.2f%s', rhoAll(i, j), star), ...
            'HorizontalAlignment', 'center', 'FontSize', 11, ...
            'Color', ifelse(rhoAll(i, j) < -0.2, 'w', 'k'));
    end
end
title(sprintf('Spearman correlation matrix (n = %d athlete-sessions)', n));
exportgraphics(fig, fullfile(cfg.outDir, 'Figure2_Spearman_matrix.png'), 'Resolution', 300);

% Figure 3: descriptive MRMR ranking
fig = figure('Color', 'w', 'Units', 'inches', 'Position', [1 1 7 5.5]);
bar(mrmrScore, 'FaceColor', [0.2 0.6 0.8]);
set(gca, 'XTick', 1:6, 'XTickLabel', featLabel(mrmrIdx), 'FontSize', 11);
xtickangle(45); grid on;
ylabel('MRMR score');
title({'MRMR ranking, full analytic sample (descriptive)', ...
       'rank 1: relevance I(x;y) in nats; ranks 2-6: mutual-information quotient'}, ...
       'FontWeight', 'normal');
exportgraphics(fig, fullfile(cfg.outDir, 'Figure3_MRMR_ranking.png'), 'Resolution', 300);

% Figures 4-5 and S1-S5: confusion matrices, primary strategy.
% Categorical labels keep the ordinal order Low -> Medium -> High
% (confusionchart sorts cell-array labels alphabetically).
figNames = {'Figure4_SVM','Figure5_AdaBoostM2','FigureS1_OrdinalLogistic', ...
            'FigureS2_Multinomial','FigureS3_CLMM','FigureS4_LMM_binned','FigureS5_SVR_binned'};
yTrueCat = categorical(ycat, 1:K, classNames);
for m = 1:nMod
    yPredCat = categorical(cv(1).pred(:, m), 1:K, classNames);
    fig = figure('Color', 'w', 'Units', 'inches', 'Position', [1 1 7 6]);
    cc = confusionchart(yTrueCat, yPredCat, ...
        'RowSummary', 'row-normalized', 'ColumnSummary', 'column-normalized');
    cc.Title  = sprintf('LOAO-CV: %s (%s)', strrep(modelNames{m}, '_', ' '), cv(1).name);
    cc.XLabel = 'Predicted class';  cc.YLabel = 'True class';
    exportgraphics(fig, fullfile(cfg.outDir, [figNames{m} '_confusion.png']), 'Resolution', 300);
end

% Figure 6: continuous predictions vs observed PLI, primary strategy
classCols = [0.27 0.00 0.55; 0.10 0.70 0.70; 0.90 0.75 0.10];
fig = figure('Color', 'w', 'Units', 'inches', 'Position', [1 1 11 5]);
tl = tiledlayout(fig, 1, 2);
contLabel = {'LMM (population-level)', 'SVR'};
for m = 1:2
    ax = nexttile(tl); hold(ax, 'on');
    xp = cv(1).predCont(:, m);
    for c = 1:K
        scatter(ax, xp(ycat == c), pli(ycat == c), 18, classCols(c, :), 'filled', ...
            'MarkerFaceAlpha', 0.65, 'DisplayName', classNames{c});
    end
    lim = [0, max([pli; xp]) * 1.05];
    plot(ax, lim, lim, 'k--', 'HandleVisibility', 'off');
    xline(ax, cfg.cutPoints, ':', 'HandleVisibility', 'off');
    yline(ax, cfg.cutPoints, ':', 'HandleVisibility', 'off');
    xlim(ax, lim); ylim(ax, lim); axis(ax, 'square'); grid(ax, 'on');
    xlabel(ax, 'Predicted PLI (AU)'); ylabel(ax, 'Observed PLI (AU)'); title(ax, contLabel{m});
    if m == 1, legend(ax, 'Location', 'northwest'); end
end
exportgraphics(fig, fullfile(cfg.outDir, 'Figure6_continuous_predictions.png'), 'Resolution', 300);

%% SAVE
save(fullfile(cfg.outDir, 'MSOC_results.mat'), 'cfg', 'modelNames', 'classNames', ...
    'flowTbl', 'descTbl', 'corrTbl', 'rhoAll', 'pAll', ...
    'constructTbl', 'mrmrTbl', 'perfTbl', 'perClassTbl', 'contTbl', 'mcnTbl', 'freqTbl', ...
    'selTbl', 'hpTbl', 'ordTbl', 'brantTbl', 'binTbl', 'poDf', 'poP', 'poPb', 'clmmTbl', 'reLR', 'reP', ...
    'iccTbl', 'cv', 'predTbl', 'addTbl', 'predAdd', 'predAddCont', 'bndTbl', ...
    'cutSensTbl', 'cbTbl', 'innerTbl', 'augTbl', 'predAug', 'lodoTbl', 'predLODO', 'predLODOcont', ...
    'fitNP', 'poLR_NP');
v = ver;
fid = fopen(fullfile(cfg.outDir, 'software_versions.txt'), 'w');
fprintf(fid, 'MATLAB %s\nPipeline %s, seed %d, %d Bayesian evaluations per fold, parallel = %d\n\n', ...
    version, cfg.version, cfg.seed, cfg.bayesEvals, cfg.useParallel);
fprintf(fid, '%s %s %s\n', string({v.Name; v.Version; v.Release}));
fclose(fid);
fprintf('\nDone in %.1f min. Outputs in %s\n', toc(tStart)/60, cfg.outDir);
diary off;


%% ========================================================================
%  LOCAL FUNCTIONS (MATLAB requires these at the end of a script file)
% =========================================================================

function T = readSheet(file, sheet, vars)
% Read the named columns of a sheet, keeping the original header text.
opts = detectImportOptions(file, 'Sheet', sheet, 'VariableNamingRule', 'preserve');
missingVars = setdiff(vars, opts.VariableNames);
if ~isempty(missingVars)
    error('MSOC:columns', 'Sheet "%s" is missing column(s): %s', sheet, strjoin(missingVars, ', '));
end
opts.SelectedVariableNames = vars;
T = readtable(file, opts);
end

function x = toDouble(x)
% Numeric column vector from numeric, logical, cell or string input.
if iscell(x) || isstring(x)
    x = str2double(string(x));
else
    x = double(x);
end
x = x(:);
end

function m = toMinutes(x)
% Minutes from an Excel time cell imported as duration, datetime or number.
if isduration(x)
    m = minutes(x);
elseif isdatetime(x)
    m = minutes(x - dateshift(x, 'start', 'day'));
else
    m = toDouble(x);
    if median(m, 'omitnan') < 1          % Excel stores times as fractions of a day
        m = m * 1440;
    end
end
m = m(:);
end

function c = pliCategory(x, cut)
% 1 = Low (x <= cut(1)), 2 = Medium (cut(1) < x < cut(2)), 3 = High (x >= cut(2)).
% Also applied to continuous LMM/SVR predictions, so they are binned with
% exactly the same boundaries as the observed PLI.
c = 2 * ones(size(x));
c(x <= cut(1)) = 1;
c(x >= cut(2)) = 3;
end

function [r, p, df] = rmcorr(x, y, g)
% Repeated-measures correlation (Bakdash & Marusich, 2017): correlation of
% within-subject-centred values, df = N - subjects - 1.
[~, ~, gi] = unique(g);
mx = accumarray(gi, x, [], @mean);
my = accumarray(gi, y, [], @mean);
xc = x - mx(gi);
yc = y - my(gi);
r  = sum(xc .* yc) / sqrt(sum(xc.^2) * sum(yc.^2));
df = numel(x) - max(gi) - 1;
t  = r * sqrt(df / (1 - r^2));
p  = 2 * tcdf(-abs(t), df);
end

function padj = holm(p)
% Holm step-down adjusted p-values.
p = p(:);  m = numel(p);
[ps, order] = sort(p);
adj = min(1, cummax((m - (1:m)' + 1) .* ps));
padj = zeros(m, 1);
padj(order) = adj;
end

function s = sigStars(p)
if p < 0.001,     s = '***';
elseif p < 0.01,  s = '**';
elseif p < 0.05,  s = '*';
else,             s = '';
end
end

function out = ifelse(cond, a, b)
if cond, out = a; else, out = b; end
end

function M = classMetrics(yt, yp, ath, K)
% Confusion matrix (rows = true, columns = predicted) and summary metrics.
C = confusionmat(yt, yp, 'Order', 1:K);
n = sum(C(:));
ids = unique(ath);
accAth = arrayfun(@(a) mean(yp(ath == a) == yt(ath == a)), ids);
sens = diag(C) ./ sum(C, 2);
prec = diag(C) ./ max(sum(C, 1)', 1);
spec = zeros(K, 1);
for k = 1:K
    TP = C(k, k);  FN = sum(C(k, :)) - TP;  FP = sum(C(:, k)) - TP;
    TN = n - TP - FN - FP;
    spec(k) = TN / (TN + FP);
end
f1 = 2 * prec .* sens ./ max(prec + sens, eps);
M = struct('C', C, 'acc', trace(C)/n, 'accAth', accAth, 'accMean', mean(accAth), ...
    'accSD', std(accAth), 'balAcc', mean(sens), 'macroF1', mean(f1), 'sens', sens, ...
    'spec', spec, 'prec', prec, 'f1', f1, 'extremeErr', C(1, K) + C(K, 1));
end

function [rmse, mae, r2] = regMetrics(y, yhat)
e = y - yhat;
rmse = sqrt(mean(e.^2));
mae  = mean(abs(e));
r2   = 1 - sum(e.^2) / sum((y - mean(y)).^2);
end

function [b, c, p] = mcnemarExact(yt, p1, p2)
% Exact (binomial) McNemar test on paired correct/incorrect outcomes.
c1 = p1 == yt;  c2 = p2 == yt;
b = sum(c1 & ~c2);  c = sum(~c1 & c2);
if b + c == 0
    p = 1;
else
    p = min(1, 2 * binocdf(min(b, c), b + c, 0.5));
end
end

function [order, score] = mrmrMIQ(X, y, q)
% Minimum-redundancy maximum-relevance ranking, mutual information quotient
% criterion (Ding & Peng, 2005). Relevance = I(x; y). Redundancy = mean
% I(x; s) over already-selected predictors s. Predictors with more than q
% distinct values are discretized into q equal-frequency bins computed from
% the data passed in (the training fold during cross-validation). Natural
% logarithms. score(k) is the criterion value of the k-th ranked predictor.
p = size(X, 2);
B = zeros(size(X));
for j = 1:p
    x = X(:, j);
    if numel(unique(x)) <= q
        [~, ~, B(:, j)] = unique(x);
    else
        e = unique(quantile(x, (1:q-1) / q));
        B(:, j) = 1 + sum(x > e(:)', 2);
    end
end
[~, ~, yc] = unique(y);
rel = zeros(1, p);
for j = 1:p
    rel(j) = mutualInfo(B(:, j), yc);
end
order = zeros(1, p);  score = zeros(1, p);
[score(1), order(1)] = max(rel);
remaining = setdiff(1:p, order(1));
for k = 2:p
    miq = zeros(1, numel(remaining));
    for i = 1:numel(remaining)
        j = remaining(i);
        red = mean(arrayfun(@(s) mutualInfo(B(:, j), B(:, s)), order(1:k-1)));
        miq(i) = rel(j) / max(red, eps);
    end
    [score(k), ii] = max(miq);
    order(k) = remaining(ii);
    remaining(ii) = [];
end
end

function I = mutualInfo(a, b)
% Mutual information (nats) between two positive-integer-coded vectors.
P  = accumarray([a(:) b(:)], 1);
P  = P / sum(P(:));
PaPb = sum(P, 2) * sum(P, 1);
nz = P > 0;
I  = sum(P(nz) .* log(P(nz) ./ PaPb(nz)));
end

function [mdl, best, minObj] = tuneAndFit(type, X, y, g, cfg)
% Bayesian optimization of hyperparameters with an inner leave-one-athlete-
% out loop over the training athletes, then refit on all training data.
%
% The search starts from a FIXED log-spaced design ('InitialX'), identical
% in every fold, so that every fold evaluates the same sensible region
% before the acquisition function takes over. The remaining evaluations use
% expected-improvement-plus. The chosen point is the one with the lowest
% OBSERVED inner loss (XAtMinObjective), and that loss is returned so it can
% be reported per fold (Table S10).
%
% Search ranges (Standardize = true, so C and KernelScale are in standardized
% units): C 1e-3..1e3, KernelScale 1e-3..1e3, AdaBoostM2 cycles 10..500,
% learning rate 1e-3..1, max splits 1..20, SVR epsilon 1e-3..1e2 robust SDs.
C  = optimizableVariable('C',  [1e-3 1e3], 'Transform', 'log');
KS = optimizableVariable('KS', [1e-3 1e3], 'Transform', 'log');
switch type
    case 'SVM'
        vars = [C, KS];
        [a, b] = ndgrid([0.1 10 1000], [0.3 3 30]);
        init = table(a(:), b(:), 'VariableNames', {'C', 'KS'});
    case 'ADA'
        vars = [optimizableVariable('NumCycles', [10 500], 'Type', 'integer', 'Transform', 'log'), ...
                optimizableVariable('LearnRate', [1e-3 1],   'Transform', 'log'), ...
                optimizableVariable('MaxSplits', [1 20],     'Type', 'integer', 'Transform', 'log')];
        [a, b, c] = ndgrid([30 300], [0.03 0.3], [2 10]);
        init = table(a(:), b(:), c(:), 'VariableNames', {'NumCycles', 'LearnRate', 'MaxSplits'});
    case 'SVR'
        s = iqr(y) / 1.349;
        vars = [C, KS, optimizableVariable('Eps', [1e-3 1e2] * s, 'Transform', 'log')];
        [a, b, c] = ndgrid([1 100], [1 10], [0.01 0.1] * s);
        init = table(a(:), b(:), c(:), 'VariableNames', {'C', 'KS', 'Eps'});
end
obj = @(x) innerLoss(type, x, X, y, g);
res = bayesopt(obj, vars, ...
    'InitialX', init, ...
    'MaxObjectiveEvaluations', max(cfg.bayesEvals, height(init)), ...
    'AcquisitionFunctionName', 'expected-improvement-plus', ...
    'IsObjectiveDeterministic', true, ...
    'UseParallel', cfg.useParallel, ...
    'Verbose', 0, 'PlotFcn', []);
best   = res.XAtMinObjective;
minObj = res.MinObjective;
mdl    = fitOne(type, best, X, y);
end

function L = innerLoss(type, x, X, y, g)
% Inner leave-one-athlete-out loss: misclassification rate (classifiers) or
% log(1 + MSE) (SVR).
ug = unique(g);
err = 0;
for i = 1:numel(ug)
    te = g == ug(i);
    mdl = fitOne(type, x, X(~te, :), y(~te));
    yh  = predict(mdl, X(te, :));
    if strcmp(type, 'SVR')
        err = err + sum((yh - y(te)).^2);
    else
        err = err + sum(yh ~= y(te));
    end
end
L = err / numel(y);
if strcmp(type, 'SVR'), L = log1p(L); end
end

function mdl = fitOne(type, x, X, y)
switch type
    case 'SVM'
        t = templateSVM('KernelFunction', 'rbf', 'BoxConstraint', x.C, ...
            'KernelScale', x.KS, 'Standardize', true);
        mdl = fitcecoc(X, y, 'Learners', t, 'Coding', 'onevsone', 'ClassNames', [1; 2; 3]);
    case 'ADA'
        % AdaBoostM2 (adaptive boosting for multiclass; Freund & Schapire,
        % 1997). It may stop before NumLearningCycles when the pseudo-loss
        % of the last learner is not positive (MATLAB warning); this is
        % benign and means the ensemble fits the training data perfectly.
        t = templateTree('MaxNumSplits', x.MaxSplits);
        mdl = fitcensemble(X, y, 'Method', 'AdaBoostM2', 'Learners', t, ...
            'NumLearningCycles', x.NumCycles, 'LearnRate', x.LearnRate, 'ClassNames', [1; 2; 3]);
    case 'SVR'
        mdl = fitrsvm(X, y, 'KernelFunction', 'gaussian', 'BoxConstraint', x.C, ...
            'KernelScale', x.KS, 'Epsilon', x.Eps, 'Standardize', true);
end
end

function s = hpString(x)
v = x.Properties.VariableNames;
parts = cell(1, numel(v));
for i = 1:numel(v)
    parts{i} = sprintf('%s=%.4g', v{i}, double(x.(v{i})));
end
s = strjoin(parts, '; ');
end

function f = lmmFormula(v, interact)
% 'PLI ~ 1 + a + b + c [+ a:b + a:c + b:c] + (1|Ath)'
terms = v;
if interact && numel(v) > 1
    pr = nchoosek(1:numel(v), 2);
    for i = 1:size(pr, 1)
        terms{end+1} = [v{pr(i, 1)} ':' v{pr(i, 2)}]; %#ok<AGROW>
    end
end
f = ['PLI ~ 1 + ' strjoin(terms, ' + ') ' + (1|Ath)'];
end

function [v, e] = reVariances(lme)
% Random-intercept variances in the order the grouping terms appear in the
% formula, and the residual variance.
[psi, e] = covarianceParameters(lme);
v = cellfun(@(p) p(1, 1), psi(:));
end

function T = asTable(x)
% Coefficient tables are dataset arrays in some releases and tables in others.
if istable(x)
    T = x;
elseif isa(x, 'dataset')
    T = dataset2table(x);
else
    T = struct2table(struct(x));
end
end

% ---- Cumulative-logit mixed model with a random intercept (CLMM) ----------
%   P(Y <= k | x, u) = F(theta_k - x*beta - u),  u ~ N(0, sigma^2),  F = logistic
%   Marginal likelihood by Gauss-Hermite quadrature; unconstrained
%   parameters: [theta_1; log(theta_2 - theta_1); ...; beta; log(sigma)].

function [ci, p] = clusterCompare(cA, cB, cl, nBoot, nPerm)
% Cluster bootstrap 95% CI for the difference in pooled accuracy (A - B,
% percentage points) and a cluster-level sign-flip p-value. cA, cB: logical
% correct/incorrect per session; cl: cluster label per session.
[~, ~, gi] = unique(cl);
nc  = max(gi);
sA  = accumarray(gi, double(cA));  sB = accumarray(gi, double(cB));  nn = accumarray(gi, 1);
draws = randi(nc, nBoot, nc);
diffs = 100 * (sum(sA(draws), 2) - sum(sB(draws), 2)) ./ sum(nn(draws), 2);
ci  = prctile(diffs, [2.5 97.5]);
d   = sA - sB;  obs = sum(d);
if nc <= 12
    signs = 2 * (dec2bin(0:2^nc-1) - '0') - 1;          % exact enumeration
    dist  = signs * d;
else
    dist  = (2 * (rand(nPerm, nc) < 0.5) - 1) * d;
end
p = mean(abs(dist) >= abs(obs) - 1e-9);
end

function [W, d, Vd, B, SE, warnMsg] = brantWald(Z, y, K)
% Brant (1990) Wald test of equal slopes across the K-1 cumulative binary
% logits. Returns the omnibus statistic W, the slope differences d (first
% cut point minus each later one), their covariance Vd, the binary-logit
% coefficients B ((p+1) x (K-1), intercept first) and their standard errors.
[n, p] = size(Z);
X = [ones(n, 1) Z];
q = p + 1;  J = K - 1;
B = zeros(q, J);  PI = zeros(n, J);  Ainv = cell(1, J);
s1 = warning('off', 'stats:glmfit:IterationLimit');
s2 = warning('off', 'stats:glmfit:PerfectSeparation');
s3 = warning('off', 'stats:glmfit:BadScaling');
lastwarn('');
for j = 1:J
    B(:, j)  = glmfit(Z, double(y > j), 'binomial', 'link', 'logit');
    PI(:, j) = 1 ./ (1 + exp(-X * B(:, j)));
    Ainv{j}  = inv(X' * (X .* (PI(:, j) .* (1 - PI(:, j)))));
end
warnMsg = lastwarn;
warning(s1); warning(s2); warning(s3);
V = zeros(J*q);
for j = 1:J
    for l = 1:J
        a = min(j, l);  c = max(j, l);
        w = PI(:, c) - PI(:, a) .* PI(:, c);          % P(Y > c) - pi_j * pi_l
        V((j-1)*q + (1:q), (l-1)*q + (1:q)) = Ainv{j} * (X' * (X .* w)) * Ainv{l};
    end
end
SE  = reshape(sqrt(diag(V)), q, J);
idx = reshape((2:q)' + (0:J-1)*q, [], 1);         % slopes only
bs  = reshape(B(2:end, :), [], 1);
Vs  = V(idx, idx);
Dm  = zeros((J-1)*p, J*p);
for j = 1:J-1
    Dm((j-1)*p + (1:p), 1:p)     = eye(p);
    Dm((j-1)*p + (1:p), j*p + (1:p)) = -eye(p);
end
d  = Dm * bs;
Vd = Dm * Vs * Dm';
W  = d' * (Vd \ d);
end

function fit = cumlogitNPFit(X, y, useFminunc, beta0, theta0)
% Cumulative-logit model with threshold-specific slopes (the non-parallel-
% slopes alternative of the proportional-odds test), maximum likelihood.
% logit P(Y <= k) = theta_k - x*beta_k, k = 1..K-1. Started from the
% proportional-odds estimates. Fitted probabilities can be negative for some
% x under this model; the likelihood floors them at 1e-12 and the number of
% such sessions at the optimum is reported.
K = max(y);  p = size(X, 2);
par0 = [theta0(:); repmat(beta0(:), K-1, 1)];
f = @(par) cumlogitNPNLL(par, X, y, K);
if useFminunc
    opt = optimoptions('fminunc', 'Algorithm', 'quasi-newton', 'Display', 'off', ...
        'MaxFunctionEvaluations', 4e4, 'MaxIterations', 4e3, ...
        'OptimalityTolerance', 1e-8, 'StepTolerance', 1e-10);
    [par, nll, exitflag] = fminunc(f, par0, opt);
else
    opt = optimset('Display', 'off', 'MaxFunEvals', 8e4, 'MaxIter', 8e4, 'TolFun', 1e-10, 'TolX', 1e-10);
    par = fminsearch(f, par0, opt);
    [par, nll, exitflag] = fminsearch(f, par, opt);
end
g = zeros(numel(par), 1);  h = 1e-5 * max(1, abs(par));
for i = 1:numel(par)
    e = zeros(numel(par), 1);  e(i) = h(i);
    g(i) = (f(par + e) - f(par - e)) / (2 * h(i));
end
[~, P] = cumlogitNPNLL(par, X, y, K);
fit.theta = par(1:K-1);  fit.beta = reshape(par(K:end), p, K-1);
fit.logLik = -nll;  fit.exitflag = exitflag;  fit.maxGrad = max(abs(g));
fit.nNegative = sum(any(P < 0, 2));
end

function [nll, P] = cumlogitNPNLL(par, X, y, K)
p = size(X, 2);  n = size(X, 1);
theta = par(1:K-1);  B = reshape(par(K:end), p, K-1);
C = 1 ./ (1 + exp(-(theta(:)' - X * B)));           % n x (K-1) cumulative probabilities
C = [C, ones(n, 1)];
P = [C(:, 1), diff(C, 1, 2)];                        % n x K category probabilities
Py = P(sub2ind(size(P), (1:n)', y(:)));
nll = -sum(log(max(Py, 1e-12)));
end

function fit = clmmFit(X, y, g, nq, useRE, useFminunc, beta0, theta0)
K = max(y);  p = size(X, 2);
par0 = [theta0(1); log(max(diff(theta0), 1e-3)); beta0(:)];
if useRE, par0 = [par0; log(0.5)]; end
[z, w] = ghNodes(nq);
f = @(par) clmmNLL(par, X, y, g, K, z, w, useRE);
if useFminunc
    opt = optimoptions('fminunc', 'Algorithm', 'quasi-newton', 'Display', 'off', ...
        'MaxFunctionEvaluations', 2e4, 'MaxIterations', 2e3, ...
        'OptimalityTolerance', 1e-8, 'StepTolerance', 1e-10);
    [par, ~, exitflag] = fminunc(f, par0, opt);
else
    opt = optimset('Display', 'off', 'MaxFunEvals', 4e4, 'MaxIter', 4e4, ...
        'TolFun', 1e-10, 'TolX', 1e-10);
    par = fminsearch(f, par0, opt);
    [par, ~, exitflag] = fminsearch(f, par, opt);   % restart to confirm convergence
end
if exitflag <= 0 || f(par) > f(par0)
    warning('MSOC:clmm', 'Ordinal mixed model did not converge cleanly (exitflag %d).', exitflag);
    if f(par) > f(par0), par = par0; end
end
if useRE, par(end) = clampLogSigma(par(end)); end
H = numHessian(f, par);
V = pinv(H);
% delta method for thresholds and sigma
J = zeros(K-1, numel(par));
J(:, 1) = 1;
for k = 2:K-1
    J(k:end, k) = exp(par(k));
end
fit.theta   = cumsum([par(1); exp(par(2:K-1))]);
fit.beta    = par(K:K-1+p);
fit.seTheta = sqrt(max(diag(J * V * J'), 0));
fit.seBeta  = sqrt(max(diag(V(K:K-1+p, K:K-1+p)), 0));
if useRE
    fit.sigma   = exp(par(end));
    fit.seSigma = fit.sigma * sqrt(max(V(end, end), 0));
else
    fit.sigma = 0;  fit.seSigma = NaN;
end
fit.logLik = -f(par);
fit.useRE  = useRE;  fit.z = z;  fit.w = w;
end

function nll = clmmNLL(par, X, y, g, K, z, w, useRE)
p = size(X, 2);
theta = cumsum([par(1); exp(par(2:K-1))]);
beta  = par(K:K-1+p);
eta   = X * beta;
if useRE
    u    = sqrt(2) * exp(clampLogSigma(par(end))) * z(:)';   % 1 x Q
    logw = log(w(:)') - 0.5 * log(pi);
else
    u = 0;  logw = 0;
end
thU = [theta; Inf];  thL = [-Inf; theta];
PU = 1 ./ (1 + exp(-(thU(y) - eta - u)));          % n x Q
PL = 1 ./ (1 + exp(-(thL(y) - eta - u)));
lp = log(max(PU - PL, 1e-300));
ll = 0;
for i = 1:max(g)
    s  = sum(lp(g == i, :), 1) + logw;
    mx = max(s);
    ll = ll + mx + log(sum(exp(s - mx)));
end
nll = -ll;
if ~isfinite(nll), nll = 1e10; end      % keeps the optimizer away from invalid regions
end

function ls = clampLogSigma(ls)
% Random-intercept SD restricted to [exp(-12), exp(3)] = [6e-6, 20] on the
% logit scale, which prevents divergence in folds with little athlete variance.
ls = min(max(ls, -12), 3);
end

function P = clmmPredict(fit, X)
% Marginal category probabilities for new athletes (integrated over u).
K = numel(fit.theta) + 1;
eta = X * fit.beta;
th  = [-Inf; fit.theta; Inf];
if fit.useRE
    u  = sqrt(2) * fit.sigma * fit.z(:)';
    wq = fit.w(:) / sqrt(pi);
else
    u = 0;  wq = 1;
end
P = zeros(size(X, 1), K);
for k = 1:K
    Pk = 1 ./ (1 + exp(-(th(k+1) - eta - u))) - 1 ./ (1 + exp(-(th(k) - eta - u)));
    P(:, k) = Pk * wq;
end
end

function [x, w] = ghNodes(n)
% Gauss-Hermite nodes and weights for weight function exp(-x^2) (Golub-Welsch).
b = sqrt((1:n-1) / 2);
[V, Dg] = eig(diag(b, 1) + diag(b, -1));
[x, order] = sort(diag(Dg));
w = sqrt(pi) * V(1, order)'.^2;
end

function H = numHessian(f, x)
% Central-difference Hessian.
n = numel(x);  H = zeros(n);
h = 1e-4 * max(1, abs(x));
f0 = f(x);
for i = 1:n
    ei = zeros(n, 1);  ei(i) = h(i);
    H(i, i) = (f(x + ei) - 2*f0 + f(x - ei)) / h(i)^2;
    for j = i+1:n
        ej = zeros(n, 1);  ej(j) = h(j);
        H(i, j) = (f(x+ei+ej) - f(x+ei-ej) - f(x-ei+ej) + f(x-ei-ej)) / (4*h(i)*h(j));
        H(j, i) = H(i, j);
    end
end
end
