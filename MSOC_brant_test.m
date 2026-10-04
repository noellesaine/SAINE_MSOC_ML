%% MSOC_brant_test.m  (2026-10-03)
% Brant test of the proportional-odds assumption of the ordinal logistic
% regression reported in Supplemental Table S7 (Section 9a of
% MSOC_ML_Pipeline.m v2.3): PLI category on z-scored sRPE, %maxHR and TD,
% full analytic sample (372 sessions, 10 athletes).
%
% Why this test: the likelihood-ratio test against the non-parallel-slopes
% cumulative-logit model (v2.3, LR = 8.36) is not valid for these data,
% because that model implies negative category probabilities for 17 sessions.
% The Brant test (Brant 1990, Biometrics 46:1171-8) never fits that model.
% It fits a separate binary logistic regression at each cut point
% (Y > Low and Y > Medium) and tests, with a Wald statistic, whether the
% slopes are equal across the two fits. The covariance between the two sets
% of estimates is the one derived by Brant:
%   Cov(b_j, b_l) = (X'W_jj X)^-1 (X'W_jl X) (X'W_ll X)^-1,
%   W_jl = diag(pi_l - pi_j .* pi_l) for l >= j, pi_j = fitted P(Y > j).
% Output: omnibus test (df = 3) and one test per predictor (df = 1), each
% with the asymptotic chi-square p-value and a parametric-bootstrap p-value
% (2,000 data sets simulated from the fitted proportional-odds model). The
% bootstrap is included because, with coefficients as large as those here,
% the Wald statistic tends to be conservative at n = 372 (checked by
% simulation before release).
%
% Data loading, exclusions and cut points are copied from Sections 1-3 of
% MSOC_ML_Pipeline.m v2.3; the script stops if the sample or the class counts
% differ from those reported (372 sessions; 149/147/76), and prints the
% ordinal-model deviance, which should be 167.297 as in the v2.3 log.
% Requirements: MATLAB with the Statistics and Machine Learning Toolbox.
% Runtime: under one minute. Writes MSOC_brant_log.txt and
% Table_S12b_brant_test.csv in the current folder.

clear; clc;
cfg.dataFile   = 'MSOC Data.xlsx';
cfg.mlSheet    = 'ML Sheet (2)';
cfg.seed       = 20260925;
cfg.minSessionsPerAthlete = 10;
cfg.excludeZeroPLI = true;
cfg.nBoot      = 2000;
K = 3;
featLabel = {'sRPE','%maxHR','TD'};

logFile = fullfile(pwd, 'MSOC_brant_log.txt');
if exist(logFile, 'file'), delete(logFile); end
diary(logFile);
fprintf('MSOC_brant_test.m (2026-10-03), MATLAB %s, %s\n', version, char(datetime('now')));

%% 1  IMPORT (as in MSOC_ML_Pipeline.m Section 1)
mlVars = {'Prim_V_Sec','Ath_ID','Prac_o_Gam','Return','sRPE','Well1_Fat', ...
          'Well2_Stress','Well3_Sore','P_HR_avg_%','P_Tot_Dis','P_Training_load_score'};
R = readSheet(cfg.dataFile, cfg.mlSheet, mlVars);
for v = 1:numel(mlVars), R.(mlVars{v}) = toDouble(R.(mlVars{v})); end
ids = R.Ath_ID;

%% 2  ANALYTIC SAMPLE (as in Section 2)
complete = all(~isnan(R{:, mlVars}), 2);
keep = complete & R.Prim_V_Sec == 1;
[g, gid] = findgroups(ids(keep));
nPerAth  = splitapply(@numel, ids(keep), g);
lowIDs   = gid(nPerAth < cfg.minSessionsPerAthlete);
keep     = keep & ~ismember(ids, lowIDs);
if cfg.excludeZeroPLI
    keep = keep & ~(R.P_Training_load_score == 0);
end
D   = R(keep, :);
n   = height(D);
pli = D.P_Training_load_score;
Xcand = [D.sRPE, D.Well1_Fat, D.Well2_Stress, D.Well3_Sore, D.('P_HR_avg_%'), D.P_Tot_Dis];

%% 3  PLI CATEGORIES (as in Section 3; cut points mean +/- 0.5 SD, analytic sample)
cutPoints = mean(pli) + [-0.5 0.5] * std(pli);
ycat = pliCategory(pli, cutPoints);
catN = accumarray(ycat, 1, [K 1]);
fprintf('Analytic sample: %d sessions, %d athletes; cut points %.2f and %.2f AU; classes %d/%d/%d\n', ...
    n, numel(unique(D.Ath_ID)), cutPoints, catN);
assert(n == 372 && isequal(catN, [149; 147; 76]), 'MSOC:sample', ...
    'Sample or class counts differ from v2.3 (372 sessions; 149/147/76). Check the data file.');

%% 4  PREDICTORS (as in Section 9) AND PROPORTIONAL-ODDS FIT
infSel = [1 5 6];                                  % sRPE, %maxHR, TD
Zall = (Xcand(:, infSel) - mean(Xcand(:, infSel))) ./ std(Xcand(:, infSel));
p = numel(infSel);
[B0, dev0] = mnrfit(Zall, ycat, 'model', 'ordinal');
fprintf('Ordinal logistic regression: deviance %.3f (v2.3: 167.297); beta = %s (v2.3: 0.810, 4.850, 4.556)\n', ...
    dev0, mat2str(-B0(K:end)', 4));

%% 5  BRANT TEST
[Wobs, dObs, VdObs, Bbin, SEbin, warnObs] = brantWald(Zall, ycat, K);
if ~isempty(warnObs)
    fprintf('Note: glmfit warning in the observed-data fits: %s\n', warnObs);
end
fprintf('\nBinary logistic regressions at each cut point (z-scored predictors):\n');
binTbl = table(featLabel', -B0(K:end), Bbin(2:end, 1), SEbin(2:end, 1), Bbin(2:end, 2), SEbin(2:end, 2), ...
    'VariableNames', {'Predictor','Ordinal_beta','Beta_Y_gt_Low','SE_Y_gt_Low', ...
                      'Beta_Y_gt_Medium','SE_Y_gt_Medium'});
disp(binTbl);

% Parametric bootstrap under the fitted proportional-odds model
rng(cfg.seed, 'twister');
Pfit = mnrval(B0, Zall, 'model', 'ordinal');       % n x K category probabilities
cumP = cumsum(Pfit, 2);
Wb = nan(cfg.nBoot, 1 + p);
nWarn = 0;  nSkip = 0;
for b = 1:cfg.nBoot
    ys = 1 + sum(rand(n, 1) > cumP(:, 1:K-1), 2);
    if numel(unique(ys)) < K, nSkip = nSkip + 1; continue; end
    [Ws, ds, Vds, ~, ~, wmsg] = brantWald(Zall, ys, K);
    nWarn = nWarn + ~isempty(wmsg);
    Wb(b, :) = [Ws, (ds.^2 ./ diag(Vds))'];
end
Wb = Wb(~any(isnan(Wb), 2), :);
nB = size(Wb, 1);

Wper  = (dObs.^2 ./ diag(VdObs))';                 % per-predictor Wald, df = 1
stat  = [Wobs, Wper];
df    = [(K-2)*p, ones(1, p)];
pAsym = 1 - chi2cdf(stat, df);
pBoot = (1 + sum(Wb >= stat, 1)) / (nB + 1);
brantTbl = table([{'Omnibus (all three predictors)'}; featLabel'], stat', df', pAsym', pBoot', ...
    'VariableNames', {'Test','Wald_chi2','df','p_asymptotic','p_bootstrap'});
fprintf('\nBrant test of proportional odds (H0: equal slopes at both cut points):\n');
disp(brantTbl);
fprintf('Bootstrap: %d valid data sets (%d skipped: a category empty; %d with a glmfit warning), seed %d\n', ...
    nB, nSkip, nWarn, cfg.seed);
writetable(brantTbl, fullfile(pwd, 'Table_S12b_brant_test.csv'));
diary off;

%% ========================================================================
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

function T = readSheet(file, sheet, vars)
opts = detectImportOptions(file, 'Sheet', sheet, 'VariableNamingRule', 'preserve');
missingVars = setdiff(vars, opts.VariableNames);
if ~isempty(missingVars)
    error('MSOC:columns', 'Sheet "%s" is missing column(s): %s', sheet, strjoin(missingVars, ', '));
end
opts.SelectedVariableNames = vars;
T = readtable(file, opts);
end

function x = toDouble(x)
if iscell(x) || isstring(x)
    x = str2double(string(x));
else
    x = double(x);
end
x = x(:);
end

function c = pliCategory(x, cut)
c = 2 * ones(size(x));
c(x <= cut(1)) = 1;
c(x >= cut(2)) = 3;
end
