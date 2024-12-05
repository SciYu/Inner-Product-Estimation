function [S_new] = calibrate_dmc(S0)
% function [S_new] = calibrate_dmc(S0)
%
% Calibrate a similarity matrix by solving min_S ||S-S0||_F^2.
%
% @param  S0        Initial similarity matrix (estimated from incomplete data)
% @return S_new     Calibrated similarity matrix
%
% <Reference>
% Wenye Li. "Estimating jaccard index with missing observations: 
% a matrix calibration approach", NeurIPS, 2015.

% Eigen-decomposition
[U, Sigma] = eig(S0);
sigma = diag(Sigma);

% Set all negative eigenvalues as zero
sigma_new = max(sigma, 0);
S_new = U * diag(sigma_new) * U';
S_new = (S_new + S_new') / 2;

end