function [S_new] = calibrate_svc(S0)
% function [S_new] = calibrate_svc(S0)
%
% Calibrate the similarity matrix by the SVC method (see reference)
%
% @param  S0      Initial similarity matrix
%
% @return S_new   Calibrated similarity matrix
%
% <Reference>
% Changyi Ma, Runsheng Yu, and Youzhi Zhang. "A fast similarity matrix 
% calibration method with incomplete query." WWW, 2024.

n = size(S0, 1);
n_off = n/2;
n_on = n - n_off;

S_off = S0(1:n_off, 1:n_off);
Simp = calibrate_dmc(S_off);

for i = 1 : n_on
    c = S0(n_off+i, n_off+i);            % default c=1
    v0 = S0(1:n_off+i-1, n_off+i);       % similarity vector
    vonl = calibrate_qn(Simp, v0, c);    % calibrated similarity vector
    Simp = [Simp, vonl; vonl', c];       % calibrated similarity matrix
end
S_new = Simp; % calibrated similarity matrix
end

%%
function [v] = calibrate_qn(S0, v0, c)
% function [v] = calibrate_qn(S0, v0, c)
%
% Correct one similarity vector using the BFGS-Quasi-Newton (QN) optimization 
% to find the optimal lambda. (see Algorithm 2 in the reference)
%
% @param  S0      Initial similarity matrix
% @param  v0      Initial similarity vector
% @param  c       Similarity value of itself (default c=1)
%
% @return v       Calibrated similarity vector
%
% <Reference>
% Changyi Ma, Runsheng Yu, and Youzhi Zhang. "A fast similarity matrix 
% calibration method with incomplete query." WWW, 2024.

if (nargin < 3)
    c = 1;
end

n = size(S0, 1); % Number of samples

% Standard Singular Value Decomposition
[U, S, ~] = svd(S0);

s = diag(S);
C = U * diag(sqrt(s));
Cinv = diag(1 ./ s) * C';

y0 = Cinv * v0;
if norm(y0)^2 <= c + 1e-4
    ycal = y0;
else
    % Define function for optimization
    f_lambda = @(lambda) (norm((s ./ (s + lambda)) .* y0)^2 - c)^2;

    % Apply the Quasi-Newton method to find the optimal lambda
    lambda0 = 0;
    options = optimoptions('fminunc', 'Algorithm', 'quasi-newton', 'Display', 'off');
    lambda_opt = fminunc(f_lambda, lambda0, options);

    % Compute the calibrated vector
    ycal = (s ./ (s + lambda_opt)) .* y0;
end

v = C * ycal; % Corrected similarity vector
end


