function [S_new] = calibrate_smc(S0)
% function [S_new] = calibrate_smc(S0)
%
% Calibrate the similarity matrix by the SMC method (see reference)
%
% @param  S0      Initial similarity matrix
%
% @return S_new   Calibrated similarity matrix
%
% <Reference>
% Fangchen Yu, Yicheng Zeng, Jianfeng Mao, and Wenye Li. "Online estimation 
% of similarity matrices with incomplete data." Uncertainty in Artificial 
% Intelligence. PMLR, 2023.

n = size(S0, 1);
n_off = n/2;
n_on = n - n_off;

S_off = S0(1:n_off, 1:n_off);
Simp = calibrate_dmc(S_off);

for i = 1 : n_on
    c = S0(n_off+i, n_off+i);            % default c=1
    v0 = S0(1:n_off+i-1, n_off+i);       % similarity vector
    vonl = calibrate_step(Simp, v0, c);  % calibrated similarity vector
    Simp = [Simp, vonl; vonl', c];       % calibrated similarity matrix
end
S_new = Simp; % calibrated similarity matrix
end

%%
function [v] = calibrate_step(S0, v0, c)
% function [v] = calibrate_step(S0, v0, c)
%
% Correct one similarity vector by the One-Step Matrix Calibration method
%
% @param  S0      Intial similarity matrix
% @param  v0      Intial similarity vector
% @param  c       Similarity value of itself (default c=1)
%
% @return v       Calibrated similarity vector
%
% <Reference>
% Fangchen Yu, Yicheng Zeng, Jianfeng Mao, and Wenye Li. "Online estimation 
% of similarity matrices with incomplete data." Uncertainty in Artificial 
% Intelligence. PMLR, 2023.

if (nargin < 3)
    c = 1;
end

n = size(S0, 1); % number of samples
tol = 1e-4;      % convergence tolerance

% Standard Singular Value Decomposition
[U, S, ~] = svd(S0);

s = diag(S);
C = U * diag(sqrt(s));
Cinv = diag(1./s) * C';

y0 = Cinv * v0;
if norm(y0)^2 <= c
    % to test if y0 is a feasible solution
    ycal = y0;
else
    % to correct y0
    lambda_min = 0;
    lambda_max = max(s) * norm(y0) / (2 * c^0.5);
    lambda = lambda_min;
    ycal = (s ./ (s+lambda)) .* y0;
    ylen = norm(ycal)^2;
    iter = 0;
    while (ylen > c) || (ylen < c-tol)
        iter = iter + 1;
        if iter > 50
            break
        end
        lambda = 0.5*(lambda_min + lambda_max);
        ycal = (s ./ (s+lambda)) .* y0;
        ylen = norm(ycal)^2;
        % use bisection search to find optimal lambda
        if ylen > c
            lambda_min = lambda;
        elseif ylen < c-tol
            lambda_max = lambda;
        end 
    end
end
v = C * ycal; % corrected similarity vector
end
