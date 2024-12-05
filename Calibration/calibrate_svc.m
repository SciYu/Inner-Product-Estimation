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

% use the DMC algorithm to correct Soff and Son
Soff_0 = S0(1:n_off, 1:n_off);
Soff = calibrate_dmc(Soff_0);

Son_0 = S0(n_off+1:n_off+n_on, n_off+1:n_off+n_on);
Son = calibrate_dmc(Son_0);

% to correct Spar_0 by parallel correction
Spar_0 = S0(1:n_off, n_off+1:n_off+n_on);
Spar = calibrate_vectors(Soff, Spar_0);

S_new = [Soff, Spar; Spar', Son];

end

%%
function [Spar] = calibrate_vectors(S0, Spar_0, parallel)
% function [Spar] = calibrate_vectors(S0, Spar_0, parallel)
%
% Calibrate multiple similarity vectors in parallel
%
% @param  S0         Pairwise similarity matrix
% @param  Spar_0     Intial similarity vectors
% @param  parallel   Default "for" ("parfor" for parallel processing)
%
% @return Spar       Calibrated similarity vectors
%
% <Reference>
% Changyi Ma, Runsheng Yu, and Youzhi Zhang. "A fast similarity matrix 
% calibration method with incomplete query." WWW, 2024.

if (nargin < 3)
    parallel = 'for';
end

n = size(Spar_0, 2);
tol = 1e-4;

if isempty(Spar_0)
    Spar = Spar_0;
else
    % Standard Singular Value Decomposition
    [V, S, ~] = svd(S0);

    s = diag(S);
    C = V * diag(sqrt(s));
    Cinv = diag(1./s) * C';
    U0 = Cinv * Spar_0;
    U = U0;
    c = S0(end, end);

    % We can use "parfor" to execute parallel correction 
    if strcmp(parallel, 'parfor')
        parfor i = 1 : n 
            u0 = U0(:, i);
            U(:, i) = calibrate_single_vector(s, u0, c, tol);
        end
    elseif strcmp(parallel, 'for')
        for i = 1 : n 
            u0 = U0(:, i);
            U(:, i) = calibrate_single_vector(s, u0, c, tol);
        end
    end
    Spar = C * U;
end

end

%%
function [u] = calibrate_single_vector(s, u0, c, tol)
% function [u] = calibrate_single_vector(s, u0, c, tol)
% 
% Calibrate one similarity vector.
%
% <Reference>
% Changyi Ma, Runsheng Yu, and Youzhi Zhang. "A fast similarity matrix 
% calibration method with incomplete query." WWW, 2024.

if (nargin < 4)
    tol = 1e-4;
end
if (nargin < 3)
    c = 1;
end

if norm(u0)^2 <= c
    u = u0;
else
    lambda_min = 0;
    lambda_max = max(s) * norm(u0) / (2 * c^0.5);
    lambda = lambda_min;
    u = (s ./ (s+lambda)) .* u0;
    u_len = norm(u)^2;
    iter = 0;
    while (u_len > c) || (u_len < c-tol)
        iter = iter + 1;
        if iter > 50
            break
        end
        lambda = 0.5*(lambda_min + lambda_max);
        u = (s ./ (s+lambda)) .* u0;
        u_len = norm(u)^2;
        % use bisection search to find optimal lambda
        if u_len > c
            lambda_min = lambda;
        elseif u_len < c-tol
            lambda_max = lambda;
        end
    end
end
end