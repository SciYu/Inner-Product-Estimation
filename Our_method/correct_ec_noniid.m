function [S_new] = correct_ec_noniid(S_miss, S_ref, top_eig)
% function [S_new] = correct_ec_noniid(S_miss, S_ref, top_eig)
%
% Correct the inner product matrix from incomplete non-i.i.d. data
% by the Eigenvalue Correction (EC) method
%
% @param  S_miss    Estimated inner product matrix on incomplete data
% @param  S_ref     Reference inner product matrix on complete data
% @param  top_eig   Number of top-k eigenvalues retained in S_miss
%
% @return S_new     Corrected inner product matrix for S_miss
%
% <Reference>
% See Algorithm 2 in Section 4.3 for more details.

n = size(S_miss, 1);

% Eigen-decomposition of S_miss
[U, Sigma_miss] = eig(S_miss);
sigma_miss = diag(Sigma_miss);

% Eigen-decomposition of S_ref
sigma_ref = eig(S_ref);

% Correct eigenvalues: keep top-k eigenvalues and replace others
if top_eig == 0
    sigma_ec = sigma_ref;
else
    idx = n - top_eig;
    sigma_ec = [sigma_ref(1:idx); sigma_miss(idx+1:end)];
end

% Re-construct the inner product matrix
S_new = U * diag(sigma_ec) * U';
S_new(1:n+1:n^2) = diag(S_miss);

end