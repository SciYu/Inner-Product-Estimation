function [S_new] = correct_ec_iid(S_miss, X_miss, r)
% function [S_new] = correct_ec_iid(S_miss, X_miss, r)
%
% Correct the inner product matrix from incomplete i.i.d. data with MCAR
% by the Eigenvalue Correction (EC) method
%
% @param  S_miss    Estimated inner product matrix on incomplete data
% @param  X_miss    Incomplete data with MCAR of size d*n
% @param  r         Missing rate of MCAR
%
% @return S_new     Corrected inner product matrix for S_miss
%
% <Reference>
% See Algorithm 1 in Section 3.3 for more details.

[d, n] = size(X_miss);

% Eigen-decomposition of S_miss
[U, Sigma_miss] = eig(S_miss);
sigma_miss = diag(Sigma_miss);

% Correct eigenvalues by linear transformation
if d < n
    sigma_ec = (1-r) * sigma_miss + r;
    % if d < n, set the smallest (n-d) eigenvalues to zero
    sigma_ec(1:n-d) = 0;
else
    sigma_ec = (1-r) * sigma_miss + r;
end

% Re-construct the inner product matrix
S_new = U * diag(sigma_ec) * U';

end