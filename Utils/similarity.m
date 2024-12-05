function [S] = similarity(X, type)
% function [S] = similarity(X, type)
%
% Approximate an inner product matrix for incomplete samples with NaN values.
%
% @param  X        Data matrix of size d*n, each column is a sample
% @param  type     Default "true" (calculate the true similarity matrix)
%
% @return S        Inner product matrix of size n*n
%
% <Reference>
% Fangchen Yu, Yicheng Zeng, Jianfeng Mao, and Wenye Li. "Online estimation 
% of similarity matrices with incomplete data." Uncertainty in Artificial 
% Intelligence. PMLR, 2023.

if (nargin < 2)
    type = 'true';
end

switch type
    case 'true'
        % calculate the true inner product matrix
        S = X' * X;
    case 'miss'
        % a fast method to estimate the inner product matrix for incomplete data
        % which has the same results with the type of 'pairwise'
        [d, n] = size(X);
        Idx = isnan(X);
        Xzero = X; Xzero(Idx) = 0;
        XX = Xzero' * Xzero;
        Idx_sum = double((~Idx))' * double((~Idx));
        S = XX ./ Idx_sum * d;
        S(isnan(S)) = 0;
    case 'pairwise'
        % estimate the pairwise inner product for incomplete samples
        % which has the same results with the type of 'miss'
        [d, n] = size(X);
        O = ~isnan(X);
        S = zeros(n);
        for i = 1 : n
            for j = i : n
                k = O(:,i) & O(:,j);
                S(i,j) = X(k,i)'*X(k,j) * d/sum(k);
            end
        end
        S = S + S' - diag(diag(S));
        S(isnan(S)) = 0;
end

end