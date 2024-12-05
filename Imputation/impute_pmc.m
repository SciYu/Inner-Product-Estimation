function [imputedX] = impute_pmc(Xmiss, method)
% function [imputedX] = impute_pmc(Xmiss, method)
%
% Impute a data matrix. Each column is a sample. The complete matrix is obtained
% by the Polynomial Matrix Completion (PMC) method (see reference).
%
% @param Xmiss     Incomplete data matrix, each column is a sample
% @param method    Default 'poly' (Polynomial kernel function)
% 
% @return imputedX Imputed matrix with all data samples
%
% <Reference>
% Jicong Fan, Yuqian Zhang, and Madeleine Udell. Polynomial matrix completion 
% for missing data imputation and transductive learning. In Proceedings of 
% the AAAI Conference on Artificial Intelligence, 2020.

if (nargin < 2)
    method = 'poly';
end
low = min(Xmiss(:)); high = max(Xmiss(:));

M = double(~isnan(Xmiss));
X = Xmiss;
X(isnan(X)) = 0;

s = 50; % lower bound of rank

if strcmp(method, 'poly')
    ker.type = 'poly'; ker.par = [1 2];
elseif strcmp(method, 'rbf')
    ker.type = 'rbf'; ker.par = []; ker.c = 3;
end
imputedX = PMC_S(X, M, 0.5, s, ker, 500);

imputedX(imputedX<low) = low;
imputedX(imputedX>high) = high;

end