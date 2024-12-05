function [S_new] = correct_ec_scale(S_miss, S_ref, partition, top_eig, top_sng, parallel)
% function [S_new] = correct_ec_scale(S_miss, S_ref, partition, top_eig, top_sng, parallel)
%
% Correct the inner product matrix by the scalable Eigenvalue Correction method
%
% @param  S_miss     Estimated inner product matrix on incomplete data
% @param  S_ref      Reference inner product matrix on complete data
% @param  partition  Partition size m in Algorithm 3
% @param  top_eig    Number of top-k eigenvalues retained in S_miss
% @param  top_sng    Number of top-k singular values retained in S_miss
% @param  parallel   'true' or 'false' for parallel processing
%
% @return S_new      Corrected inner product matrix for S_miss
%
% <Reference>
% See Algorithm 3 in Section 5.1 and Appendix B.1 for more details.

if (nargin < 6)
    parallel = 'false';
end

n_miss = size(S_miss,1);
n_ref = size(S_ref,1);

if mod(n_miss, partition) ~= 0 || mod(n_ref, partition) ~= 0
    disp('Choose a suitable partition size.');
    return;
end

% Step 1. Partition
S_miss_sub = partitionMatrix(S_miss, partition);
S_ref_sub = partitionMatrix(S_ref, partition);

% Step 2. Calculate eigenvalues for all diagonal blocks
[eigValue_Smiss, eigVector_Smiss] = parallel_evd(S_miss_sub, parallel);
[eigValue_Sref, ~] = parallel_evd(S_ref_sub, parallel);

% Step 3. Calculate singular values for all off-diagonal blocks
[sngValue_Smiss, sngVectorLeft_Smiss, sngVectorRight_Smiss] = parallel_svd(S_miss_sub, parallel);
[sngValue_Sref, ~, ~] = parallel_svd(S_ref_sub, parallel);

% Step 4. Correct all eigenvalues and singular values
eigValue_Snew = eigCorrection(eigValue_Smiss, eigValue_Sref, top_eig, 'eig');
sngValue_Snew = eigCorrection(sngValue_Smiss, sngValue_Sref, top_sng, 'sng');

% Step 5. Reconstruct each sub-matrix
S_new_diagonal = reconstruct_diagonal(n_miss, eigValue_Snew, eigVector_Smiss, parallel);
S_new_offdiagonal = reconstruct_offdiagonal(n_miss, sngValue_Snew, sngVectorLeft_Smiss, sngVectorRight_Smiss, parallel);

% Step 6. Reconstruct the whole matrix
S_new = S_new_diagonal + S_new_offdiagonal;
S_new(1:n_miss+1:n_miss^2) = diag(S_miss);
end

%%
function S_offdiagonal = reconstruct_offdiagonal(n_miss, sngValues, sngVectorsLeft, sngVectorsRight, parallel)

% Reconstruct off-diagonal blocks

partition = length(sngValues{1});
n_submatrix = n_miss / partition;

S_offdiagonal = zeros(n_miss, n_miss);

idx = find(triu(ones(n_submatrix), 1)); % Indices of upper off-diagonal elements
if strcmp(parallel, 'true')
    tempBlocks = cell(1, length(idx));
    parfor k = 1:length(idx)
        U = sngVectorsLeft{k};
        sigma = sngValues{k};
        V = sngVectorsRight{k};
        
        tempBlocks{k} = U * diag(sigma) * V';
    end
    for k = 1:length(idx)
        [i, j] = ind2sub([n_submatrix, n_submatrix], idx(k));
        rowIndices = (i-1)*partition + (1:partition);
        colIndices = (j-1)*partition + (1:partition);
        S_offdiagonal(rowIndices, colIndices) = tempBlocks{k};
    end
else
    for k = 1:length(idx)
        [i, j] = ind2sub([n_submatrix, n_submatrix], idx(k)); % Convert linear index to subscript
        
        U = sngVectorsLeft{k};
        sigma = sngValues{k};
        V = sngVectorsRight{k};
        
        rowIndices = (i-1)*partition + (1:partition);
        colIndices = (j-1)*partition + (1:partition);
        S_offdiagonal(rowIndices, colIndices) = U * diag(sigma) * V';
    end
end

S_offdiagonal = S_offdiagonal + S_offdiagonal';
    
end

%%
function S_diagonal = reconstruct_diagonal(n_miss, eigValues, eigVectors, parallel)

% Reconstruct diagonal blocks

n_diagonal = length(eigValues);
partition = n_miss / n_diagonal;

S_diagonal = zeros(n_miss, n_miss);

if strcmp(parallel, 'true')
    tempBlocks = cell(1, n_diagonal);
    parfor i = 1:n_diagonal
        U  = eigVectors{i};
        sigma = eigValues{i};
        tempBlocks{i} = U * diag(sigma) * U';
    end
    for i = 1:n_diagonal
        rowIndices = (i-1)*partition + (1:partition);
        S_diagonal(rowIndices, rowIndices) = tempBlocks{i};
    end
else
    for i = 1:n_diagonal
        rowIndices = (i-1)*partition + (1:partition);
        U  = eigVectors{i};
        sigma = eigValues{i};
        S_diagonal(rowIndices, rowIndices) = U * diag(sigma) * U';
    end
end

end

%%
function eigValue_Snew = eigCorrection(eigValue_Smiss, eigValue_Strain, top_eig, type)

% Correct eigenvalues or singular values for each sub-matrix 

eigAvg = cellAverage(eigValue_Strain);
eigValue_Snew = eigValue_Smiss;
n_eig = length(eigValue_Smiss{1});

if top_eig == 0
   for i = 1:length(eigValue_Smiss)
       eigValue_Snew{i} = eigAvg;
   end
else
    idx = n_eig - top_eig;
    for i = 1:length(eigValue_Smiss)
        eigvalue_miss = eigValue_Smiss{i};
        if strcmp(type, 'eig')
            eigvalue_new = [eigAvg(1:idx); eigvalue_miss(idx+1:end)];
        elseif strcmp(type, 'sng')
            eigvalue_new = [eigvalue_miss(1:top_eig); eigAvg(top_eig+1:end)];
        end
        eigValue_Snew{i} = eigvalue_new;
    end
end
end

%%
function avg = cellAverage(value)

% Calculate average eigenvalues or singular values

avg = value{1};
for i = 2:length(value)
    avg = avg + value{i};
end
avg = avg / length(value);

end

%%
function [S_submatrix] = partitionMatrix(S, partition)

% Partition the whole matrix into sub-matrices of the partition size

n = size(S, 1); 
n_submatrix = n / partition; 
S_submatrix = cell(n_submatrix, n_submatrix);

for i = 1:n_submatrix
    for j = i:n_submatrix
        rowIndices = (i-1)*partition + (1:partition);
        colIndices = (j-1)*partition + (1:partition);
        S_submatrix{i, j} = S(rowIndices, colIndices);
    end
end
end

%%
function [eigValues, eigVectors] = parallel_evd(S_submatrix, parallel)

% Perform eigen-decompositions for all diagonal blocks in parallel

n_submatrix = size(S_submatrix, 1);
eigValues = cell(1, n_submatrix);
eigVectors = cell(1, n_submatrix);

if strcmp(parallel, 'true')
    parfor i = 1:n_submatrix % Parallel loop for principal minors
        [U, Sigma] = eig(S_submatrix{i, i});
        eigValues{i} = diag(Sigma);
        eigVectors{i} = U;
    end
else
    for i = 1:n_submatrix
        [U, Sigma] = eig(S_submatrix{i, i});
        eigValues{i} = diag(Sigma);
        eigVectors{i} = U;
    end
end
end

%%
function [sngValues, sngVectorsLeft, sngVectorsRight] = parallel_svd(S_submatrix, parallel)

% Perform singular value decomposition for all off-diagonal blocks in parallel

n_submatrix = size(S_submatrix, 1);
idx = find(triu(ones(n_submatrix), 1)); % Indices of upper off-diagonal elements
sngValues = cell(1, length(idx));
sngVectorsLeft = cell(1, length(idx));
sngVectorsRight = cell(1, length(idx));

if strcmp(parallel, 'true')
    parfor k = 1:length(idx) % Parallel loop for off-diagonal matrices
        [i, j] = ind2sub([n_submatrix, n_submatrix], idx(k)); % Convert linear index to subscript
        [U, S, V] = svd(S_submatrix{i, j});
        sngValues{k} = diag(S);
        sngVectorsLeft{k} = U;
        sngVectorsRight{k} = V;
    end
else
    for k = 1:length(idx)
        [i, j] = ind2sub([n_submatrix, n_submatrix], idx(k)); % Convert linear index to subscript
        [U, S, V] = svd(S_submatrix{i, j});
        sngValues{k} = diag(S);
        sngVectorsLeft{k} = U;
        sngVectorsRight{k} = V;
    end
end
end
