function recall = eval_recall(S, Strue, top_k, search_N, task)
% function recall = eval_recall(S, Strue, top_k, search_N, task)
%
% Evaluate the similarity search performance by Recall k@N, which calculates 
% the average proportion of true top-k results within the top-N retrieved 
% candidates across all queries.
%
% @param  S         Estimated inner product matrix on incomplete data
% @param  S_true    True inner product matrix on incomplete data
% @param  top_k     Number of top-k results from S_true
% @param  search_N  Number of top-N retrieved candidates from S
% @param  task      Similarity search tasks (MIPS or NNS)
%
% @return recall    Average Recall@10 (k=N=10)
%
% <Reference>
% See Appendix C.3 for more details.

if strcmp(task, 'MIPS')
    % MIPS: Maximum Inner Product Search
    order = 'descend';
    n = size(Strue, 1);
    Strue(1:n+1:n^2) = 0;
    S(1:n+1:n^2) = 0;
elseif strcmp(task, 'NNS')
    % NNS: Nearest Neighbor Search
    order = 'ascend';
    n = size(Strue, 1);
    Strue(1:n+1:n^2) = nan;
    S(1:n+1:n^2) = nan;
end

num_query = size(S, 2);

[~, Idx_true] = sort(Strue, order);
[~, Idx_ini] = sort(S, order);
Idx_true_k = Idx_true(1:top_k, :);

num_N = length(search_N);
recall_one = zeros(num_N, num_query);

for i = 1:num_N
    top_N = search_N(i);
    Idx_ini_N = Idx_ini(1:top_N, :);
    
    for j = 1:num_query
        recall_one(i,j) = length(intersect(Idx_true_k(:,j), Idx_ini_N(:,j))) / top_k;
    end
end
recall = mean(recall_one, 2);

end