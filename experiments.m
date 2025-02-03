clear all; warning off; clc; 
addpath(genpath(pwd));

%% Settings
dataset = 'CIFAR10';  % dataset
mechanism = 'MCAR';   % missing mechanism
n = 1000;             % number of incomplete samples 
r = 0.8;              % missing ratio
top_eig = 5;          % hyper-parameter: top-k eigenvalues
seed = 2024;          % random seed

fprintf('\nWWW-2025 paper "A Theory-Driven Approach to Inner Product Matrix Estimation for Incomplete Data: An Eigenvalue Perspective"');
fprintf('\nDemo: Evaluation on Estimation Error and Similarity Search in Sections 6.2 and 6.3');
fprintf('\nNote: Due to limited size of supplementary materials, only one iteration of data can be provided\n');

%% Load Incomplete Dataset
load([dataset,'_n',num2str(n),'_',num2str(r*100),'miss.mat']);
d = size(Xtrue_list{1}, 1);
niter = length(Xmiss_list);
fprintf(['\n',dataset,'-',mechanism,': n=',num2str(n),', r=',num2str(r)]);

time = zeros(niter, 10);
for i = 1:niter
    rng(seed + i);
    clear X_train X_true X_miss S_train S_true S_miss D_miss S_list D_list X_impute
    fprintf(['\n  iter = ',num2str(i),': ']);
    
    X_miss = Xmiss_list{i}; % incomplete data subset
    X_ref = Xref_list{i};   % complete data subset
    X_true = Xtrue_list{i}; % ground-truth of incomplete data

    % calculate the normalized inner product matrix
    S_ref = 1/d * similarity(X_ref);
    S_true = 1/d * similarity(X_true);
    
    % calculate the squared Euclidean distance matrix
    J = ones(n, n);
    D_true = diag(diag(d*S_true))*J + J*diag(diag(d*S_true)) - 2*d*S_true;

    %% Data Imputation
    fprintf('Mean, '); tic; X_impute{1} = impute_mean([X_ref, X_miss]); time(i,1) = toc;
    fprintf('kNN, ');  tic; X_impute{2} = impute_knn([X_ref, X_miss]);  time(i,2) = toc;
    fprintf('SVT, ');  tic; X_impute{3} = impute_svt([X_ref, X_miss]);  time(i,3) = toc;
    fprintf('KFMC, '); tic; X_impute{4} = impute_kfmc([X_ref, X_miss]); time(i,4) = toc;
    fprintf('PMC, ');  tic; X_impute{5} = impute_pmc([X_ref, X_miss]);  time(i,5) = toc;
    
    %% Similarity Calibration
    fprintf('S0, ');  tic; S_miss = 1/d * similarity(X_miss, 'miss'); time(i,9) = toc;
    fprintf('DMC, '); tic; S_dmc = calibrate_dmc(S_miss); time(i,6) = toc;
    fprintf('SMC, '); tic; S_smc = calibrate_smc(S_miss); time(i,7) = toc;
    fprintf('SVC, '); tic; S_svc = calibrate_svc(S_miss); time(i,8) = toc;
    
    %% Our Method
    fprintf('Ours. '); tic; S_ec = correct_ec_noniid(S_miss, S_ref, top_eig); time(i,10) = toc;
    
    %% Calculation
    fprintf('Calculating, ');
    num_impute = length(X_impute);
    % calculate the normalized inner product matrix
    for j = 1 : num_impute
        S_list{j} = 1/d * similarity(X_impute{j}(:,n+1:2*n));
    end
   
    S_list{num_impute+1} = S_dmc;
    S_list{num_impute+2} = S_smc;
    S_list{num_impute+3} = S_svc;
    S_list{num_impute+4} = S_miss; 
    S_list{num_impute+5} = S_ec;
    
    % calculate the squared Euclidean distance matrix
    for j = 1 : length(S_list)
        D_list{j} = diag(diag(d*S_list{j}))*J + J*diag(diag(d*S_list{j})) - 2*d*S_list{j};
    end
    
    %% Evaluation
    fprintf('Evaluating, ');
    for j = 1 : length(S_list)
        % Relative Error of Inner Product Estimation
        S_RE(i,j) = norm(S_list{j}-S_true,'fro') / norm(S_true,'fro');
        
        % Recall@10 of Maximum Inner Product Search
        S_Recall(i,j) = eval_recall(S_list{j}, S_true, 10, 10, 'MIPS');

        % Relative Error of Euclidean Distance Estimation
        D_RE(i,j) = norm(D_list{j}-D_true,'fro') / norm(D_true,'fro');
        
        % Recall@10 of Nearest Neighbor Search
        D_Recall(i,j) = eval_recall(D_list{j}, D_true, 10, 10, 'NNS');
    end

    %% Statistics
    fprintf('Statistics, ');
    Stat_all = [mean(S_RE,1)', mean(S_Recall,1)', mean(D_RE,1)', mean(D_Recall,1)', mean(time,1)'];
    
    %%
    fprintf('Finished.\n\n');
    Stat_table = array2table(Stat_all, 'VariableNames', {'RE(S)','Recall(S)','RE(D)','Recall(D)','Time (sec)'},...
               'RowNames', {'Mean','kNN','SVT','KFMC','PMC','DMC','SMC','SVC','S0 (D0)','Ours'});
    disp(Stat_table);
end
