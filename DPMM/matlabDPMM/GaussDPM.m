% Dirichlet Process Mixture Model with Gaussian likelihood, fixed variance
% and unknown mean with a gaussian prior

% Generate Artificial Data
%N = 40;
%MU = [repmat([1,1],10,1);repmat([1,-1],10,1);...
%        repmat([-1,-1],10,1);repmat([-1,1],10,1)];
%SIGMA = 0.01*eye(2);
%X = mvnrnd(MU, SIGMA);
%C_true = [ones(10,1);ones(10,1)*2;ones(10,1)*3;ones(10,1)*4];

% Use toyclusters
XX = load('../../data_files/toyclusters/toyclusters.dat');
[X, m, invMat, whMat] = whiten(XX);

% Initialize everything
[N, D] = size(X);
K = N;

clusters = zeros(N, 1);        % Contains cluster assignments
for i = 1:N
    clusters(i) = mod(i-1,K) + 1;  % initialize as [1,2,...,K,1,2,...K,1,...]
end

clust_sizes = zeros(N, 1);       % contains cluster sizes
for k = 1:K
    clust_sizes(k) = sum(clusters == k);
end
clust_sizes(K+1:end) = nan;

mu = zeros(N, D);       % contains cluster parameters
for k = 1:K
    mu(k,:) = mean(X(clusters == k,:),1);
end
mu(K+1:end,:) = nan;

% Initialize hyperparameters
alpha = 1;              % DP Concentration parameter
m_0 = zeros(D, 1);
V_0 = 0.01*eye(D);
SIGMA = 0.15*eye(D);

% pre-compute posterior hyperparameters
inv_V_0 = pinv(V_0);
inv_SIGMA = pinv(SIGMA);
V_1 = pinv(inv_V_0 + inv_SIGMA);

% Initialize sampling parameters
NUM_SWEEPS = 100;
BURN_IN= 50;
chain_K = zeros(1, NUM_SWEEPS - BURN_IN);
chain_c = zeros(N, NUM_SWEEPS - BURN_IN);
chain_cn = zeros(N, NUM_SWEEPS - BURN_IN);
chain_mu = zeros(N, D, NUM_SWEEPS - BURN_IN);

% Start chain
for sweep = 1:NUM_SWEEPS
% "E-Step": for i=1:N in random order
    sel = randperm(N);
    for j = 1:N
        i = sel(j);
    % Remove c_i
        c = clusters(i);       % current cluster
        clust_sizes(c) = clust_sizes(c) - 1;
        if clust_sizes(c) == 0       % remove empty clusters
            clust_sizes(c) = clust_sizes(K);  % move entries for K onto position c
            mu(c,:) = mu(K,:);
            clusters(clusters == K) = c;
            mu(K,:) = nan;
            clust_sizes(K) = nan;
            K = K - 1;
        end
    
        % Sample c_i from categorical
        p = zeros(1, K+1);
        x_ = X(i,:)';               % current data point
        for k = 1:K
            mu_k = mu(k,:)';
            p(k) = clust_sizes(k)/(N-1+alpha) * mvnpdf(x_, mu_k, SIGMA);
        end
            % find the partition function
        mu_dummy = zeros(D,1);
        lklihood = mvnpdf(x_, mu_dummy, SIGMA);
        prior = mvnpdf(mu_dummy, m_0, V_0);
        m_1 = V_1 * (inv_SIGMA*x_ + inv_V_0*m_0);
        posterior = mvnpdf(mu_dummy, m_1, V_1);
        partition = lklihood * prior / posterior;

        p(K+1) = alpha/(N-1+alpha) * partition;
        p = p/sum(p);

        [~, c] = max(mnrnd(1,p));       % c is now the new cluster identity of i
        clusters(i) = c;

        % sample mean if new cluster
        if c == K + 1
            clust_sizes(K+1) = 1;
            mu_ = mvnrnd(m_1, V_1);     % Sample from posterior
            mu(K+1,:) = mu_;
            K = K + 1;
        else
            clust_sizes(c) = clust_sizes(c) + 1;
        end
    end
    
% "M-Step": for c = 1:K
    % sample mean from posterior
    for k = 1:K
        Xk = X(clusters == k,:);        % cluster data
        Nk = size(Xk,1);                % cluster size 
        
        % posterior hyper parameters        
        V_k = pinv(inv_V_0 + Nk * inv_SIGMA);
        m_k = V_k * (Nk * inv_SIGMA * mean(Xk,1)' + inv_V_0 * m_0);
        
        % sample mean
        mu(k,:) = mvnrnd(m_k, V_k);
    end
    fprintf('Iteration %d done: K = %d\n', sweep, K);
    
% keep values for the chain
    if sweep > BURN_IN
        chain_K(sweep - BURN_IN) = K;
        chain_c(:,sweep - BURN_IN) = clusters;
        chain_cn(:, sweep - BURN_IN) = clust_sizes;
        chain_mu(:,:, sweep - BURN_IN) = mu;
    end
end

fprintf('cluster sizes:\t');
for k = 1:K
    fprintf('%d\t',clust_sizes(k));
end
fprintf('\n');

% FOR 2D DATA ONLY:
figure(2)
subplot(1,2,1);
scatter(X(:,1),X(:,2));
%gscatter(X(:,1),X(:,2),C_true)
%title('true clusters');
subplot(1,2,2)
gscatter(X(:,1),X(:,2),clusters);
hold on
for k = 1:K
    plot(mu(k,1),mu(k,2),'k^')
end
title('DPMM output')