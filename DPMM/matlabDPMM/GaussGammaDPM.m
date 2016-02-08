% Dirichlet Process Mixture Model with Gaussian likelihood, unknown
% isotropic variance and unknown mean with a gauss-gamma prior

% Generate Artificial Data
generate_N = 40;
generate_MU = [repmat([1,1],10,1);repmat([1,-1],10,1);...
        repmat([-1,-1],10,1);repmat([-1,1],10,1)];
generate_SIGMA = 0.01*eye(2);
X = mvnrnd(generate_MU, generate_SIGMA);
C_true = [ones(10,1);ones(10,1)*2;ones(10,1)*3;ones(10,1)*4];

% Use toyclusters
%XX = load('../../data_files/toyclusters/toyclusters.dat');
%[X, m, invMat, whMat] = whiten(XX);

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

mu = zeros(N, D);       % contains mean
for k = 1:K
    mu(k,:) = mean(X(clusters == k,:),1);
end
mu(K+1:end,:) = nan;

sigma_2 = zeros(N, 1);  % contains variance
for k = 1:K
    sigma_2(k) = 0.01*trace(cov(X(clusters == k, :))) / D;
end
sigma_2(K+1:end) = nan;

% Initialize hyperparameters
alpha = 1;              % DP Concentration parameter
% prior: NiG(mu,sigma_2|m_0,V_0^-1,a_0,b_0) 
% = N(mu|m_0,V_0*sigma_2)*IG(sigma_2|a_0,b_0)
m_0 = zeros(D, 1);
V_0 = eye(D);
a_0 = 0.9;
b_0 = 5;
% Likelihood: N(x|mu,sig_2*S)
S = eye(D);

% Cache hyperparameters
a_1 = a_0 + D/2;
inv_V_0 = pinv(V_0);
inv_S = pinv(S);
inv_V_1 = inv_V_0 + inv_S;
V_1 = pinv(inv_V_1);
V1_invV0_m0 = V_1*inv_V_0*m_0;  % needed for m_1
V1_invS = V_1*inv_S;            % needed for m_1
m0_invV0_m0 = m_0'*inv_V_0*m_0; % needed for b_1

% Initialize sampling parameters
NUM_SWEEPS = 100;
BURN_IN= 50;
chain_K = zeros(1, NUM_SWEEPS - BURN_IN);
chain_c = zeros(N, NUM_SWEEPS - BURN_IN);
chain_cn = zeros(N, NUM_SWEEPS - BURN_IN);
chain_mu = zeros(N, D, NUM_SWEEPS - BURN_IN);
chain_sigma_2 = zeros(N, NUM_SWEEPS - BURN_IN);

% Start chain
for sweep = 1:NUM_SWEEPS
% "E-Step": for i=1:N in random order
    sel = randperm(N);
    for j = 1:N
        i = sel(j);
    % Remove c_i
        c = clusters(i);             % current cluster
        clust_sizes(c) = clust_sizes(c) - 1;
        if clust_sizes(c) == 0       % remove empty clusters
            clust_sizes(c) = clust_sizes(K);  % move entries for K onto position c
            mu(c,:) = mu(K,:);
            sigma_2(c) = sigma_2(K);
            clusters(clusters == K) = c;
            mu(K,:) = nan;
            sigma_2(K) = nan;
            clust_sizes(K) = nan;
            K = K - 1;
        end
        
        % Sample c_i from categorical
        p = zeros(1, K+1);
        x_ = X(i,:)';               % current data point
        for k = 1:K
            mu_ = mu(k,:)';
            sig_2_ = sigma_2(k);
            Sig_ = sig_2_*S;
            p(k) = clust_sizes(k)/(N-1+alpha) * mvnpdf(x_, mu_, Sig_);
        end
        
        % Find the partition function
        mu_dummy = zeros(D,1);
        sig_2_dummy = 1;
        lklihood = mvnpdf(x_, mu_dummy, sig_2_dummy*S);
        prior = NIGpdf(mu_dummy,sig_2_dummy,m_0,inv_V_0,a_0,b_0);
        m_1 = V1_invV0_m0 + V1_invS*x_;
        b_1 = b_0 + 0.5*x_'*inv_S*x_ + 0.5*m0_invV0_m0 ...
                - 0.5*m_1'*inv_V_1*m_1;
        posterior = NIGpdf(mu_dummy,sig_2_dummy,m_1,inv_V_1,a_1,b_1);
        partition = lklihood * prior / posterior;
        
        p(K+1) = alpha/(N-1+alpha) * partition;
        p = p / sum(p);
        
        [~, c] = max(mnrnd(1,p));   % c is the new cluster identity
        clusters(i) = c;
        
        % sample mean if new cluster
        if c == K + 1
            clust_sizes(K+1) = 1;
            %[mu_,sig_] = NIGrnd(m_1,V_1,a_1,b_1);
            s = gamrnd(a_1,1/b_1);
            sig2_ = 1/s;
            mu_ = mvnrnd(m_1, sig2_*V_1);
            mu(K+1,:) = mu_;
            sigma_2(K+1) = sig2_;
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
        a_Nk = a_0 + Nk*D/2;
        inv_V_Nk = inv_V_0 + Nk*inv_S;
        V_Nk = pinv(inv_V_Nk);
        m_Nk = V_Nk*(inv_V_0*m_0 + Nk*inv_S*mean(Xk,1)');
        b_Nk = b_0 + 0.5*trace(X*inv_S*X') + 0.5*m_0'*inv_V_0*m_0 ...
               - 0.5*m_Nk'*inv_V_Nk*m_Nk; 
        
        % sample from posterior
        s = gamrnd(a_Nk,1/b_Nk);
        sig2_ = 1/s;
        mu_ = mvnrnd(m_Nk, sig2_*V_Nk);
        mu(k,:) = mu_;
        sigma_2(k) = sig2_;
    end 
    fprintf('Iteration %d done: K = %d\n', sweep, K);
    
    % keep values for the chain
    if sweep > BURN_IN
        chain_K(sweep - BURN_IN) = K;
        chain_c(:,sweep - BURN_IN) = clusters;
        chain_cn(:, sweep - BURN_IN) = clust_sizes;
        chain_mu(:,:, sweep - BURN_IN) = mu;
        chain_sigma_2(:, sweep - BURN_IN) = sigma_2;
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

