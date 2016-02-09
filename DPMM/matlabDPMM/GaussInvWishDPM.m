% Dirichlet Process Mixture Model with Gaussian likelihood, unknown
% isotropic variance and unknown mean with a gauss-gamma prior

% Generate Artificial Data
generate_N = 20;
generate_MU = [repmat([1,1],5,1);repmat([1,-1],5,1);...
        repmat([-1,-1],5,1);repmat([-1,1],5,1)];
generate_SIGMA = 0.01*eye(2);
X = mvnrnd(generate_MU, generate_SIGMA);
C_true = [ones(5,1);ones(5,1)*2;ones(5,1)*3;ones(5,1)*4];


% Use toyclusters
%X = load('../../data_files/toyclusters/toyclusters.dat');
%[X, m, invMat, whMat] = whiten(XX);

% Fisher Iris Data
%load fisheriris
%X = meas;

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
    %mu(k,:) = mean(X(clusters == k,:),1);
    mu(k,:) = 0;
end
mu(K+1:end,:) = nan;

sigma = zeros(D, D, N);  % contains covariance matrices
for k = 1:K
    %sigma(:,:,k) = cov(X(clusters == k, :)) + 0.001*eye(D);
    sigma(:,:,k) = eye(D);
end
sigma(:,:,K+1:end) = nan;

% Initialize hyperparameters
alpha = 1;              % DP Concentration parameter
    % prior: NIW(mu,sigma|m_0,k_0,S_0,v_0) 
    % = N(mu|m_0,sigma/k_0)*IW(sigma|S_0,v_0)
S_0 = cov(X) * (N-1)/N;         % S_xbar
v_0 = D + 2;
m_0 = mean(X)';
k_0 = 0.01;

% Initialize sampling parameters
NUM_SWEEPS = 250;
SAVE_CHAIN = false;
if SAVE_CHAIN
    BURN_IN = 50;
    chain_K = zeros(1, NUM_SWEEPS - BURN_IN);
    chain_c = zeros(N, NUM_SWEEPS - BURN_IN);
    chain_cn = zeros(N, NUM_SWEEPS - BURN_IN);
    chain_mu = zeros(N, D, NUM_SWEEPS - BURN_IN);
    chain_sigma = zeros(D, D, N, NUM_SWEEPS - BURN_IN);
end

% Start chain
fprintf('Starting algorithm with K = %d\n', K);

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
            sigma(:,:,c) = sigma(:,:,K);
            clusters(clusters == K) = c;
            mu(K,:) = nan;
            sigma(:,:,K) = nan;
            clust_sizes(K) = nan;
            K = K - 1;
        end
        
        % find categorical distribution for c_i
            % p(existing clusters)
        p = zeros(1, K+1);
        x_ = X(i,:)';
        for k = 1:K
            p(k) = clust_sizes(k)/(N-1+alpha) ...
                    * mvnpdf(x_, mu(k,:)', sigma(:,:,k));
        end
            % p(new cluster): find partition function     
        dummy_mu = zeros(D,1);
        dummy_sigma = eye(D);
        [~, logprior] = NIWpdf(dummy_mu,dummy_sigma,m_0,k_0,S_0,v_0);
        logLklihd = log(mvnpdf(x_, dummy_mu, dummy_sigma));
            % posterior hyperparameters:
        k_1 = k_0 + 1;
        m_1 = (k_0*m_0 + x_)/k_1;
        v_1 = v_0 + 1;
        S_1 = S_0 + x_*x_' + k_0*(m_0*m_0') - k_1*(m_1*m_1');
        [~, logpstr] = NIWpdf(dummy_mu,dummy_sigma,m_1,k_1,S_1,v_1);
            % partition = prior*lklihd/pstr
        logPartition = logprior + logLklihd - logpstr;
        partition = exp(logPartition);
        
        p(K+1) = alpha/(N-1+alpha) * partition;
        p = p/sum(p);
            
        % Sample c_i
        [~, c] = max(mnrnd(1,p));
        clusters(i) = c;
     
        
        % if new cluster created: sample parameters
        if c == K+1
            clust_sizes(K+1) = 1;
            [mu_, sigma_] = NIWrnd(m_1,k_1,S_1,v_1);
            mu(K+1,:) = mu_;
            sigma(:,:,K+1) = sigma_;
            K = K + 1;
        else
            clust_sizes(c) = clust_sizes(c) + 1;
        end
    end
   
    % sample new cluster parameters from posterior
    for k = 1:K
        Xk = X(clusters == k, :);
        Nk = clust_sizes(k);
        if Nk ~= size(Xk,1)
            disp('Error');
        end
        
        % posterior hyperparameters
        sum_k = sum(Xk,1)';
        cov_k = zeros(D);
        for l = 1:Nk
            cov_k = cov_k + Xk(l,:)'*Xk(l,:);
        end
        k_Nk = k_0 + Nk;
        m_Nk = (k_0*m_0 + sum_k) / k_Nk;
        v_Nk = v_0 + Nk;
        S_Nk = S_0 + cov_k + k_0*(m_0*m_0') - k_Nk*(m_Nk*m_Nk');
        
        % sample
        [mu_, sigma_] = NIWrnd(m_Nk,k_Nk,S_Nk,v_Nk);
        mu(k,:) = mu_;
        sigma(:,:,k) = sigma_;
    end
    
    fprintf('Iteration %d / %d done. K = %d\n', sweep, NUM_SWEEPS,K);
    % save the chains
    if SAVE_CHAIN
        if sweep > BURN_IN
            chain_K(sweep - BURN_IN) = K;
            chain_c(:,sweep - BURN_IN) = clusters;
            chain_cn(:, sweep - BURN_IN) = clust_sizes;
            chain_mu(:,:, sweep - BURN_IN) = mu;
            chain_sigma(:,:,:, sweep - BURN_IN) = sigma;
        end
    end
end

fprintf('cluster sizes:\t');
for k = 1:K
    fprintf('%d\t',clust_sizes(k));
end
fprintf('\n');

% FOR 2D DATA ONLY:
if D == 2
    figure(2)
    %subplot(1,2,1);
    %scatter(X(:,1),X(:,2));
    %gscatter(X(:,1),X(:,2),C_true)
    %title('true clusters');
    %subplot(1,2,2)
    gscatter(X(:,1),X(:,2),clusters);
    hold on
    for k = 1:K
        plot(mu(k,1),mu(k,2),'k^')
    end
    title('Gaussian-Inverse-Wishart Dirichlet Process Mixture')
end
% FOR 3D DATA ONLY:
if D == 3
    figure(2)
    cmap = colormap('jet');
    ncolours = size(cmap,1);
    col_step = ceil(ncolours/K);
    colours = 'ymcrgbk';
    for k = 1:K
        data = X(clusters==k,:);
        plot3(data(:,1),data(:,2),data(:,3),...
            'Color', cmap((k-1)*col_step+1,:),'Marker','.',...
                'LineStyle','none','MarkerSize',15);
        hold on;
    end
    grid;
end
    
        
        
        