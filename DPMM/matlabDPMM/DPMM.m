% Gibbs ampling for DPMM based on Polya Urn representation
% Using a N(0,1) base distribution with strength parameter 1
% and N(mu, eye(D)) mixture components with unknown mu.


% Load data
X = load('../../data_files/toyclusters/toyclusters.dat');
[NN DD] = size(X);

% Normalize and whiten data:
[XX, m, invMat, whMat] = whiten(X);

% Initialise each data point is it's own cluster:

ZZ = (1:NN)';
MU = XX;
KK = NN;

% Set parameters of model:
alpha = 0.1;
numSweeps = 200;

for l = 1:numSweeps
    % permute data:
    sel = randperm(NN);
    for jj = 1:NN
        ii = sel(jj);
        
        % uninitialize current datum
        if sum(ZZ == ZZ(ii)) == 1
            ZZ(ii) = 0;
            CC = ZZ;
            u = unique(ZZ);
            for kk = 0:size(u,1) - 1;
                ZZ(CC == u(kk+1)) = kk;
            end
            KK = KK - 1;
        end    
        
        % draw new mean according to DP posterior
        x = XX(ii,:)';
        q = (4*pi)^(-DD/2)*exp(-0.25 * x'*x);
        draw = rand();
        if draw < alpha/(alpha+NN - 1)
            KK = KK+1;
            MU(ii,:) = randn(DD,1)'*sqrt(1/2) + x'/2;
            ZZ(ii) = KK;
        else
            while(true)
        
                mu_old_index = ZZ(randi(NN));
                if mu_old_index ~= ii & mu_old_index ~= 0
                    break;
                end
            end
            mu_old = MU(mu_old_index,:);
            MU(ii,:) = mu_old;
            ZZ(ii) = mu_old_index;
        end
        
    end
end
        % update 
        
        
        