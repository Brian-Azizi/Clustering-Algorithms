% Gibbs ampling for DPMM based on Polya Urn representation
% Using a N(0,1) base distribution with strength parameter 1
% and N(mu, eye(D)) mixture components with unknown mu.


% Load data
X = load('../../data_files/toyclusters/toyclusters.dat');
[NN DD] = size(X);

% Normalize and whiten data:
[XX, m, invMat, whMat] = whiten(X);

% Initialise each data point is it's own cluster:
B = (1:NN)';
PHI = XX;
KK = NN;
a = zeros(1,KK+1);

alpha = 0.1;
numSweeps = 2000;

for ll = 1:numSweeps
    sel = randperm(NN);
    for jj = 1:NN
        ii = sel(jj);
        x = XX(ii,:);
        
        % uninitialize current datum
        if sum(B == B(ii)) == 1
            rm_idx = B(ii);
            PHI = [PHI([1:rm_idx-1, rm_idx+1:end],:);-9999,-9999];
            B(ii) = 0;
            CC = B;
            u = unique(B);
            for kk = 0:size(u,1) - 1;
                B(CC == u(kk+1)) = kk;
            end
            B(ii) = -1;
            KK = KK - 1;
        end
        
        % compute a_i_0 and a_i,k
        a(1) = alpha/NN * (4*pi)^(-DD/2) * exp(-1/4 * x*x');
        for kk = 1:KK
            a(kk+1) = 1/NN * (2*pi)^(-DD/2) *...
                     exp(-0.5*(x-PHI(kk,:))*(x-PHI(kk,:))');
        end
        a(KK+2:end) = 0;
        a = a/sum(a);
        
        % sample B_i
        [~, B(ii)] = max(mnrnd(1,a));
        B(ii) = B(ii) - 1;
        % add cluster if appropriate
        if B(ii) == 0
            B(ii) = KK+1;
            KK = KK+1;
        end
        
        % sample PHI from N(x_i_k/(1+n_k),I/(1+n_k)) = N(Xk,Sx)
        for kk = 1:KK
            Nk = sum(B == kk);
            Xk = sum(XX(B == kk,:))/(Nk+1);
            z = randn(1,DD);
            PHI(kk,:) = z / sqrt(Nk+1) + Xk;
        end 
    end
end
        