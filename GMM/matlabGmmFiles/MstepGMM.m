function [MU, SIGMA, PI] = MstepGMM(X, K, Gamma)
    [N, D] = size(X);
    MU = zeros(K, D);
    SIGMA = zeros(D*K, D);
    PI = zeros(K,1);
    
    Nk = sum(Gamma);
    
    for k = 1:K
        for i = 1:N
            MU(k,:) = MU(k,:) + Gamma(i,k)*X(i,:);
        end
        MU(k,:) = MU(k,:) / Nk(k);
        
        sigma = zeros(D,D);
        for i = 1:N
            sigma = sigma + Gamma(i,k) * (X(i,:)-MU(k,:))'*(X(i,:)-MU(k,:));
        end
        SIGMA((k-1)*D+1:k*  D, :) = sigma / Nk(k);
        PI(k) = Nk(k)/N;
    end
               
end