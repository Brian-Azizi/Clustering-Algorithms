function Gamma = EstepGMM(X, K, MU, SIGMA, PI)
    [N, D] = size(X);
    Gamma = zeros(N, D);
    for i = 1:N
       for k = 1:K
        r = PI(k);
        mu = MU(k,:)';
        sigma = SIGMA(D*(k-1)+1:D*k, :);
        Gamma(i,k) = r*(2*pi)^(-0.5*D)*det(sigma)^(-0.5)*exp(...
            -0.5*(X(i,:)'-mu)' * pinv(sigma) * (X(i,:)' - mu));
       end
       Gamma(i, :) = Gamma(i,:) / sum(Gamma(i,:));
    end
end