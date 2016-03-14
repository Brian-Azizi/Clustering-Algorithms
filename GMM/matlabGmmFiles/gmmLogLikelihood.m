function L = gmmLogLikelihood(X,K,MU,SIGMA,PI)
[N, D] = size(X);

L = 0;

for i = 1:N
    p = 0;
    for k = 1:K
        r = PI(k);
        mu = MU(k,:)';
        sigma = SIGMA(D*(k-1)+1:D*k, :);
        p = p + r*(2*pi)^(-0.5*D)*det(sigma)^(-0.5)*exp(...
            -0.5*(X(i,:)'-mu)' * pinv(sigma) * (X(i,:)' - mu));
    end
    L = L + log(p);
end

        