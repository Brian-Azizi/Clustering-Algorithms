function W = testiwrnd(S,df)

D = size(S,1);
A = chol(S)';

X = zeros(D);
for i=1:D
    a = 0.5*(df - i + 1);
    X(i,i) = sqrt(gamrnd(a,2));
end

for i = 1:D
    for j = i+1:D
        X(i,j) = randn();
    end
end

  
W = A*pinv(X'*X)*A';
