function [mu, sig2] = NIGrnd(m, V, a, b)
s = gamrnd(a,1/b);
sig2 = 1/s;
mu = mvnrnd(m, sig2*pinv(V));
end