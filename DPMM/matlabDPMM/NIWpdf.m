function [pdf, logpdf] = NIWpdf(mu, sigma, m, k, S, v)
[~, logIW] = invwishpdf(sigma,S,v);
logN = log(mvnpdf(mu, m, sigma/k));

logpdf = logIW + logN;
pdf = exp(logpdf);
end