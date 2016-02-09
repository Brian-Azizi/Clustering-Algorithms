% produces a random sample from the normal-inverse-Wishart distribution

function [mu, sigma] = NIWrnd(m, k, S, v)
sigma = iwishrnd(S,v);      % matlab uses same parametrization
mu = mvnrnd(m, sigma/k);
end