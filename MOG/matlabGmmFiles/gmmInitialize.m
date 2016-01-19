function [MU, SIGMA, PI] = gmmInitialize(X, K)
% Initialize the parameters of the Gaussian Mixture Model to small random
% values
% Input:
% X = Input data
% K = number of base distributions (clusters)
% Output:
% MU = K x D matrix containing the class means in its rows
% SIGMA = (K*D) x D matrix, containing the K class covariance matrices
% stacked vertically (each covariance matrix is small and positive
% semi-definite
% PI = K x 1 matrix containing the mixing coefficients of each base
% distribution (sum(PI) = 1)

[N, D] = size(X);

sel = randperm(N);

MU = X(sel(1:K),:);

SIGMA = zeros(D*K,D);
for i = 1:K
    sigma = rand(D);
    SIGMA(D*(i-1) + 1 : D*i, :) = sigma'*sigma;
end

PI = ones(K,1) / K;
end
