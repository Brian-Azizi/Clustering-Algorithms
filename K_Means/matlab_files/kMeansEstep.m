function idx = kMeansEstep(X, centroids)

K = size(centroids, 1);

idx = zeros(size(X,1), 1);
D = zeros(size(X, 1), K);
for k = 1:K
    D(:, k) = sum((X - ones(size(X, 1), 1) * centroids(k, :)).^2, 2);
end
[~, idx] = min(D, [], 2);

end

