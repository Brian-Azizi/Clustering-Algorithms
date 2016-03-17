function centroids = kMeansMstep(X, idx, K)
[N D] = size(X);

centroids = zeros(K, D);

for k = 1:K
    centroids(k, :) = mean(X(idx == k, :));
end


end

