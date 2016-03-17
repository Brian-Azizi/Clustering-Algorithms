function centroids = kMeansInitialize(X, K)

randix = randperm(size(X, 1));
centroids = X(randix(1:K), :);

end

