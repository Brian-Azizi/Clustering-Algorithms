function [centroids, idx] = runkMeans(X, initial_centroids, max_iters)

% Initialize values
[N D] = size(X);
K = size(initial_centroids, 1);
centroids = initial_centroids;
idx = zeros(N, 1);

% Run K-Means
for i=1:max_iters
    
    % Output progress
    fprintf('K-Means iteration %d/%d...\n', i, max_iters);
    
    % E-Step
    idx = kMeansEstep(X, centroids);
    
    % M-Step
    centroids = kMeansMstep(X, idx, K);
end


end

