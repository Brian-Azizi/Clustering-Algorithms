
% load Data
X = load('demo.dat');
[N, D] = size(X);


% Set range of K for elbow curve
K = 10;
maxK = K;
minK = K;

J = zeros(1,maxK-minK+1);

for K = minK:maxK
    maxIter = 200;

    initialCentroids = kMeansInitialize(X,K);

    [centroids, idx]  = runkMeans(X,initialCentroids,maxIter);

    % Plot output for 2d data
    if D == 2
        figure(2);
        gscatter(X(:,1),X(:,2),idx);
        hold on;
        plot(centroids(:,1), centroids(:,2), 'x', ...
            'MarkerEdgeColor','k', ...
            'MarkerSize', 10, 'LineWidth', 3);
        %axis off;
        dd = distortion(X,centroids,idx);
        J(K) = dd;
        hold off
    end
    pause;
end

% plot Elbow curve and data for Figure 3
% figure(3);
% subplot(1,2,1);
% scatter(X(:,1),X(:,2),'.')
% axis off;
% subplot(1,2,2);
% plot(J)
% grid on;