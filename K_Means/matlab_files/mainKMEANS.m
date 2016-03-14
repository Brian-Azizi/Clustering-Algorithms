clear, clc, clf;

X = load('demo.dat');
[N, D] = size(X);

maxK = 15;
J = zeros(1,maxK);

for K = 1:maxK
maxIter = 200;

initialCentroids = kMeansInitCentroids(X,K);
%initialCentroids = [-1 6; 1 6; 8 0];
[centroids, idx]  = runkMeans(X,initialCentroids,maxIter,false);

%figure(2);
%subplot(2,2,1);
%plotDataPoints(X,idx,K);
%hold on;
%plot(centroids(:,1), centroids(:,2), 'x', ...
%     'MarkerEdgeColor','k', ...
%     'MarkerSize', 10, 'LineWidth', 3);
%axis off;
dd = distortion(X,centroids,idx);
%saveas(gcf,'local.png');
J(K) = dd;
end

figure(2);
subplot(1,2,1);
scatter(X(:,1),X(:,2),'.')
%plotDataPoints(X,ones(N,1),1);
axis off;
subplot(1,2,2);
plot(J)
grid on;