X = load('polar.dat');
[N D] = size(X);

G = load('../idx.out');
M = load('../centroids.out');

orig = double(imread('polar.jpg'))/255;
[H W C] = size(orig);

u = unique(G);
K = size(u,1);

% Reconstruct
for k=1:K
    for i = 1:N
        if G(i) == u(k)
            X(i,:) = M(k,:);
        end
    end
end

reco = reshape(X,H,W,C);
figure(2);
subplot(1,2,1);
imagesc(orig);
axis off;
subplot(1,2,2);
imagesc(reco);
axis off;