function J = distortion(X,centroids,idx)
[N D] = size(X);
[K d] = size(centroids);
assert(d==D);

J = 0;
for k=1:K
    nk = sum(idx==k);
    J = J + sum(sum((X(idx==k,:)-repmat(centroids(k,:),nk,1)).^2));
end