% Mixture of Gaussians

% load the data:
dataFile = '../../data_files/toyclusters/toyclusters.dat';
X = load(dataFile);
[N, D] = size(X);

% Enter K:
K = 3;

% Initialize the parameter:
[MU, SIGMA, PI] = gmmInitialize(X,K);

% Evaluate log likelihood:
J_old = gmmLogLikelihood(X,K,MU,SIGMA,PI);
J_initial = J_old;
% Start EM algorithm

maxIter = 100;
J = zeros(maxIter,1);

for n = 1:maxIter
     Gamma = EstepGMM(X, K, MU, SIGMA, PI);
     [MU, SIGMA, PI] = MstepGMM(X, K, Gamma);
     J(n) = gmmLogLikelihood(X,K,MU,SIGMA,PI);
     if abs(J(n) - J_old) < 0.00001*(1+abs(J_old))
         J(n+1:end) = [];
         break;
     end
     J_old = J(n);
end

% plot log likelihood
plot(J);

% plot data and visualize distribution
figure;
scatter(X(:,1),X(:,2),'r.');
hold on;
set(gca, 'color', [0 0 0]);
plot(MU(1,1), MU(1,2),'c*', 'MarkerSize', 15);
plot(MU(2,1), MU(2,2),'g*', 'MarkerSize', 15);
plot(MU(3,1), MU(3,2),'y*', 'MarkerSize', 15);

a = -2:0.1:10;
b = 0:0.1:6;
[A B] = meshgrid(a,b);
Z = zeros(size(A(:)));
for k = 1:K
    r = PI(k);
    mu = MU(k,:)';
    sigma = SIGMA(D*(k-1)+1:D*k, :);
    Z = Z + r*(2*pi)^(-0.5*D)*det(sigma)^(-0.5)*exp(...
            -0.5*sum(([A(:),B(:)]-ones(size(A(:)))*mu') *...
            pinv(sigma) .* ([A(:),B(:)]-ones(size(A(:)))*mu'),2));
end
Z = reshape(Z, size(A));
contour(A,B,Z);
figure
surf(A,B,Z);






