% Mixture of Gaussians

% load the data:
dataFile = 'toyclusters.dat';
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
    disp(n);
     Gamma = EstepGMM(X, K, MU, SIGMA, PI);
     [MU, SIGMA, PI] = MstepGMM(X, K, Gamma);
     
     % check convergence
     J(n) = gmmLogLikelihood(X,K,MU,SIGMA,PI);
     if abs(J(n) - J_old) < 0.00001*(1+abs(J_old))
         J(n+1:end) = [];
         fprintf('Converged after %d iterations.\n',n);
         fprintf('log Likelihood = %f\n', J(n));
         break;
     end
     
     J_old = J(n);
end

% plot log likelihood
plot(J,'LineWidth',2.5);

% plot 2d data and visualize distribution
if D == 2
    figure(2);
    subplot(1,2,1);
    scatter(X(:,1),X(:,2),'bo','filled');
    hold on;
    %set(gca, 'color', [0 0 0]);

    for k=1:K
    plot(MU(k,1), MU(k,2),'or','MarkerSize',15,'MarkerFaceColor','r');
    %plot(MU(2,1), MU(2,2),'g*', 'MarkerSize', 15);
    %plot(MU(3,1), MU(3,2),'y*', 'MarkerSize', 15);
    end

        
    % Anomaly detection
    x_test = [2; 3];
    p_test = 0;
    for k = 1:K
        r = PI(k);
        mu = MU(k,:)';
        sigma = SIGMA(D*(k-1)+1:D*k, :);
        p_test = p_test + r*(2*pi)^(-0.5*D)*det(sigma)^(-0.5)*exp(...
                -0.5*(x_test - mu)' * pinv(sigma) * (x_test - mu));
    end
    plot(x_test(1),x_test(2),'mo','MarkerFaceColor','m','MarkerSize',10);
    fprintf('Test point: (%f,%f)\n',x_test(1),x_test(2));
    fprintf('Density at test point = %f\n',p_test);

    % surface plot
    amin = floor(min(X(:,1))-1);
    amax = ceil(max(X(:,1))+1);
    bmin = floor(min(X(:,2))-1);
    bmax = ceil(max(X(:,2))+1);
    a = linspace(amin,amax,100);
    b = linspace(bmin,bmax,100);
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
    subplot(1,2,2);
    surf(A,B,Z);
    

end








