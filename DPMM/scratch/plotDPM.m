%DPM

% Clear everything
%     clear; clc; close ALL;

rngsetting = rng;  % save the current rng settings for reproducability
% Generate Gaussian clusters of data

% True number of clusters:
K = 10;
N = 500; 
% Number of clusters in GMM algorithm 

    MU = zeros(K, 2);
    SIGMA = zeros(2, 2, K);
    X = zeros(K * N, 2);
    for k = 1:K;
        MU(k,:) = 1.2*K*(rand(1, 2)-0.5);
        s = (rand(2,2) - 0.5);
        SIGMA(:,:,k) = 0.5*K*(s'*s + 0.1*eye(2));
        %SIGMA(:,:,k) = eye(2);
        X(N*(k-1)+1 : N*k , :) = mvnrnd(MU(k,:), SIGMA(:,:,k), N);
    end
    %sel = randperm(N*K); 
    sel = [1:N*K];
figure('units','normalized','position',[.1 .1 .8 .8]);
hold on;
set(gca, 'color', [1 1 1])
% scatter(X(:,1),X(:,2),10,'r.')
for k=1:K
    scatter(X(sel(N*(k-1)+1:N*k),1),X(sel(N*(k-1)+1:N*k),2),10,'b.'),
    pause;
end
%hold on
