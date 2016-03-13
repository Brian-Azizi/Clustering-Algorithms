clear,clc;



%kk = [1 3 6 0];
%aa = [1 3 2 4];
%for qq = 1:4
%k = kk(qq);
k=[];

Xfile = strcat('demo',num2str(k),'.dat');
MUfile = strcat('dpmMU',num2str(k),'.out');
Sfile = strcat('dpmSIGMA',num2str(k),'.out');
Gfile = strcat('dpmIDX',num2str(k),'.out');

X = load(Xfile);
MU = load(MUfile);
SIGMA = load(Sfile);
G = load(Gfile);
u = unique(G);
K = size(u,1);

figure(2);

%subplot(2,2,aa(qq));
%pic = gscatter(X(:,1), X(:,2), G);
pic = scatter(X(:,1),X(:,2),'b.');
axis([-10 10 -10 10]);
axis off;
legend off;
hold on


for j=1:K
    k = u(j);
    if sum(G == k) < 6
        continue;
    end
    %# indices of points in this group
    idx = ( G == k );

    %# substract mean
    Mu = mean( X(idx,:) );
    X0 = bsxfun(@minus, X(idx,:), Mu);

    %# eigen decomposition [sorted by eigen values]
    [V D] = eig( X0'*X0 ./ (sum(idx)-1) );     %#' cov(X0)
    [D order] = sort(diag(D), 'descend');
    D = diag(D);
    V = V(:, order);

    t = linspace(0,2*pi,100);
    e = [cos(t) ; sin(t)];        %# unit circle
    VV = V*sqrt(D);               %# scale eigenvectors
    e = bsxfun(@plus, VV*e, Mu'); %#' project circle back to orig space
    
    color = 'k';
    %color = pic(j).Color;
    %# plot cov and major/minor axes
    plot(e(1,:), e(2,:), 'Color',color,'LineWidth',1.5);
    %#quiver(Mu(1),Mu(2), VV(1,1),VV(2,1), 'Color','k')
    %#quiver(Mu(1),Mu(2), VV(1,2),VV(2,2), 'Color','k')
end

%end
%saveas(gcf,'test.png');