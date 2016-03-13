X = loadMNISTImages('train-images.idx3-ubyte')';
y = loadMNISTLabels('train-labels.idx1-ubyte');

% shuffle X
[N D] = size(X);
shuffleIDX = randperm(N);
X = X(shuffleIDX,:);
y = y(shuffleIDX,:);

num = 10;  % Save num samples of each class
saveX = zeros(10*num,D);
saveY = zeros(10*num,1);

for k=1:10
    sel = y==k-1;
    selX = X(sel,:);
    selY = y(sel,:);
    saveX((k-1)*num+1:k*num,:) = selX(1:num,:);
    saveY((k-1)*num+1:k*num,:) = selY(1:num,:);
end

save -ascii 'MNIST.dat' saveX;
save -ascii 'MNISTlabels.dat' saveY;

disp('Done.');