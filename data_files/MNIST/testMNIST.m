% X = loadMNISTImages('train-images.idx3-ubyte')';
% y = loadMNISTLabels('train-labels.idx1-ubyte');

figure(2)
for i=1:100
    a = reshape(saveX(350+i,:),28,28);
    subplot(10,10,i);
    imagesc(a);
    axis off;
end
    