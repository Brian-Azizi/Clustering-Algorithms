C = load('centroids.out');
K = size(C,1);
figure(2);
h = displayData(C);

% hold on;
% for k = 1:64
%     a = reshape(C(k,:),20,20);
%     subplot(8,8,k)
%     imagesc(a);
% end
