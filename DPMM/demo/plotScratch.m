X = load('demo.dat');
figure(3);
for k=1:10
    subplot(2,5,k);
    scatter(X(1:k*1000,1),X(1:k*1000,2),'b.');
    axis([-10 10 -10 10]);
    axis off;
end
