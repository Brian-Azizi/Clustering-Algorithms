kk = [1 3 6 0];
lb = ['(a)','(b)','(c)','(d)'];
figure(2);

for j = 1:4
    k = kk(j);
    filename = strcat('K',num2str(k),'.out');

    kh = load(filename);
    
    subplot(2,2,j);
    
    histogram(kh);
    axis([0,11,0,1000]);
    %xlabel(lb([3*j-2,3*j-1,3*j]));
    
end


saveas(gcf,'kHist.png');
