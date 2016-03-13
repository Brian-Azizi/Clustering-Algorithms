clear,clc;

input = 'monarch';
output = strcat(input,'_reconstructed.png');
input = strcat(input,'.png');

MU = load('dpmMU.out');
IDX = load('dpmIDX.out');
pic = double(imread(input))/255;

dms = size(pic);
pic_vec = reshape(pic,dms(1)*dms(2),3);

u = unique(IDX);

for k = 1:length(u)
    sel = find(IDX == k-1);
    for j = 1:length(sel)
        pic_vec(sel(j),:) = MU(k,:);
    end
end
pic_redo = reshape(pic_vec,dms);

figure(2);
subplot(1,2,1);
imagesc(pic);
axis off
subplot(1,2,2);
imagesc(pic_redo);
axis off;
saveas(gcf,'output.png');
figure(3);
imagesc(pic_redo);
axis off;
saveas(gcf,output);