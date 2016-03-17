% save image files as ascii
A = double(imread('polar.jpg'))/255;
[H W C] = size(A);

fprintf('Image size: %dx%d\nChannel: %d\n',H,W,C);

% reshape
B = reshape(A,H*W,C);

save -ascii 'polar.dat' B;