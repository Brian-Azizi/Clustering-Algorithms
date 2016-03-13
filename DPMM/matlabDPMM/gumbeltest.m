y = [-0.2, 0.2, 1, 0.5, -0.1];
q = exp(y)/sum(exp(y));

cq = cumsum(q);
u = rand();
D = zeros(1,5);
G = zeros(1,5);

numtries = 1000000;
for i = 1: numtries
u = rand();
for j = 1:5
    if cq(j) > u
        D(j) = D(j) + 1;
        break;
    end
end

g = -log(-log(rand(1,5)));
[~,j] = max(y+g);
G(j) = G(j) + 1;
end

D = D/numtries;
G = G/numtries;

figure(2)
subplot(1,2,1)
bar(D)
subplot(1,2,2)
bar(G)
disp(q');