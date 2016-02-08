function nig = NIGpdf(mu,sigma_2,m,inv_V,a,b)

D = size(inv_V,1);
nig = sqrt(det(inv_V)) * (2*pi)^(-D/2) * b^(a) / gamma(a) * ...
     sigma_2^(-D/2-a-1) * exp(-1/(2*sigma_2)*(2*b + (mu-m)'*inv_V*(mu-m)));

%N = mvnpdf(mu,m,sigma_2*pinv(inv_V));
%G = gampdf(1/sigma_2,a,1/b);
%nig = N*G;
end