% Compute the inverse Wishart pdf at IW (positive definite DxD matrix)
% with scale parameter S (positive definite DxD matrix) and 
% degrees of freedom v (> D-1)
% Warning: Does NOT check for positive definite-ness of IW and of S
% ...Seems to be working.

function [pdf, logpdf] = invwishpdf(IW, S, v)

D = size(IW);
if D(1) ~= D(2)
    disp('Error: input is not square matrix');
    return;
elseif D ~= size(S)
        disp('Error: Scale Matrix has wrong size');
        return;
end
D = D(1);
if v <= D - 1
    disp('Error: Degrees of freedom must exceed D-1');
    return;
end
logS_term = 0.5*v * log(det(S));
log2_term = 0.5*v*D * log(2);
logGamma_term = 0.25*D*(D-1) * log(pi);
for j = 1:D
    logGamma_term = logGamma_term + log(gamma(0.5*(v+1-j)));
end
logIW_term = 0.5*(v+D+1) * log(det(IW));
inv_IW = pinv(IW);
trace_term = 0.5*trace(S * inv_IW);

logpdf = logS_term - log2_term - logGamma_term - logIW_term - trace_term;

pdf = exp(logpdf);
end