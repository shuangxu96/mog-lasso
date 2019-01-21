function [beta, sigma2, pik, gamma,w] = mogspreg(X, y, lambda, k, rho)

[n,p] = size(X);
maxiter = 100;
tol = 1e-4;
res = y;
betaold = zeros(p,1);
gamma = rand(n,k);
gamma = bsxfun(@rdivide, gamma, sum(gamma,2));
for i = 1:maxiter
    % M step
    [pik, sigma2] = mstep(X, gamma, res);
    w = sqrt(sum(bsxfun(@rdivide, gamma, sigma2), 2));
    w = n*w/sum(w);
    wX = bsxfun(@times, X, w);
    wy = bsxfun(@times, y, w);
    beta = spreg_fast_admm(wX, wy, lambda, rho);
    res = y - X * beta;
    % E step
    gamma = estep(res, sigma2, pik);
    % converge?
    if sum(abs(betaold-beta))<tol
        break;
    else
        betaold = beta;
    end
end
end


function [pik, sigma2] = mstep(X, gamma, res)
n = size(X,1);
nk = sum(gamma,1);
pik = nk/n;
res2 = res.^2;
sigma2 = bsxfun(@rdivide, sum(bsxfun(@times, gamma, res2), 1), nk);
end %-end mstep

function [gamma, llh] = estep(res, sigma2, pik)
n = length(res);
k = length(pik);
logRho = zeros(n,k);

for i = 1:k
    logRho(:,i) = loggausspdf(res,0,sigma2(i));
end
logRho = bsxfun(@plus,logRho,log(pik));
T = logsumexp(logRho,2);
llh = sum(T)/n; % loglikelihood
logR = bsxfun(@minus,logRho,T);
gamma = exp(logR);
end %-end estep

function y = loggausspdf(X, mu, sigma2)
X = bsxfun(@minus,X,mu);
q = X.^2 / sigma2;
c = log(2*pi*sigma2);   % normalization constant
y = -(c+q)/2;
end %-end loggausspdf

function s = logsumexp(x, dim)
% Compute log(sum(exp(x),dim)) while avoiding numerical underflow.
%   By default dim = 1 (columns).
% Written by Michael Chen (sth4nth@gmail.com).
if nargin == 1
    % Determine which dimension sum will use
    dim = find(size(x)~=1,1);
    if isempty(dim), dim = 1; end
end

% subtract the largest in each column
y = max(x,[],dim);
x = bsxfun(@minus,x,y);
s = y + log(sum(exp(x),dim));
i = find(~isfinite(y));
if ~isempty(i)
    s(i) = y(i);
end 
end %-end logsumexp
