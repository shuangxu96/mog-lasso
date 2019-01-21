function [BETA, W, meanBIC, meanRSS, lambda] = cvmogspreg(X, y, nfold, nlambda, k, rho)
if nargin<=6
    rho = 1;
end
if nargin<=5
    k = 2;
end
    
[n,p] = size(X);
v = zeros(p, 1);
for j = 1: (p); v(j) = X(:,j)' * X(:,j)*0.1;end
lambda_max = max(abs(X'*y)./v);  %mcp/scad
lambda = logspace(log10(lambda_max)-5, log10(lambda_max), nlambda);

for i = 1:nlambda
    [BETA(:,i), ~,~,~, W(:,i)] = mogspreg(X, y, lambda(i), k, rho);
end
ind0 = sum(BETA,1)~=0;
if length(ind0)>=2
    lambda = lambda(ind0(2:end));
    nlambda = length(lambda);
    BETA(:,~ind0(2:end)) = [];
    W(:,~ind0(2:end)) = [];
end

cvid = 1 + mod((1:n)',nfold);
cvid = randsample(cvid, n, false);

for nf = 1:nfold
    Xtest = X(cvid == nf,:);
    ytest = y(cvid == nf,:);
    Xtrain = X(cvid ~= nf,:);
    ytrain = y(cvid ~= nf,:);
    beta = [];
    for i = 1:nlambda
        [beta(:,i), ~,~,~, ~] = mogspreg(Xtrain, ytrain, lambda(i), k, rho);
    end
    yhat = Xtest*beta;
    cvRSS(nf,:) = sum(bsxfun(@minus,yhat,ytest).^2,1);
    p1 = sum(beta~=0, 1);
    cvBIC(nf,:) = cvRSS(nf,:) + p1*log(n);
end

meanRSS = mean(cvRSS);
meanBIC = mean(cvBIC);
