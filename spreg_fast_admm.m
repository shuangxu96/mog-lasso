function [z, history] = spreg_fast_admm(A, b, lambda, rho)
% lasso  Solve lasso problem via fast ADMM with restart
% adapted from https://web.stanford.edu/~boyd/papers/admm/lasso/lasso.html
% [z, history] = lasso(A, b, lambda, rho, alpha);
%
% Solves the following problem via ADMM:
%
%   minimize 1/2*|| Ax - b ||_2^2 + \lambda || x ||_1
%
% The solution is returned in the vector x.
%
% history is a structure that contains the objective value, the primal and
% dual residual norms, and the tolerances for the primal and dual residual
% norms at each iteration.
%
% rho is the augmented Lagrangian parameter.
%
%
%% Global constants and defaults

MAX_ITER = 50;
ABSTOL   = 1e-4;
RELTOL   = 1e-2;
eta = 0.999;
%% Data preprocessing

[m, n] = size(A);

% save a matrix-vector multiply
Atb = A'*b;
%% ADMM solver

z_old = zeros(n,1);
u_old = zeros(n,1);
zhat = z_old;
uhat = u_old;
alpha = 1;
alpha_old = 0;
c_old = 0;
% cache the factorization
[L, U] = factor(A, rho);

for k = 2:MAX_ITER

    % x-update
    q = Atb + rho*(zhat - uhat);    % temporary value
    if( m >= n )    % if skinny
       x = U \ (L \ q);
    else            % if fat
       x = q/rho - (A'*(U \ ( L \ (A*q) )))/rho^2;
    end

    % z-update
    z = shrinkage(x + uhat, lambda/rho);

    % u-update
    xz = x - z;
    u = uhat + xz;

    % c-update
    zz = z - zhat;
    c = rho * (xz' * xz + zz' * zz);
    if c < eta * c_old
        alpha_new = 1 + sqrt(1 + 4 * alpha ^ 2);
        alpha_factor = alpha_old / alpha_new;
        zhat_new = z + (z - z_old) * alpha_factor;
        uhat_new = u + (u - u_old) * alpha_factor;
    else
        alpha_new = 1;
        zhat_new = z_old;
        uhat_new = u_old;
        c = c_old / eta;
    end
    
    % new parameter update
    z_old = z; u_old = u; c_old = c;
    alpha_old = alpha; alpha = alpha_new;
    zhat = zhat_new; uhat = uhat_new;
    
    % diagnostics, reporting, termination checks
    history.objval(k)  = objective(A, b, lambda, x, z);

    history.r_norm(k)  = norm(x - z);
    history.s_norm(k)  = norm(-rho*(z - z));

    history.eps_pri(k) = sqrt(n)*ABSTOL + RELTOL*max(norm(x), norm(-z));
    history.eps_dual(k)= sqrt(n)*ABSTOL + RELTOL*norm(rho*u);

    if (history.r_norm(k) < history.eps_pri(k) && ...
       history.s_norm(k) < history.eps_dual(k))
         break;
    end

end
end

function p = objective(A, b, lambda, x, z)
    p = ( 1/2*sum((A*x - b).^2) + lambda*norm(z,1) );
end

function z = shrinkage(x, kappa)
z = max( 0, x - kappa ) - max( 0, -x - kappa );
end

function [L, U] = factor(A, rho)
    [m, n] = size(A);
    if ( m >= n )    % if skinny
       L = chol( A'*A + rho*speye(n), 'lower' );
    else            % if fat
       L = chol( speye(m) + 1/rho*(A*A'), 'lower' );
    end

    % force matlab to recognize the upper / lower triangular structure
    L = sparse(L);
    U = sparse(L');
end