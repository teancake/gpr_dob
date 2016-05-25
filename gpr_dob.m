function [out1, out2, out3, out4] = gpr_dob(logtheta, x, y, xd, dy, idx, xstar)

% gpr_dob - Gaussian process regression with derivative observations.
%
% This program is modified upon the gpr function in the GPML toolbox. The
% original declaration is as follows.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% gpr - Gaussian process regression, with a named covariance function. Two
% modes are possible: training and prediction: if no test data are given, the
% function returns minus the log likelihood and its partial derivatives with
% respect to the hyperparameters; this mode is used to fit the hyperparameters.
% If test data are given, then (marginal) Gaussian predictions are computed,
% whose mean and variance are returned. Note that in cases where the covariance
% function has noise contributions, the variance returned in S2 is for noisy
% test targets; if you want the variance of the noise-free latent function, you
% must substract the noise variance.
%
% usage: [nlml dnlml] = gpr(logtheta, covfunc, x, y, xd, dy)
%    or: [mu S2]  = gpr(logtheta, covfunc, x, y, xd, dy, xstar)

%
% where:
%
%   logtheta is a (column) vector of log hyperparameters
%   covfunc  is the covariance function
%   x        is a n by D matrix of training inputs
%   y        is a (column) vector (of size n) of targets
%   xstar    is a nn by D matrix of test inputs
%   nlml     is the returned value of the negative log marginal likelihood
%   dnlml    is a (column) vector of partial derivatives of the negative
%                 log marginal likelihood wrt each log hyperparameter
%   mu       is a (column) vector (of size nn) of prediced means
%   S2       is a (column) vector (of size nn) of predicted variances
%
% For more help on covariance functions, see "help covFunctions".
%
% (C) copyright 2006 by Carl Edward Rasmussen (2006-03-20).
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% In order to incorporate derivative observations, the following modifications
% are made.
% input added:
%   xd	     is a nd by D matrix of training inputs where derivatives are
%		observed.
%   dy	     is a column vector of ndxD by 1 of derivatives at xd.
%   idx	     is a subvector of [1:D], determining the dimensions of x 
%	       along whichhave derivative observations are available.
%
% input deleted:
%   covfunc is fixed to covSEard + covNoise
%
% usage:
%   [nlml dnlml] = gpr_dob(logtheta, x, y, xd, dy)
%   [mu S2]  = gpr_dob(logtheta, x, y, xd, dy, xstar)
%   [mu S2 dmu_dxstar dS2_dxstar] = gpr_dob(logtheta, x, y, xd, dy, xstar)
%
% output added:
%   dmu_dxstar	is a Dx1 (column) vector of the derivative of mu w.r.t. xstar.
%   dS2_dxstar	is a DxDxD matrix of the derivative of S2 w.r.t. xstar.
%
% Additionally, this program requires two covariance function files
% 'covSEardN_dob_dy.m' and 'covSEardN_dob_y.m' which implement the covariance
% function and its gradient w.r.t. the hyper-parameters.
%
% This program is not optimised in its performance, thus may be slow.
% Xiaoke Yang <das.xiaoke@hotmail.com> (2016-02-17)

[n, D] = size(x);
if size(logtheta, 1) ~= D+2
    error('Error: Number of hyperparameters do not agree with input dimension.');
end

if nargin < 4;
    xd=[]; dy=[]; idx = [];
end

if nargin == 5
    idx = [];
end

if isempty(idx)
    idx = 1:D;
end

[nd, ~] = size(xd);

% With derivative observation, the covariance matrix needs to be computed
% block by block
Kxx = feval('covSEardN_dob_y',logtheta, x);	     % output covariance matrix
Kdd = feval('covSEardN_dob_dy',logtheta, xd, [], [], idx); % derivative cov mat
Kdx = feval('covSEardN_dob_dy',logtheta, xd, x, [], idx);  % cross cov mat
Kxd = Kdx';
K = [Kdd+1e-6*eye(size(Kdd)) Kdx;
    Kxd Kxx];
L = chol(K)';                        % cholesky factorization of the covariance
alpha = solve_chol(L',[dy; y]);

if nargin <= 6 % if no test cases, compute the negative log marginal likelihood
    out1 = 0.5*[dy; y]'*alpha + sum(log(diag(L))) + 0.5*(n+nd*D)*log(2*pi);
    
    if nargout > 1             % ... and if requested, its partial derivatives
        out2 = zeros(size(logtheta));  % set the size of the derivative vector
        W =  L'\(L\eye(nd*numel(idx)+n))-alpha*alpha';	 % precompute for convenience
        for i = 1:length(out2)
            dKxx = feval('covSEardN_dob_y',logtheta, x, [], i);
            [~, dKdd] = feval('covSEardN_dob_dy',logtheta, xd, [], i, idx);
            [~, dKdx] = feval('covSEardN_dob_dy',logtheta, xd, x, i, idx);
            dKxd = dKdx';
            dK = [dKdd dKdx; dKxd dKxx];
           % out2(i) = sum(sum(W.*dK))/2
           % this trace operation is equivalent to the above sum method.
            out2(i) = trace(W'*dK)/2;
        end
    end
    
else                    % ... otherwise compute (marginal) test predictions ...
    Kss = feval('covSEardN_dob_y', logtheta, xstar, 'diag');    % self-variance
    Kxs = feval('covSEardN_dob_y',logtheta, x, xstar);
    Kds = feval('covSEardN_dob_dy',logtheta, xd, xstar, [], idx);
    Ks = [Kds; Kxs];
    
    out1 = Ks' * alpha;                                    % predicted means
    
    if nargout > 1
        v = L\Ks;
        out2 = Kss - sum(v.*v)';
    end
end
if nargout > 2			% compute the derivatives w.r.t. the test input
    if isempty(xd)
        Kd_ds = [];
    else
        xds = [xd; xstar];
        Kd_ds = feval('covSEardN_dob_dy',logtheta, xds, [], [], 1:D);
        tmp = zeros(D,1); tmp(idx)=1; tmp=repmat(tmp,nd,1);
        Kd_ds = Kd_ds(tmp==1,end-D+1:end);
    end
    Kd_sx = feval('covSEardN_dob_dy',logtheta, xstar, x, [], 1:D);
    Kd = [Kd_ds' Kd_sx];
    out3 = Kd*alpha;
    invK = L'\(L\eye(nd*numel(idx)+n));
    out4 = -2*Kd*invK*Ks;
end
