function A = covSEardN_dob_y(hyp, x, z, i)
% covSEardN_dob_y: the sum of a covSEard and a covNoise covariance function.
% computation of covariance matrices of output points.
% %
% This program is modified upon the covSEard function in the GPML toolbox. The
% original declaration is as follows.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Squared Exponential covariance function with Automatic Relevance Detemination
% (ARD) distance measure. The covariance function is parameterized as:
%
% k(x^p,x^q) = sf^2 * exp(-(x^p - x^q)'*inv(P)*(x^p - x^q)/2)
%
% where the P matrix is diagonal with ARD parameters ell_1^2,...,ell_D^2, where
% D is the dimension of the input space and sf2 is the signal variance. The
% hyperparameters are:
%
% hyp = [ log(ell_1)
%         log(ell_2)
%          .
%         log(ell_D)
%         log(sf) ]
%
% Copyright (c) by Carl Edward Rasmussen and Hannes Nickisch, 2010-09-10.
%
% See also COVFUNCTIONS.M.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% input:
%   loghyper is a (column) vector of log hyperparameters
%   x        is a n by D matrix of training inputs
%   z        is another n by D matrix of training inputs
%   l	     derivative of the covariance matrix w.r.t the l^th hyper-parameter
% usage:
%   K = covSEardN_dob_y(loghyper, x);
%   dK = covSEardN_dob_y(loghyper, x, [], l);
%   K = covSEardN_dob_y(loghyper, x, 'diag');
%
% output added:
%   K	    covariance matrix of xd, or covariance matrix between xd and x.
%   dK	    derivative of K w.r.t. the l^th hyper-parameter.
%
% This program is supposed to run with gpr_dob.m which implements Gaussian
% process regression with derivative observations.
%
% This program is not optimised in its performance, thus may be slow.
% Xiaoke Yang <das.xiaoke@hotmail.com> (2016-02-17)



if nargin<2, K = '(D+2)'; return; end              % report number of parameters
if nargin<3, z = []; end                                   % make sure, z exists
xeqz = isempty(z); dg = strcmp(z,'diag');                       % determine mode

[n,D] = size(x);
ell = exp(hyp(1:D));                               % characteristic length scale
sf2 = exp(2*hyp(D+1));                                         % signal variance
sn2 = exp(2*hyp(D+2));                                     % noise variance

% precompute squared distances
if dg                                                               % vector kxx
    K = zeros(size(x,1),1);
    Kn = 0;
else
    if xeqz                                                 % symmetric matrix Kxx
        K = sq_dist(diag(1./ell)*x');
        Kn = eye(n)*sn2;
    else                                                   % cross covariances Kxz
        K = sq_dist(diag(1./ell)*x',diag(1./ell)*z');
        Kn = 0;
    end
end

K = sf2*exp(-K/2);                                               % covariance
A = K+Kn;
if nargin>3                                                        % derivatives
    if i<=D                                              % length scale parameters
        if dg
            K = K*0;
        else
            if xeqz
                K = K.*sq_dist(x(:,i)'/ell(i));
            else
                K = K.*sq_dist(x(:,i)'/ell(i),z(:,i)'/ell(i));
            end
        end
    elseif i==D+1                                            % magnitude parameter
        K = 2*K;
    elseif i==D+2                                       % noise variance parameter
        if xeqz
            K = eye(n)*2*sn2;
        else
            K = zeros(n,n);
        end
        
    else
        error('Unknown hyperparameter')
    end
    A = K;
end
