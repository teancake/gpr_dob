function [K, dK] = covSEardN_dob_dy(loghyper, xd, x, l, idx)
% covSEardN_dob_dy: the sum of a covSEard and a covNoise covariance function.
% computation of covariance matrices of derivative observations.
%
% This program is modified upon the covSEard function in the GPML toolbox. The
% original declaration is as follows.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Squared Exponential covariance function with Automatic Relevance Detemination
% (ARD) distance measure. The covariance function is parameterized as:
%
% k(x^p,x^q) = sf2 * exp(-(x^p - x^q)'*inv(P)*(x^p - x^q)/2)
%
% where the P matrix is diagonal with ARD parameters ell_1^2,...,ell_D^2, where
% D is the dimension of the input space and sf2 is the signal variance. The
% hyperparameters are:
%
% loghyper = [ log(ell_1)
%              log(ell_2)
%               .
%              log(ell_D)
%              log(sqrt(sf2)) ]
%
% For more help on design of covariance functions, try "help covFunctions".
%
% (C) Copyright 2006 by Carl Edward Rasmussen (2006-03-24)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% input:
%   loghyper is a (column) vector of log hyperparameters
%   xd	     is a nd by D matrix of training inputs where derivatives are
%		observed.
%   x        is a n by D matrix of training inputs
%   l	     use with dK output, dK w.r.t the l^th hyper-parameter
%   idx	     is a subvector of [1:D], determining the dimensions of x 
%	       along whichhave derivative observations are available.
%
% usage:
%   [K, dK] = covSEardN_dob_dy(loghyper, xd, [], [], idx);
%   [K, dK] = covSEardN_dob_dy(loghyper, xd, x, [], idx);
%   [K, dK] = covSEardN_dob_dy(loghyper, xd, x, l, idx);
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


if nargin < 2, K = '(D+2)'; return; end	    % report number of parameters
if isempty(xd), K=[]; dK=[]; return; end    % check whether xd is empty or not
xempty = isempty(x);
lempty = isempty(l);
[nd, D] = size(xd);

F = numel(idx);                                 % number of dimensions

if lempty
    l = D+2;
end

ell = exp(loghyper(1:D));                         % characteristic length scale
sf2 = exp(2*loghyper(D+1));                                   % signal variance
sn2 = exp(2*loghyper(D+2));                                    % noise variance

Gam = diag(1./ell(idx)).^2;              % Gamma
Gamh = diag(1./ell(idx));                % half Gamma
            
if xempty   % compute covariances of derivative observations and their gradient
    K = zeros(nd*F,nd*F);
    dK = zeros(nd*F,nd*F);
    for i = 1:nd
        for j = 1:nd
            Kij = sf2*exp(-sq_dist(Gamh*xd(i,idx)',Gamh*xd(j,idx)')/2);  % Kf
            xdif = (xd(i,idx)-xd(j,idx))';           % x difference 
            xdifs = (xdif*xdif');                    % square
            coef = Gam - Gam*xdifs*Gam;              % coefficient of Kf
            K((i-1)*F+1:i*F,(j-1)*F+1:j*F) = coef*Kij;
            if l == D+2                  % d. w.r.t. loghyper(D+2), zero matrix
            elseif l == D+1                           % d. w.r.t. loghyper(D+1)
                dK = K*2;
            else                                      % d. w.r.t. loghyper(1:D)
                Delta_l = zeros(F,1); Delta_l(idx==l) = 1;
                Delta_l = diag(Delta_l);
                coef1 = Delta_l - Delta_l*xdifs*Gam - Gam*xdifs*Delta_l;
                coef2 = coef*(-1/2*(xd(i,l)-xd(j,l))^2);
                dK1 = coef1*Kij;
                dK2 = coef2*Kij;
                dK((i-1)*F+1:i*F,(j-1)*F+1:j*F) = -2*ell(l)^(-2)*(dK1+dK2);
            end
        end
    end
else % compute cross-covariances between derivative and output and the gradient
    [n, ~] = size(x);
    K = zeros(nd*F,n);
    dK = zeros(nd*F,n);
    for i = 1:nd
        for j = 1:n
            Kij = sf2*exp(-sq_dist(Gamh*xd(i,idx)',Gamh*x(j,idx)')/2);  % Kf
            coef = -Gam*(xd(i,idx)'-x(j,idx)');
            K((i-1)*F+1:i*F,j) = coef*Kij;
            if l == D+2                  % d. w.r.t. loghyper(D+2), zero matrix
            elseif l == D+1
                dK = K*2;                             % d. w.r.t. loghyper(D+1)
            else
                Delta_l = zeros(F,1); Delta_l(idx==l) = 1;
                Delta_l = diag(Delta_l);
                coef1 = -Delta_l*(xd(i,idx)'-x(j,idx)');
                coef2 = coef*(-1/2*(xd(i,l)-x(j,l))^2);
                dK1 = coef1*Kij;
                dK2 = coef2*Kij;            
                dK((i-1)*F+1:i*F,j) = -2*(dK1+dK2)*ell(l)^(-2);
            end
        end
    end
end
