% checkgrad_gpr_dob - check the gradient of the gpr_dob program
% two columns of results will be displayed, the left one of which (dy) is
% computed by the gpr_dob program and the right one (dh) through 
% numerical differentiation.
% d = norm(dh-dy)/norm(dh+dy);   
%
% Xiaoke Yang <das.xiaoke@hotmail.com> (2016-02-17)

% add path of the gpr_dob program
addpath('..')

% switches, change to 1 if to be checked 
test_cov = 1;	    % derivative of the covariance function covSEardN_dob_y
test_covd = 1;	    % derivative of the covariance function covSEardN_dob_dy
test_cov_mat = 1;   % derivative of the covSEardN_dob_dy in a matrix form 
test_dmdx = 1;	    % derivative: dmu_dxstar
test_dsdx = 1;	    % derivative: dS2_dxstar
test_nlml = 1;	    % derivative of nlml

% dimensions
D = 3;	    % input dimension
N = 1;	    % number of input-output data points
ND = 2;     % number of derivative data points

% step size
dh = 1e-6;

% random hyper-parameters
ell = 0.01+2*abs(randn(D,1));	% length scales 
sf = 2*abs(randn(1));		% function magnitude
sn = 0.5*abs(randn(1))+0.01;	% noise magnitude
hyp = log([ell; sf; sn]);	% log-hyperparameters

% random training data
XD = randn(ND,D)+0.1;		% derivative data input
X = randn(N,D);			% input-output data input
Y = randn(N,1)+0.2;		% input-output data output
idx = 1:D;			% availability indices of the derivatives
DY = randn(ND*numel(idx),1);	% derivative data output
xstar = randn(D,1);		% test input

% check covariance function covSEardN_dob_y
if test_cov
    disp('### Checking derivatives of covariance function covSEardN_dob_y ...');
    sumd = 0;
    for i = 1:size(X,1)
        for j = 1:size(X,1)
            [d, dy] = checkgrad('gpr_dob_cov_y_checkgrad_wrapper', hyp, dh, X, X, i, j);
            sumd = sumd +d;
        end
    end
    fprintf('%2.5g\n',sumd)
end

% check covariance function covSEardN_dob_dy
if test_covd
    disp('### Checking derivatives of covariance function covSEardN_dob_dy ...');
    sumd = 0;
    for i = 1:size(XD,1)
        for j = 1:size(X,1)
            [d, dy] = checkgrad('gpr_dob_cov_dy_checkgrad_wrapper', hyp, dh, XD, X, i, j, idx);
            sumd = sumd +d;
        end
    end
    fprintf('%2.5g\n',sumd)
end

% check derivative: dmu_dxstar 
if test_dmdx
    disp('### Checking derivative: dmu_dxstar ...');
    d = checkgrad('gpr_dob_dmdx_checkgrad_wrapper', xstar, dh,  hyp, X, Y, XD, DY, idx)
end

% check derivative: dS2_dxstar 
if test_dsdx
    disp('### Checking derivative: dS2_dxstar ...');
    d = checkgrad('gpr_dob_dsdx_checkgrad_wrapper', xstar, dh,  hyp, X, Y, XD, DY, idx)
end

if test_nlml
    disp('### Checking derivative of nlml...')
    d = checkgrad('gpr_dob', hyp, dh, X, Y, XD, DY, idx)
end

if test_cov_mat
    disp('### Checking derivatives of covSEardN_dob_dy in a matrix format ...');
    disp('### Checking derivatives of k(XD,XD)...');
    % kdd
    for i = 1:D
        gpr_dob_cov_checkgrad_subfun(dh, hyp, XD, [], i, idx)
    end
    disp('### Checking derivatives of k(XD,X)...');
    % kdx
    for i = 1:D
        gpr_dob_cov_checkgrad_subfun(dh, hyp, XD, X, i, idx)
    end
end

