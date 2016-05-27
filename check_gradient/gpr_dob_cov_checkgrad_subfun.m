function [d, dy] = gpr_dob_cov_checkgrad_subfun(e, logtheta, xd, x, i, idx)
% gpr_dob_cov_checkgrad_subfun - check the gradient of the covSEardN_dob_dy
% covariance function.
% This is a sub-function called by the main checking program
% checkgrad_gpr_dob
%
% Xiaoke Yang <das.xiaoke@hotmail.com> (2016-02-17)


if isempty(x)
    % k(xd,xd)
    [~, dy] = feval('covSEardN_dob_dy',logtheta, xd, [], i, idx);
    dlogtheta = zeros(size(logtheta));
    dlogtheta(i) = dlogtheta(i) + e;
    
    logtheta1 = logtheta - dlogtheta;
    logtheta2 = logtheta + dlogtheta;
    y1 = feval('covSEardN_dob_dy',logtheta1, xd, [], i, idx);
    y2 = feval('covSEardN_dob_dy',logtheta2, xd, [], i, idx);
    dh = (y2 - y1)/(2*e);
    d = norm(dh-dy)/norm(dh+dy);       % return norm of diff divided by norm of sum
else
    % k(xd,x)
    [~, dy] = feval('covSEardN_dob_dy',logtheta, xd, x, i, idx);
    dlogtheta = zeros(size(logtheta));
    dlogtheta(i) = dlogtheta(i) + e;
    
    logtheta1 = logtheta - dlogtheta;
    logtheta2 = logtheta + dlogtheta;
    y1 = feval('covSEardN_dob_dy',logtheta1, xd, x, i, idx);
    y2 = feval('covSEardN_dob_dy',logtheta2, xd, x, i, idx);
    dh = (y2 - y1)/(2*e);
    d = norm(dh-dy)/norm(dh+dy);       % return norm of diff divided by norm of sum
end



