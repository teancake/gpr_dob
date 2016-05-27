function [y, dy] = gpr_dob_cov_y_checkgrad_wrapper(loghyper, x, z, i, j)
% gpr_dob_cov_y_checkgrad_wrapper - wrapper function to check the gradient
% of covariance function covSEardN_dob_y
% This is a sub-function called by the main checking program
% checkgrad_gpr_dob
%
% Xiaoke Yang <das.xiaoke@hotmail.com> (2016-02-17)

K = covSEardN_dob_y(loghyper, x, z);
y = K(i,j);
tmp = [];
for l = 1:numel(loghyper)
    Kd =  covSEardN_dob_y(loghyper, x, z, l);
    tmp = [tmp; Kd(i,j)];
end
dy = tmp;
