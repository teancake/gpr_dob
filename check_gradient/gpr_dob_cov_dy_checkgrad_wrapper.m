function [y, dy] = gpr_dob_cov_dy_checkgrad_wrapper(loghyper, xd, x, i, j, idx)
% gpr_dob_cov_dy_checkgrad_wrapper - wrapper function to check the gradient
% of covariance function covSEardN_dob_dy
% This is a sub-function called by the main checking program
% checkgrad_gpr_dob
%
% Xiaoke Yang <das.xiaoke@hotmail.com> (2016-02-17)
K = covSEardN_dob_dy(loghyper, xd, x, [], idx);
y = K(i,j);
tmp = [];
for l = 1:numel(loghyper)
    [~, Kd] =  covSEardN_dob_dy(loghyper, xd, x, l, idx);
    tmp = [tmp; Kd(i,j)];
end
dy = tmp;
