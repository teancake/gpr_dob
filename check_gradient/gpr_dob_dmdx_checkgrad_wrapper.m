function [y, dy] = gpr_dob_dmdx_checkgrad_wrapper(xstar,logtheta, x, y, xd, dy, idx)
% gpr_dob_dmdx_checkgrad_wrapper - wrapper function to check the gradient
% of dmu_dxstar of gpr_dob
% This is a sub-function called by the main checking program
% checkgrad_gpr_dob
%
% Xiaoke Yang <das.xiaoke@hotmail.com> (2016-02-17)

if size(xstar,2) == 1
    xstar = xstar';
end
[y, ~, dy, ~] = gpr_dob(logtheta, x, y, xd, dy, idx, xstar);

