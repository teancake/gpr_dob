% This program test the one dimensional case of gpr_dob.m which implements Gaussian
% process regression with derivative observations.
%
% Xiaoke Yang <das.xiaoke@hotmail.com> (2016-02-17)


close all
write_fig = 0;

ell = 1; sf = 1; sn = 0.01;
hyp = log([ell; sf; sn]);

% training data
X = [-3 -2 1]';
Y = [1.5 0 -1.5]';
XD = X;
DY = [-2 -1 0]';

% make predictions without training
nlml = gpr_dob(hyp, X, Y);
z = linspace(-5, 5, 101)';
[m, s2] = gpr_dob(hyp, X, Y, [], [], [], z);


figure(2)
f = [m+2*sqrt(s2); flip(m-2*sqrt(s2),1)];
fill([z; flip(z,1)], f, [7 7 7]/8);
hold on; plot(z, m, 'LineWidth', 2); plot(X, Y, '+', 'MarkerSize', 12)
grid on
xlabel('input, x')
ylabel('output, y')
if write_fig, print -depsc f2.eps; end


[m, s2] = gpr_dob(hyp, X, Y, XD, DY, [], z);

figure(3)
f = [m+2*sqrt(s2); flip(m-2*sqrt(s2),1)];
fill([z; flip(z,1)], f, [7 7 7]/8);
hold on; plot(z, m, 'LineWidth', 2); plot(X, Y, '+', 'MarkerSize', 12)
grid on
xlabel('input, x')
ylabel('output, y')
for i=1:numel(XD)
    d = 0.5/sqrt(1+DY(i)^2);
    xtmp = [XD(i)-d:d/100:XD(i)+d];
    ytmp = (xtmp-XD(i)).*DY(i)+Y(i);
    plot(xtmp,ytmp,'r','LineWidth',2);
end
if write_fig, print -depsc f3.eps; end

