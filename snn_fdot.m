function y = snn_fdot(x)
%z = snn_f(x);
%y = z.*(1-z);
%y = ones(size(x));
y = 1./(1+exp(-x));
%y = 1-z.^2;
