function y = snn_f(x)

%y = 1./(1+exp(-x));
%y = x;
y = log(1+exp(x));
%z1 = exp(x);
%z2 = exp(-x);
%y = (z1-z2)./(z1+z2);
