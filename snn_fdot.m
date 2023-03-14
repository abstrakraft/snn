function y = snn_fdot(x)
z = snn_f(x);
y = z.*(1-z);
