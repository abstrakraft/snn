function C = snn_cost(p, x, y)

C = sum((snn(p, x) - y).^2, 1)/2;
