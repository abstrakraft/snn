function C = snn_cost(p, net, x, y)

C = sum((snn(p, net, x) - y).^2, 1)/2;
