function snn_test()

net = [1 6 4 8 5 3];
p_len = sum(net(1:(end-1)) .* net(2:end));
p = randn([p_len 1]);
x = randn([net(1) 1]);
y = randn([net(end) 1]);

J_numerical = numerical_jacobian(p, @snn_cost, {net, x, y});
J_computed = snn_derivatives(p, net, x, y);

%J_numerical - J_computed
diff = (J_numerical - J_computed)/max(abs(J_numerical));
max(abs(diff(:)))
