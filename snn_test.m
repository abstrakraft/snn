function snn_test()

p = randn([43 1]);
x = randn([1 1]);
y = randn([3 1]);

J_numerical = numerical_jacobian(p, @snn_cost, {x, y});
J_computed = snn_derivatives(p, x, y);

J_numerical - J_computed
