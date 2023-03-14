function matrix_only_nn_test()

p = randn([66 1]);
x = randn([6 1]);
y = randn([5 1]);

J_numerical = numerical_jacobian(p, @matrix_only_nn, {x, y})
J_computed = matrix_only_nn_derivatives(p, x, y)
