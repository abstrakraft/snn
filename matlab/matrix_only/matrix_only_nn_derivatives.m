function J = matrix_only_nn_derivatives(p, x, y)

W5 = reshape(p(1:20), [5 4]);
W4 = reshape(p(21:28), [4 2]);
W3 = reshape(p(29:36), [2 4]);
W2 = reshape(p(37:48), [4 3]);
W1 = reshape(p(49:66), [3 6]);

C = sum((W5*W4*W3*W2*W1*x - y).^2);

err = W5*W4*W3*W2*W1*x - y;

W5_dot = 2* err * (W4*W3*W2*W1*x).';
W4_dot = 2* (err.' * W5).' * (W3*W2*W1*x).';
W3_dot = 2* (err.' * W5*W4).' * (W2*W1*x).';
W2_dot = 2* (err.' * W5*W4*W3).' * (W1*x).';
W1_dot = 2* (err.' * W5*W4*W3*W2).' * x.';

J = [reshape(W5_dot, [20 1]); reshape(W4_dot, [8 1]); reshape(W3_dot, [8 1]); reshape(W2_dot, [12 1]); reshape(W1_dot, [18 1])]';
