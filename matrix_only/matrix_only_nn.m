function C = matrix_only_nn(p, x, y)

W5 = reshape(p(1:20), [5 4]);
W4 = reshape(p(21:28), [4 2]);
W3 = reshape(p(29:36), [2 4]);
W2 = reshape(p(37:48), [4 3]);
W1 = reshape(p(49:66), [3 6]);

C = sum((W5*W4*W3*W2*W1*x - y).^2);
