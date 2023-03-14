function J = snn_derivatives(p, x, y)

W5 = reshape(p(1:12), [3 4]);
W4 = reshape(p(13:20), [4 2]);
W3 = reshape(p(21:28), [2 4]);
W2 = reshape(p(29:40), [4 3]);
W1 = reshape(p(41:43), [3 1]);

z1 = W1*x;
a1 = snn_f(z1);
z2 = W2*a1;
a2 = snn_f(z2);
z3 = W3*a2;
a3 = snn_f(z3);
z4 = W4*a3;
a4 = snn_f(z4);
z5 = W5*a4;
a5 = snn_f(z5);
err = a5 - y;
p5 = (err .* snn_fdot(z5)).';
p4 = p5 * W5 .* snn_fdot(z4).';
p3 = p4 * W4 .* snn_fdot(z3).';
p2 = p3 * W3 .* snn_fdot(z2).';
p1 = p2 * W2 .* snn_fdot(z1).';
W5_dot = p5.' * a4.';
W4_dot = p4.' * a3.';
W3_dot = p3.' * a2.';
W2_dot = p2.' * a1.';
W1_dot = p1.' * x.';

J = [reshape(W5_dot, [12 1]); reshape(W4_dot, [8 1]); reshape(W3_dot, [8 1]); reshape(W2_dot, [12 1]); reshape(W1_dot, [3 1])]';
