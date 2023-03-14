function y = snn(p, x)

W5 = reshape(p(1:12), [3 4]);
W4 = reshape(p(13:20), [4 2]);
W3 = reshape(p(21:28), [2 4]);
W2 = reshape(p(29:40), [4 3]);
W1 = reshape(p(41:43), [3 1]);

y = snn_f(W5*snn_f(W4*snn_f(W3*snn_f(W2*snn_f(W1*x)))));
