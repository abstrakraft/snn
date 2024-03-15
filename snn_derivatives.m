function J = snn_derivatives(p, net, x, y)

layer_count = length(net)-1;
W = {};
z = {};
a = {x};
p_cursor = 1;
for ldx = 1:layer_count
	W_size = [net(ldx+1) net(ldx)+1];
	W_numel = prod(W_size);
	W{end+1} = reshape(p(p_cursor:(p_cursor+W_numel-1)), W_size);
	p_cursor = p_cursor + W_numel;

	z{end+1} = W{end} * [a{end}; 1];
	a{end+1} = snn_f(z{end});
end

J = zeros([1 p_cursor-1]);
err = a{end} - y;
prev_partial = err.' .* snn_fdot(z{end}).';
for ldx = layer_count:-1:1
	W_numel = numel(W{ldx});
	p_cursor = p_cursor - W_numel;
	W_dot = prev_partial.' * [a{ldx}; 1].';
	J(p_cursor:(p_cursor+W_numel-1)) = reshape(W_dot, [1 W_numel]);
	if ldx > 1
		pp = prev_partial;
		prev_partial = pp * W{ldx}(:,1:end-1) .* snn_fdot(z{ldx-1}).';
	end
end

%p5 = err.' .* snn_fdot(z5).';
%p4 = p5 * W5 .* snn_fdot(z4).';
%p3 = p4 * W4 .* snn_fdot(z3).';
%p2 = p3 * W3 .* snn_fdot(z2).';
%p1 = p2 * W2 .* snn_fdot(z1).';
%W5_dot = p5.' * a4.';
%W4_dot = p4.' * a3.';
%W3_dot = p3.' * a2.';
%W2_dot = p2.' * a1.';
%W1_dot = p1.' * x.';

%J = [reshape(W5_dot, [15 1]); reshape(W4_dot, [40 1]); reshape(W3_dot, [32 1]); reshape(W2_dot, [24 1]); reshape(W1_dot, [6 1])]';
