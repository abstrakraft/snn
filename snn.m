function y = snn(p, net, x)

y = x;
p_cursor = 1;
for ldx = 1:(length(net)-1)
	W_size = [net(ldx+1) net(ldx)+1];
	W_numel = prod(W_size);
	W = reshape(p(p_cursor:(p_cursor+W_numel-1)), W_size);
	p_cursor = p_cursor + W_numel;
	y = snn_f(W*[y; ones([1 size(x, 2)])]);
end
