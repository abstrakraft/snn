function p = snn_train()

N = 100000;

p = randn([43 1]);
x_train = rand([1 N]) * 2*pi;
y_train = [sin(x_train); cos(x_train); tan(x_train)];
x_eval = rand([1 100]) * 2*pi;
y_eval = [sin(x_eval); cos(x_eval); tan(x_eval)];
step = 1e-6;

cost_history = [];
for trial = 1:N
	x = x_train(1, trial);
	y = y_train(:, trial);
	C = snn_cost(p, x, y);
	p = p - C*step*snn_derivatives(p, x, y).';
	
	if mod(trial, 10) == 0
		cost_history(end+1) = sum(snn_cost(p, x_eval, y_eval));
	end
end

plot(cost_history);
