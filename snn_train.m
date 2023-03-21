function p = snn_train()

N = 1000000;

net = [1 2 2 2 1];
p_len = sum(net(1:(end-1)) .* net(2:end));

p = rand([p_len 1]);
x_train = rand([1 N])*2*pi;
x_eval = rand([1 500])*2*pi;
%y_train = [sin(x_train); cos(x_train); tan(x_train)];
%y_eval = [sin(x_eval); cos(x_eval); tan(x_eval)]

y_train = [5*x_train];% 10*x_train; 15*x_train];
y_eval = [5*x_eval];% 10*x_eval; 15*x_eval];

%y_train = [sin(x_train)];%; x_train.^2; x_train.^3];
%y_eval = [sin(x_eval)];%; x_eval.^2; x_eval.^3];

step = 0.001;

cost_history = [];
dt = 0.1;
t = 0:dt:2*pi;
figure();
coeff = 1;
for trial = 1:N
	x = x_train(1, trial);
	y = y_train(:, trial);
	c = snn_cost(p, net, x, y);
	%J_numerical = numerical_jacobian(p, @snn_cost, {net, x, y});
	J = snn_derivatives(p, net, x, y);
	%keyboard;
	%p_prev = p;
	%coeff = max(abs(J));
	p = p - step/max(1,max(abs(J)))*J.';

	if mod(trial, 100) == 0
		c = snn_cost(p, net, x_eval, y_eval);
		length(find(J));
		cost_history(end+1) = sqrt(mean(c));
        subplot(211);
		semilogy(cost_history);
        subplot(212);
        plot(t, [sin(t); snn(p, net, t)]);
		drawnow();
        
		%if length(cost_history) > 10
		%	if sum(diff(cost_history((end-10):end))>=0) > 5
		%		step = step/2
		%	elseif all(diff(cost_history(end-10:end)) < 0)
		%		step = min(1, step * 1.1);
		%	end
        %end
	end
	if any(isnan(p))
		keyboard;
	end
end

subplot(211);
semilogy(cost_history);
subplot(212);
plot(t, [sin(t); snn(p, net, t)]);