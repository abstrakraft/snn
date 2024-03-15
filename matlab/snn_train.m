function p = snn_train()

N = 1000000;

net = [1 5 10 10 5 1];
p_len = sum((1+net(1:(end-1))) .* net(2:end));

p = rand([p_len 1]);
x_train = rand([1 N])*2*pi;
x_eval = rand([1 500])*2*pi;
x_plot = 0:0.01:(2*pi);
%y_train = [sin(x_train); cos(x_train); tan(x_train)];
%y_eval = [sin(x_eval); cos(x_eval); tan(x_eval)]

%y_train = [5*x_train + 0.3];% 10*x_train; 15*x_train];
%y_eval = [5*x_eval + 0.3];% 10*x_eval; 15*x_eval];
%y_plot = [5*x_plot + 0.3];

y_train = [sin(2*x_train)+1.5];%; x_train.^2; x_train.^3];
y_eval = [sin(2*x_eval)+1.5];%; x_eval.^2; x_eval.^3];
y_plot = [sin(2*x_plot)+1.5];

step = 0.03;

cost_history = [];
figure();
coeff = 1;
for trial = 1:N
	x = x_train(:, trial);
	y = y_train(:, trial);
	c = snn_cost(p, net, x, y);
	%J_numerical = numerical_jacobian(p, @snn_cost, {net, x, y});
	J = snn_derivatives(p, net, x, y);
	%keyboard;
	%p_prev = p;
	%coeff = max(abs(J));
	%p = p - min(step,1/(norm(J)/p_len))*J.';
    p = p - step/max(1, max(abs(J)))*J.';
    %p = p - step*J.';

	if mod(trial, 100) == 0
		c = snn_cost(p, net, x_eval, y_eval);
		length(find(J));
		cost_history(end+1) = sqrt(mean(c));
        subplot(211);
		semilogy(cost_history);
        subplot(212);
        plot(x_plot, [y_plot; snn(p, net, x_plot)]);
        ylim([min(y_plot) max(y_plot)]);
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
