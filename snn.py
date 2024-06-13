import matplotlib.pyplot as plt
import numpy as np

def main():
	my_net = SimpleNeuralNet([( 1, None, None),
		                        ( 5, softplus, softplus_derivative),
													  (10, softplus, softplus_derivative),
													  (10, softplus, softplus_derivative),
		                        ( 5, softplus, softplus_derivative),
													  ( 1, softplus, softplus_derivative)],
													 half_squared_norm, half_squared_norm_derivative)

	#print(my_net.W)
	#print(my_net.eval([[1]]))
	#dW = my_net.cost_derivative([[1]],[[1]])
	#ndW = my_net.cost_derivative_numerical([[1]],[[1]])
	#for layer_idx in range(len(dW)):
	#	print(dW[layer_idx] - ndW[layer_idx])

	N = 1000000
	rng = np.random.default_rng()
	x_train = rng.random((1,N))*2*np.pi
	x_eval = rng.random((1,500))*2*np.pi
	x_plot = np.atleast_2d(np.linspace(0,2*np.pi,1000))

	learn_function = lambda x : np.sin(2*x) + 1.5

	y_train = learn_function(x_train)
	y_eval = learn_function(x_eval)
	y_plot = learn_function(x_plot)

	my_net.train(x_train, y_train, x_eval, y_eval, x_plot, y_plot)

	plt.show()

def identity(x):
	return x

def identity_derivative(x):
	return np.ones(x.shape)

def softplus(x):
	return np.log1p(np.exp(x))

def softplus_derivative(x):
	return 1/(1+np.exp(-x))

def half_squared_norm(x, y):
	# accepts 2D arrays, returns 1D array
	return ((x-y)**2).sum(0)/2

def half_squared_norm_derivative(x, y):
	return (x-y).T

class SimpleNeuralNet(object):
	def __init__(self, structure, cost, dcost):
		'''config should be a list of 3-tuples.  Each tuple represents the
		   configuration of a layer of the net, with the first element containing
		   the count of nodes in the layer, the second, the activation function, and
		   the third, the derivative of the activation function.  The first 3-tuple
		   indicates the number of inputs to the net, and the activation function and
		   derivative are ignored.
		'''
		self.structure = structure
		# W is a list of weight matrices
		self.W = []
		rng = np.random.default_rng()
		for layer_idx in range(len(self.structure)-1):
			# +1 is for the offset term in each node
			self.W.append(rng.random((self.structure[layer_idx+1][0], self.structure[layer_idx][0]+1)))
		self.cost = cost
		self.dcost = dcost

	def eval(self, x):
		x = np.asarray(x)
		for layer_idx in range(len(self.structure)-1):
			x = self.structure[layer_idx+1][1](self.W[layer_idx] @ np.vstack((x, np.ones((1, x.shape[1])))))
		return x

	def cost_derivative(self, x, y):
		'''Returns a list of arrays containing the derivative, with respect to the
		   cost function, of each net parameter.
			 x -- column vector net input
			 y -- column vector net desired output
		'''
		layer_count = len(self.W)

		# z is the input to each node of each layer, just before the activation function is applied
		# indices correspond to layer_idx
		z = []
		# a is the input to each layer, before weights are applied
		a = [x]
		for layer_idx in range(layer_count):
			z.append(self.W[layer_idx] @ np.vstack((a[-1], 1)))
			a.append(self.structure[layer_idx+1][1](z[-1]))

		dcost = self.dcost(a[-1], y)
		prev_partial = dcost.T * self.structure[-1][2](z[-1]).T
		J = [None] * layer_count
		for layer_idx in reversed(range(layer_count)):
			J[layer_idx] = prev_partial.T @ np.vstack((a[layer_idx], 1)).T
			if layer_idx == 0: continue
			prev_partial = prev_partial @ self.W[layer_idx][:,:-1] * self.structure[layer_idx][2](z[layer_idx-1]).T

		return J

	def cost_derivative_numerical(self, x, y):
		delta = 1e-8
		base_cost = self.cost(self.eval(x), y)
		J = []
		for layer_idx in range(len(self.structure) - 1):
			J.append(np.zeros(self.W[layer_idx].shape))
			d = J[-1]
			for rdx in range(d.shape[0]):
				for cdx in range(d.shape[1]):
					tmp = self.W[layer_idx][rdx,cdx]
					self.W[layer_idx][rdx,cdx] += delta
					J[layer_idx][rdx, cdx] = (self.cost(self.eval(x), y) - base_cost)/delta
					self.W[layer_idx][rdx,cdx] = tmp

		return J

	def train(self, x_train, y_train, x_eval, y_eval, x_plot, y_plot):
		step = 0.03
		trial_count = x_train.shape[1]
		layer_count = len(self.W)

		cost_history = []
		fig, axs = plt.subplots(2, 1)#, layout='constrained')
		for trial_idx in range(trial_count):
			x = x_train[:,trial_idx]
			y = y_train[:,trial_idx]
			J = self.cost_derivative(x, y)

			# compute scaling factor
			max_abs_J = 0
			for layer_idx in range(layer_count):
				max_abs_J = max(max_abs_J, np.abs(J[layer_idx]).max())
			scaling = step/max(1, max_abs_J)

			for layer_idx in range(layer_count):
				self.W[layer_idx] -= scaling * J[layer_idx]

			if trial_idx % 1000 == 0:
				c = self.cost(self.eval(x_eval), y_eval)
				cost_history.append(np.sqrt(np.mean(c)))
				#import pdb; pdb.set_trace()

				axs[0].clear()
				axs[0].semilogy(cost_history);
				axs[0].grid()

				axs[1].clear()
				axs[1].plot(x_plot[0,:], y_plot[0,:], x_plot[0,:], self.eval(x_plot)[0,:]);
				axs[1].set_ylim(y_plot.min(), y_plot.max());
				axs[1].grid()
				#plt.show(block=False);
				plt.draw()
				plt.pause(0.001)

if __name__ == '__main__':
	main()
