import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

def main():
	my_net = TfSimpleNeuralNet([( 1, None, None),
		                          ( 5, softplus, softplus_derivative),
							  						  (10, softplus, softplus_derivative),
							  						  (10, softplus, softplus_derivative),
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

	N = 100000
	rng = np.random.default_rng()
	x_train = rng.random((1,N))*2*np.pi
	x_eval = rng.random((1,500))*2*np.pi
	x_plot = np.atleast_2d(np.linspace(0,2*np.pi,1000))

	learn_function = lambda x : np.sin(2*x) + 1.5

	y_train = learn_function(x_train)
	y_eval = learn_function(x_eval)
	y_plot = learn_function(x_plot)

	my_net.train(x_train.T, y_train.T, x_eval.T, y_eval.T, x_plot.T, y_plot.T)

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

class TfSimpleNeuralNet(object):
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

		self.model = tf.keras.models.Sequential()
		self.model.add(tf.keras.Input(shape=[structure[0][0]]))
		for layer in structure[1:]:
			# TODO: replace 'softplus' with activation function from structure
			self.model.add(tf.keras.layers.Dense(layer[0], activation='softplus'))

		#self.model.compile(optimizer='adam', loss=cost, metrics=['accuracy'])
		self.model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError, metrics=['accuracy'])


	#def eval(self, x):
	#	x = np.asarray(x)
	#	for layer_idx in range(len(self.structure)-1):
	#		x = self.structure[layer_idx+1][1](self.W[layer_idx] @ np.vstack((x, np.ones((1, x.shape[1])))))
	#	return x

	def train(self, x_train, y_train, x_eval, y_eval, x_plot, y_plot):
		N = x_train.shape[0]
		step_size = 1000
		fig, axs = plt.subplots(2, 1)#, layout='constrained')
		cost_history = []
		for trials in range(int(N/step_size)):
			history = self.model.fit(x_train, y_train, initial_epoch=trials, epochs=trials+1, steps_per_epoch=step_size, validation_data=(x_eval, y_eval))
			cost_history.append(history.history['val_loss'])
			y_plot_predict = self.model.predict(x_plot)
			axs[0].clear()
			axs[0].semilogy(cost_history);
			axs[0].grid()

			axs[1].clear()
			axs[1].plot(x_plot, y_plot, x_plot, y_plot_predict);
			axs[1].set_ylim(y_plot.min(), y_plot.max());
			axs[1].grid()
			#plt.show(block=False);
			plt.draw()
			plt.pause(0.001)

if __name__ == '__main__':
	main()
