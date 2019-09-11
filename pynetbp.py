from random import normalvariate as norm
from math import e as exp

class net():

	def __init__(self, structure):
		self.activations = [[[None for _ in range(i)]] for i in structure]
		self.bias = [[[0 for _ in range(i)]] for i in structure[1:]]
		self.structure = structure
		self.error = 100
		self.weights = self.init_weights(self.activations)


	def init_weights(self, act):
		ret = []
		for n, layer in enumerate(act[:-1]):
			layer = layer[0]
			ret.append([])
			next_layer = act[n+1][0]
			for node_1 in range(len(layer)):
				ret[-1].append([])
				for node_2 in range(len(next_layer)):
					ret[-1][-1].append(norm(0,1))
		return ret


	def mat_mul(self, a, b):
		ret = []
		for row in a:
			ret.append([])
			for column in range(len(b[0])):
				ret[-1].append(0)
				for n, a_i in enumerate(row):
					ret[-1][-1] += a_i*b[n][column]
		return ret

	def mat_add(self, a, b):
		ret = []
		for row in range(len(a)):
			ret.append([])
			for item in range(len(a[row])):
				ret[-1].append(a[row][item] + b[row][item])
		return ret

	def sigmoid(self, x):
		return 1/(1+(exp**-x))

	def activate(self, activation_matrix):
		# Using sigmoid
		return [[self.sigmoid(i) for i in activation_matrix[0]]]

	def fire(self, x):
		self.activations[0] = x
		for index in range(len(self.activations[:-1])):
			layer = self.activations[index]
			#forward_propagation = self.activate(self.mat_add(self.mat_mul(layer, self.weights[index]), self.bias[index]))
			forward_propagation = self.activate(self.mat_mul(layer, self.weights[index]))
			self.activations[index+1] = forward_propagation
		return self.activations

	def __str__(self):
		return str(self.activations[-1])

	# Time for the big boy

	def sigmoid_prime(self, x):
		return self.sigmoid(x)*(1-self.sigmoid(x))

	def find_error(self, desired_output):
		return sum([0.5*(self.activations[-1][0][n] - desired_output[0][n])**2 for n in range(len(desired_output[0]))])

	# https://wikimedia.org/api/rest_v1/media/math/render/svg/991c8f020800ec1da130849e20a3a415613e9bdb

	def output_delta(self, activation, t):
		return (activation - t)*activation*(1 - activation)

	def train(self, input_data, desired_output, epochs=1000, learning_rate=0.5, print_error=0):
		for epoch in range(epochs):
			if print_error: print(self.error)
			for position in range(len(input_data)):
				self.fire(input_data[position])
				error = self.find_error(desired_output[position])
				self.error = error
				deltas = [[[None for _ in range(i)]] for i in self.structure]

				# Find the deltas of each neuron

				for layer in range(len(deltas)-1,-1,-1):
					if layer == len(deltas)-1:
						for node in range(len(deltas[layer][0])):
							deltas[layer][0][node] = self.output_delta(self.activations[layer][0][node], desired_output[position][0][node])
					else:
						fixed = [[i] for i in deltas[layer+1][0]]
						deltas[layer] = [[i[0] for i in self.mat_mul(self.weights[layer], fixed)]]
						for node in range(len(deltas[layer][0])):
							deltas[layer][0][node] *= self.activations[layer][0][node]*(1-self.activations[layer][0][node])

				# Now that the deltas are found, it's just a matter of multiplying activations to find the change required

				for layer in range(len(self.weights)-1,-1,-1):
					for oi in range(len(self.activations[layer][0])):
						for oj in range(len(self.activations[layer+1][0])):
							change = -learning_rate*deltas[layer+1][0][oj]*self.activations[layer][0][oi]
							self.weights[layer][oi][oj] += change

if __name__ == '__main__':
	nnet = net([3,5,3])
	input_data = [[[0,0,0]],[[0,0,1]],[[0,1,0]],[[0,1,1]],[[1,0,0]],[[1,0,1]],[[1,1,0]]]
	desired_output = [[[0,0,1]],[[0,1,0]],[[0,1,1]],[[1,0,0]],[[1,0,1]],[[1,1,0]],[[1,1,1]]]

	nnet.train(input_data, desired_output)
	for i in input_data:
		nnet.fire(i)
		print('input:', i, 'network output:', nnet)
