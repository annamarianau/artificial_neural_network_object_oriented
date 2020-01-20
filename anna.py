import numpy as np

"""
Defining the following classes:
- Neuron
- FullyConnectedLayer
- NeuralNetwork
"""


class Neuron:
    # Initializing Neuron class with an activation function (activationFunction), input values (inputVals), learning
    # rate (learnRate), vector with weights based on the number of neurons of previous layer / input layer (weights),
    # and one value for the bias of a neuron. Weights and bias are randomized between 0 and 1 from a uniform
    # distribution if no optional argument is given.

    def __init__(self, activationFunction, numInput, learnRate, weights=None, bias=None):
        self.activation_function = activationFunction
        self.number_input = numInput
        self.learn_rate = learnRate
        self.bias = bias
        if weights is not None:
            self.weights = weights
        else:
            self.weights = np.random.uniform(0, 1, len(numInput))

    # Class method activate - phi(bias + sum of weights * input)
    def activate(self, z):
        if self.activation_function == "logistic":
            return log_act(z)
        elif self.activation_function == "linear":
            return lin_act(z)

    def calculate(self, input_vec):
        return self.activate(self.bias + np.dot(self.weights, input_vec))


class FullyConnectedLayer:
    def __init__(self, num_neurons, activationFunction, num_Input, learnRate, weights=None, bias=None):
        self.number_neurons = num_neurons
        self.activation_function = activationFunction
        self.number_input = num_Input
        self.learn_rate = learnRate
        self.weights = weights
        if bias is not None:
            self.bias = bias
        else:
            self.bias = np.random.uniform(0, 1)
        # create neurons (stored in a list) of layer
        if weights is None:
            self.neurons = [Neuron(activationFunction=activationFunction, numInput=self.number_input, learnRate=self.learn_rate, weights=None, bias=self.bias) for i in range(num_neurons)]
        else:
            self.neurons = [Neuron(activationFunction=activationFunction, numInput=self.number_input, learnRate=self.learn_rate, weights=self.weights[i], bias=self.bias) for i in range(num_neurons)]


    def calculate(self, input_vec):
        """
        Computes the output of all neurons in one layer
        :return:
        """
        output_vec = []
        for neuron in self.neurons:
            output = neuron.calculate(input_vec)
            output_vec.append(output)
        return output_vec


def log_act(z):
    return 1 / (1 + np.exp(-z))


def lin_act(z):
    return z



test_input = [0, 0]
test_weights = [(1, 1),(2,2)]
test_bias = [1]

# Driver code main()
def main():
    one_layer = FullyConnectedLayer(num_neurons=2, activationFunction="linear", num_Input=2, learnRate=0.1, weights=test_weights, bias=test_bias)
    print(one_layer.calculate(test_input))

if __name__ == '__main__':
    main()