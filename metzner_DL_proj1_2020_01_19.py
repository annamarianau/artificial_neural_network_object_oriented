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
        if weights is not None:
            self.weights = weights
        else:
            self.weights = np.random.uniform(0, 1, len(inputVals))
        if bias is not None:
            self.bias = bias
        else:
            self.bias = np.random.uniform(0, 1)

    # Class method activate - phi(bias + sum of weights * input)
    def activate(self, z):
        if self.activation_function == "logistic":
            return log_act(z)
        elif self.activation_function == "linear":
            return lin_act(z)

    def calculate(self):
        return self.activate(self.bias + np.dot(self.weights, self.number_input))


class FullyConnectedLayer:
    def __init__(self, num_neurons, activationFunction, num_Input, learnRate, weights=None, bias=None):
        self.number_neurons = num_neurons
        self.activation_function = activationFunction
        self.number_input = num_Input
        self.learn_rate = learnRate
        self.weights = weights
        self.bias = bias

    def calculate(self):
        """
        Computes the output of all neurons in one layer
        :return:
        """
        return pass






class NeuralNetwork:
    def __init__(self, num_layers, num_neurons_layer):
        self.number_layers = num_layers
        self.number_neurons_layer = num_neurons_layer




















"""



class FullyConnectedLayer:
    def __init__(self, activation_function, number_neurons, input_vals, learn_rate, weights = True, bias = True):
        self.activation_function = activation_function
        self.number_neurons = number_neurons
        self.input_vals = input_vals
        self.learn_rate = learn_rate
        self.weights = weights if weights is not True else np.random.uniform(0, 1, len(input_vals))
        self.bias = bias if bias is not True else np.random.uniform(0, 1)


    def Generate_Neurons(self):
        return [Neuron(self.activation_function, self.input_vals, self.learn_rate, self.weights[i], self.bias[i]) for i in range(self.number_neurons)]

    def Print_Neurons(self):
        print(self.Generate_Neurons())

    def calculate(self):
        pass

"""
"""

class NeuralNetwork:
    def __init__(self, number_layers, vec_number_neurons_layer, vec_activation_function, loss_function, learn_rate,
                 input_network, actual_output, vec_weights, vec_biases):
        self.number_layers = number_layers
        self.vec_number_neurons_layer = vec_number_neurons_layer
        self.vec_activation_function = vec_activation_function
        self.loss_function = loss_function
        self.actual_output = actual_output

    def calculate(self):
        pass

    def calculateloss(self, output):
        if self.loss_function == "MSE":
            return mse_loss(predicted_output, self.actual_output)
        elif self.loss_function == "Bin_Cross_Entropy":
            return bin_cross_entropy_loss(predicted_output, self.actual_output)

    def feedforward(self):
        pass

    def backprop(self):
        pass
"""

"""
Activation Functions
- logistic (log_act)
- linear (lin_act)
- z: result of the weighted sum of weights (w), biases (b), and inputs (x) - z = np.dot(w,x)-b
"""


def log_act(z):
    return 1 / (1 + np.exp(-z))


def lin_act(z):
    return z


"""
Loss Functions
- Mean squared error (mse_loss); https://en.wikipedia.org/wiki/Mean_squared_error
- Binary cross entropy loss; https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html (bin_cross_entropy_loss)
- predicted: Array containing all predicted/computed output for each sample by the neural network.
- actual: Array containing the ground truth value for each sample, respectively.
"""

"""
def mse_loss(predicted, actual):
    return np.sum(predicted - actual) ** 2 / len(actual)


def bin_cross_entropy_loss(predicted, actual):
    return predicted, actual
"""

test_input = [0.05, 0.10]
test_weights = [(0.15, 0.20)]  # , (0.25, 0.30)]
test_bias = [0.35]  # , 0.60]

<<<<<<< Updated upstream
=======
test_weights = [0.15, 0.20]
test_input = [0.05, 0.10]
test_bias = 0.35
a = 1
>>>>>>> Stashed changes

# Driver code main()
def main():
    neural_net = NeuralNetwork(2, (2,2))
    neural_net.print_init()
    neural_net.gen_Layers()


    first_neuron = Neuron("logistic", test_input, 0.1, test_weights, test_bias)
    output_first_layer = first_neuron.calculate()
    print(output_first_layer)






    # one_layer = FullyConnectedLayer("logistic", 2, test_input, 0.1, test_weights, test_bias)
    # one_layer.Print_Neurons()


if __name__ == '__main__':
    main()
