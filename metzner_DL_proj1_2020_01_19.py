import numpy as np

"""
Defining the following classes:
- Neuron
- FullyConnectedLayer
- NeuralNetwork
"""


class Neuron:

    """
    Initializing Neuron class with an activation function (activationFunction), input values (numInput), learning
    rate (learnRate), vector with weights based on the number of neurons of previous layer / input layer (weights),
    and one value for the bias of a neuron. Weights and bias are randomized between 0 and 1 from a uniform
    distribution if no optional argument is given.
    """

    def __init__(self, activationFunction, numInput, learnRate, weights=None, bias=None):
        self.activation_function = activationFunction
        self.number_input = numInput
        self.learn_rate = learnRate
        if weights is not None:
            self.weights = weights
        else:
            self.weights = np.random.uniform(0, 1, len(numInput))
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
        pass
        return



class NeuralNetwork:
    def __init__(self, num_layers, num_neurons_layer, vec_activationFunction, num_Input, lossFunction,
                 learnRate, actualOutput, weightsNetwork=None, biasNetwork=None):
        self.number_layers = num_layers
        self.number_neurons_layer = num_neurons_layer
        self.activation_function = vec_activationFunction
        self.number_input = num_Input
        self.loss_function = lossFunction
        self.learn_rate = learnRate
        self.actual_output = actualOutput
        self.weights = weightsNetwork
        self.bias = biasNetwork

        if self.weights is None:
            self.FullyConnectedLayers = []
            for i in range(self.number_layers):
                self.FullyConnectedLayers.append(FullyConnectedLayer(num_neurons=self.number_neurons_layer[i],
                                                                     activationFunction=self.activation_function[i],
                                                                     num_Input=self.number_input,
                                                                     learnRate=self.learn_rate,
                                                                     weights=None,
                                                                     bias=None))
        else:
            self.FullyConnectedLayers = []
            for i in range(self.number_layers):
                print(i)
                self.FullyConnectedLayers.append(FullyConnectedLayer(num_neurons=self.number_neurons_layer[i],
                                                                     activationFunction=self.activation_function[i],
                                                                     num_Input=self.number_input,
                                                                     learnRate=self.learn_rate,
                                                                     weights=self.weights[i],
                                                                     bias=self.bias[i]))


    def calculateloss(self, predicted_output):
        if self.loss_function == "MSE":
            return mse_loss(predicted_output, self.actual_output)
        elif self.loss_function == "BinCrossEntropy":
            return bin_cross_entropy_loss(predicted_output, self.actual_output)

    def train(self):
        pass




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


def mse_loss(predicted_output, actual_output):
    return np.sum(predicted_output - actual_output) ** 2 / len(actual_output)


def bin_cross_entropy_loss(predicted_output, actual_output):
    pass


vec_AF = ["logistic", "logistic"]
weights_TEST = [[(1,2),(2,3)],[(3,4),(4,5)]]
bias_Test = [1,2]
# Driver code main()
def main():
    NN = NeuralNetwork(num_layers=2, num_neurons_layer=[2, 2], vec_activationFunction=vec_AF, num_Input=2, lossFunction="MSE", learnRate=0.01, actualOutput=[0.01, 0.99], weightsNetwork=weights_TEST, biasNetwork=bias_Test)


if __name__ == '__main__':
    main()

    