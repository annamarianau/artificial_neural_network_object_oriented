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

    def net_output_neurons(self, input_vec):
        return self.bias + np.dot(self.weights, input_vec)


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
            self.neurons = []
            for i in range(num_neurons):
                self.neurons.append(
                    Neuron(activationFunction=activationFunction, numInput=self.number_input, learnRate=self.learn_rate,
                           weights=None, bias=self.bias))
        else:
            self.neurons = []
            for i in range(num_neurons):
                self.neurons.append(
                    Neuron(activationFunction=activationFunction, numInput=self.number_input, learnRate=self.learn_rate,
                           weights=self.weights[i], bias=self.bias))


    def calculate(self, input_vec):
        """
        Computes the output of all neurons in one layer
        :return:
        """
        output_vec = []
        net_output_vec = []
        for neuron in self.neurons:
            output = neuron.calculate(input_vec)
            net_output = neuron.net_output_neurons(input_vec)
            output_vec.append(output)
            net_output_vec.append(net_output)
        return output_vec, net_output_vec


class NeuralNetwork:
    def __init__(self, num_layers, num_neurons_layer, vec_activationFunction, num_Input, lossFunction,
                 learnRate, weightsNetwork=None, biasNetwork=None):
        self.number_layers = num_layers
        self.number_neurons_layer = num_neurons_layer
        self.activation_function = vec_activationFunction
        self.number_input = num_Input
        self.loss_function = lossFunction
        self.learn_rate = learnRate
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
                self.FullyConnectedLayers.append(FullyConnectedLayer(num_neurons=self.number_neurons_layer[i],
                                                                     activationFunction=self.activation_function[i],
                                                                     num_Input=self.number_input,
                                                                     learnRate=self.learn_rate,
                                                                     weights=self.weights[i],
                                                                     bias=self.bias[i]))


    def calculateloss(self, predicted_output, actual_output):
        if self.loss_function == "MSE":
            return mse_loss(predicted_output, actual_output)
        elif self.loss_function == "BinCrossEntropy":
            return bin_cross_entropy_loss(predicted_output, actual_output)

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
    return np.square(np.subtract(predicted_output, actual_output))*1/2


def bin_cross_entropy_loss(predicted_output, actual_output):
    pass


vec_AF = ["logistic", "logistic"]
weights_TEST = [[(0.15, 0.2), (0.25, 0.3)], [(0.4, 0.45), (0.50, 0.55)]]
bias_Test = [0.35, 0.60]
input_vec = [0.05, 0.10]
actual_output = [0.01, 0.99]


# Driver code main()
def main():
    NN = NeuralNetwork(num_layers=2, num_neurons_layer=[2, 2], vec_activationFunction=vec_AF, num_Input=2, lossFunction="MSE", learnRate=0.01, weightsNetwork=weights_TEST, biasNetwork=bias_Test)
    """
    Feedforward algorithm
    for-loop to compute individual (ind_loss) and total loss (total_loss) of each layers (hidden and output) of the 
    network
    original input variable (input_vec; "input neurons") of network is updated to values of next layer neurons output 
    values    
    """
    global input_vec
    predicted_output_list = []
    net_output_list = []
    for i, layer in enumerate(NN.FullyConnectedLayers):
        predicted_output = layer.calculate(input_vec)
        predicted_output_list.append(predicted_output[0])
        net_output_list.append(predicted_output[1])
        input_vec = predicted_output_list[i]
    ind_loss = NN.calculateloss(predicted_output[0], actual_output)
    print("List with individual Losses for output neurons - Output_1 and Output_2: ", ind_loss)
    total_loss = np.sum(ind_loss)
    print("Total Loss accrued in Network: ", total_loss)
    print("Net output neurons per layer: ", net_output_list)
    print("List with output of individual neurons for all layers: ", predicted_output_list)



if __name__ == '__main__':
    main()