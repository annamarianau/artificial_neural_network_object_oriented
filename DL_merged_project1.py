import numpy as np

"""
Defining the following classes:
- Neuron
- FullyConnectedLayer
- NeuralNetwork
"""


class Neuron:
    """
    Initializing the following attributes for the Neuron class:
    - Activation Function: activationFunction
    - Number Inputs: numInput
    - Learning Rate: learnRate
    - Tuple with weights: weights
    - Bias: bias
    """

    # Initializing Neuron class with an activation function (activationFunction), input values (inputVals), learning
    # rate (learnRate), vector with weights based on the number of neurons of previous layer / input layer (weights),
    # and one value for the bias of a neuron. Weights and bias are randomized between 0 and 1 from a uniform
    # distribution if no optional argument is given.

    def __init__(self, activation_function, num_input, learn_rate, weights=None, bias=None):
        self.activation_function = activation_function
        self.number_input = num_input
        self.learn_rate = learn_rate
        self.bias = bias
        self.output = None
        self.delta = None
        self.updated_weights = []
        """
        If no tuple with weights is given then generate number of weights based on number of inputs
        If tuple with weights is given then self.weights = weights
        """
        if weights is None:
            self.weights = np.random.uniform(0, 1, self.number_input)
        else:
            self.weights = weights

    """
    Class Method
    Selection of activation function with input variable z
    z = bias + weights*input (self.bias + self.weights * input_vec)
    """

    def activate(self, z):
        if self.activation_function == "logistic":
            return log_act(z)
        elif self.activation_function == "linear":
            return lin_act(z)

    def calculate(self, input_previous_layer):
        return self.activate(self.bias + np.dot(self.weights, input_previous_layer))

    def calculate_delta_output(self, actual_output_network):
        if self.activation_function == "logistic":
            return -(actual_output_network - self.output) * log_act_prime(self.output)
        elif self.activation_function == "linear":
            return -(actual_output_network - self.output) * lin_act_prime(self.output)

    def calculate_delta_hidden(self, delta_sum):
        if self.activation_function == "logistic":
            return delta_sum * log_act_prime(self.output)
        elif self.activation_function == "linear":
            return delta_sum * lin_act_prime(self.output)


class FullyConnectedLayer:
    def __init__(self, num_neurons, activation_function, num_input, learn_rate, weights=None, bias=None):
        self.number_neurons = num_neurons
        self.activation_function = activation_function
        self.number_input = num_input
        self.learn_rate = learn_rate
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
                    Neuron(activation_function=activation_function, num_input=self.number_input,
                           learn_rate=self.learn_rate,
                           weights=None, bias=self.bias))
        else:
            self.neurons = []
            for i in range(num_neurons):
                self.neurons.append(
                    Neuron(activation_function=activation_function, num_input=self.number_input,
                           learn_rate=self.learn_rate,
                           weights=self.weights[i], bias=self.bias))

    def calculate(self, input_previous_layer):
        """
        Computes the output of all neurons in one layer
        :return:
        """
        output_vec = []
        for neuron in self.neurons:
            neuron.output = neuron.calculate(input_previous_layer)
            output_vec.append(neuron.output)
            print(neuron.output)
        return output_vec


class NeuralNetwork:
    def __init__(self, num_layers, num_neurons_layer, vec_activation_function, num_input, loss_function,
                 learn_rate, weights_network=None, bias_network=None):
        self.number_layers = num_layers
        self.number_neurons_layer = num_neurons_layer
        self.activation_function = vec_activation_function
        self.number_input = num_input
        self.loss_function = loss_function
        self.learn_rate = learn_rate
        self.weights = weights_network
        self.bias = bias_network

        if self.weights is None:
            self.FullyConnectedLayers = []
            for i in range(self.number_layers):
                self.FullyConnectedLayers.append(FullyConnectedLayer(num_neurons=self.number_neurons_layer[i],
                                                                     activation_function=self.activation_function[i],
                                                                     num_input=self.number_input,
                                                                     learn_rate=self.learn_rate,
                                                                     weights=None,
                                                                     bias=None))
        else:
            self.FullyConnectedLayers = []
            for i in range(self.number_layers):
                self.FullyConnectedLayers.append(FullyConnectedLayer(num_neurons=self.number_neurons_layer[i],
                                                                     activation_function=self.activation_function[i],
                                                                     num_input=self.number_input,
                                                                     learn_rate=self.learn_rate,
                                                                     weights=self.weights[i],
                                                                     bias=self.bias[i]))

    def calculateloss(self, predicted_output, actual_output):
        if self.loss_function == "MSE":
            return mse_loss(predicted_output, actual_output)
        elif self.loss_function == "BinCrossEntropy":
            return bin_cross_entropy_loss(predicted_output, actual_output)


"""
Activation Functions with their respective prime functions
- logistic (log_act)
- linear (lin_act)
- z: result of the weighted sum of weights (w), biases (b), and inputs (x) - z = np.dot(w,x)-b
"""


def log_act(z):
    return 1 / (1 + np.exp(-z))


def log_act_prime(output):
    return output * (1 - output)


def lin_act(z):
    return z


def lin_act_prime(z):
    return 1


"""
Loss Functions
- Mean squared error (mse_loss); https://en.wikipedia.org/wiki/Mean_squared_error
- Binary cross entropy loss; https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html (bin_cross_entropy_loss)
- predicted: Array containing all predicted/computed output for each sample by the neural network.
- actual: Array containing the ground truth value for each sample, respectively.
"""


def mse_loss(predicted_output, actual_output):
    return np.square(np.subtract(predicted_output, actual_output)) * 1 / 2


def bin_cross_entropy_loss(predicted_output, actual_output):
    pass


vec_AF = ["logistic", "logistic", "logistic"]
weights_TEST = [[(0.15, 0.2), (0.25, 0.3)], [(0.4, 0.45), (0.50, 0.55)]]
bias_Test = [0.35, 0.60]
input_vec = [0.05, 0.10]
actual_output_network = [0.01, 0.99]


# Driver code main()
def main():
    NN = NeuralNetwork(num_layers=2, num_neurons_layer=[2, 2], vec_activation_function=vec_AF, num_input=2,
                       loss_function="MSE", learn_rate=0.5, weights_network=weights_TEST, bias_network=bias_Test)
    """
    Feedforward algorithm
    for-loop to compute individual (ind_loss) and total loss (total_loss) of each layers (hidden and output) of the 
    network
    original input variable (input_vec; "input neurons") of network is updated to values of next layer neurons output 
    values    
    """
    global input_vec
    vec_input_neurons = input_vec
    for i, layer in enumerate(NN.FullyConnectedLayers):
        predicted_output = layer.calculate(input_vec)
        input_vec = predicted_output
    ind_loss = NN.calculateloss(input_vec, actual_output_network)
    print("List with individual Losses for output neurons - Output_1 and Output_2: ", ind_loss)
    total_loss = np.sum(ind_loss)
    print("Total Loss accrued in Network: ", total_loss)


    """
    Backpropagation Algorithm
    """
    # Reversing list with layer objects for back propagation
    # back prop starts with updating weights connected to output layer
    NN.FullyConnectedLayers.reverse()
    for i, layer in enumerate(NN.FullyConnectedLayers):
        if i == 0:
            print("This is the output layer: ", layer)
            for j, neuron in enumerate(layer.neurons):
                neuron.delta = neuron.calculate_delta_output(actual_output_network[j])
                for k, neuron_hidden in enumerate(NN.FullyConnectedLayers[i+1].neurons):
                    error_weight = neuron.delta * neuron_hidden.output
                    updated_weight = neuron.weights[k] - NN.learn_rate * error_weight
                    neuron.updated_weights.append(updated_weight)
        else:
            print("This is a hidden layer: ", layer)
            for j, neuron in enumerate(layer.neurons):
                sum = 0
                for m in range(len(layer.neurons)):
                    sum += NN.FullyConnectedLayers[i-1].neurons[m].delta *\
                           NN.FullyConnectedLayers[i-1].neurons[m].weights[j]
                for k in range(len(neuron.weights)):
                    neuron.delta = neuron.calculate_delta_hidden(sum)
                    error_weight = neuron.delta * vec_input_neurons[k]
                    updated_weight = neuron.weights[k] - NN.learn_rate * error_weight
                    neuron.updated_weights.append(updated_weight)
                print("updated weights hidden: ", neuron.updated_weights)


if __name__ == '__main__':
    main()
