import numpy as np
import sys


class Neuron:
    def __init__(self, activation_function, number_input, learning_rate, weights=None, bias=None):
        self.activation_function = activation_function
        self.number_input = number_input
        self.learning_rate = learning_rate
        self.bias = bias

        # stores calculated output
        self.output = None

        if weights is None:
            self.weights = np.random.uniform(0, 1, self.number_input)
        else:
            self.weights = weights

    # Method for activation of neuron using variable z as input
    # z = bias + sum(weights*inputs)
    # If-statement to select correct activation function based on given string-input ("logistic" or "linear")
    def activate(self, z):
        if self.activation_function == "logistic":
            return log_act(z)
        elif self.activation_function == "linear":
            return lin_act(z)

    # Method for calculating output of neuron based on weighted sum
    def calculate(self, input_vector):
        return self.activate(self.bias + np.dot(self.weights, input_vector))

    def print(self):
        print(self.weights)
        print(self.bias)
        print(self.number_input)


class FullyConnectedLayer:
    def __init__(self, number_neurons, activation_function, number_input, learning_rate, weights=None, bias=None):
        self.number_neurons = number_neurons
        self.activation_function = activation_function
        self.number_input = number_input
        self.learning_rate = learning_rate

        # If bias not given by user create one random bias from a uniform distribution for whole layer
        # this initial bias value is passed on each neuron in respective layer
        if bias is None:
            self.bias = np.random.uniform(0, 1)
        else:
            self.bias = bias

        self.weights = weights
        self.neurons = []
        if weights is None:
            for i in range(self.number_neurons):
                self.neurons.append(Neuron(activation_function=activation_function, number_input=self.number_input,
                                           learning_rate=self.learning_rate, weights=None, bias=self.bias))
        else:
            for i in range(self.number_neurons):
                self.neurons.append(Neuron(activation_function=activation_function, number_input=self.number_input,
                                           learning_rate=self.learning_rate, weights=self.weights[i], bias=self.bias))

    def print(self):
        print(self.neurons)
        for neuron in self.neurons:
            neuron.print()

    # Method calculates the output of each neuron based on the sum of the weights * input + bias of this neuron
    # storing computed output of each neuron in the neuron --> later used for back propagation
    # returns array with final output --> necessary to compute the total accrued loss
    def calculate(self, input_vector):
        output_curr_layer = []
        for neuron in self.neurons:
            neuron.output = neuron.calculate(input_vector)
            print("Output Neuron: ",neuron.output)
            output_curr_layer.append(neuron.output)
        return output_curr_layer




class NeuralNetwork:
    def __init__(self, number_layers, number_neurons_layer, loss_function, activation_functions_layer, number_input_nn,
                 learning_rate, weights=None, bias=None):
        self.number_layers = number_layers  # Scalar-value (e.g., 2 - 1 hidden and 1 output layer)
        self.number_neurons_layer = number_neurons_layer  # Array (e.g., [2,2] - 2 Neurons HL and 2 Neurons in OL)
        self.loss_function = loss_function  # String-variable (e.g., "MSE" or "BCE")
        self.activation_functions_layer = activation_functions_layer  # Array with string-variables (e.g.,
        # ['logistic', 'logistic'] for two layer architecture])
        self.number_input_nn = number_input_nn  # Scalar-value (e.g., 2 - 2 "input neurons")
        self.learning_rate = learning_rate  # Scalar-value (e.g., 0.5 - passed down to each neuron)
        self.bias = bias  # Bias is generated in or passed to FullyConnectedLayer
        self.weights = weights  # Weights are passed to FullyConnectedLayer / Neuron object

        self.FullyConnectedLayers = []
        for i in range(self.number_layers):
            if weights is None:
                # IF-statement necessary to determine the number of inputs into neurons of certain layer
                # i == 0 --> first hidden layer --> number of input in neurons determined by inputs into network
                # i != 0 --> all successive layers --> number of input in neurons determined by
                # number of neurons in previous layer
                if i == 0:
                    self.FullyConnectedLayers.append(FullyConnectedLayer(number_neurons=self.number_neurons_layer[i],
                                                                         activation_function=
                                                                         self.activation_functions_layer[i],
                                                                         number_input=self.number_input_nn,
                                                                         learning_rate=self.learning_rate,
                                                                         weights=None, bias=None))
                else:
                    self.FullyConnectedLayers.append(FullyConnectedLayer(number_neurons=self.number_neurons_layer[i],
                                                                         activation_function=
                                                                         self.activation_functions_layer[i],
                                                                         number_input=(
                                                                             self.number_neurons_layer[i - 1]),
                                                                         learning_rate=self.learning_rate,
                                                                         weights=None, bias=None))
            else:
                if i == 0:
                    self.FullyConnectedLayers.append(FullyConnectedLayer(number_neurons=self.number_neurons_layer[i],
                                                                         activation_function=
                                                                         self.activation_functions_layer[i],
                                                                         number_input=self.number_input_nn,
                                                                         learning_rate=self.learning_rate,
                                                                         weights=self.weights[i], bias=self.bias[i]))
                else:
                    self.FullyConnectedLayers.append(FullyConnectedLayer(number_neurons=self.number_neurons_layer[i],
                                                                         activation_function=
                                                                         self.activation_functions_layer[i],
                                                                         number_input=(
                                                                             self.number_neurons_layer[i - 1]),
                                                                         learning_rate=self.learning_rate,
                                                                         weights=self.weights[i], bias=self.bias[i]))
    # Method to compute the losses at each output neuron
    # mse_loss: Mean Squared Error
    # bin_cross_entropy_loss: Binary Cross Entropy
    # predicted_output: Output after activation for each output neuron
    # actual_output: Actual output of network
    def calculateloss(self, predicted_output, actual_output):
        if self.loss_function == "MSE":
            return mse_loss(predicted_output, actual_output)
        elif self.loss_function == "BinCrossEntropy":
            return bin_cross_entropy_loss(predicted_output, actual_output)



    def print(self):
        print(self.FullyConnectedLayers)
        for layer in self.FullyConnectedLayers:
            layer.print()

    # Method for Feed-Forward algorithm
    # Computing output for each layer by calling Method (.calculate(current_input)) from FullyConnectedLayer object
    # Setting generated output of current layer equal with input_vector
    # --> Next layer / iteration uses output values as input values
    # returns computed output from neurons at final - output layer --> used to compute the loss
    def feed_forward(self, input_vector):
        global output_curr_layer
        for i, layer in enumerate(self.FullyConnectedLayers):
            print("Layer: ", i)
            output_curr_layer = layer.calculate(input_vector)
            input_vector = output_curr_layer
        return(output_curr_layer)

    # Method for Back-Propagation algorithm: Used for updating weights and biases of all neurons in all layers
    def back_propagation(self):
        # Reverse list containing layer objects since back propagation
        # starts updating with weights connected to output layer
        self.FullyConnectedLayers.reverse()
        for j, layer in enumerate(self.FullyConnectedLayers):
            # IF-statement to determine at which layer the weights are updated
            # Weights at output layer need slightly different calculations / algorithm
            # j = 0 --> output layer
            # j > 0 --> any hidden layer
            if j == 0:
                pass


    def train(self, input_vector, actual_output_network):
        predicted_output_network = self.feed_forward(input_vector)
        total_loss = np.sum(self.calculateloss(predicted_output_network, actual_output_network))
        print('Total Loss Network:', total_loss)

        self.back_propagation()



def perceptron_act(z):
    if z < 0:
        return 0
    elif z >= 0:
        return 1


def log_act(z):
    return 1 / (1 + np.exp(-z))


def log_act_prime(output):
    return output * (1 - output)


def lin_act(z):
    return z


def lin_act_prime(z):
    z = 1
    return z


def mse_loss(predicted_output, actual_output):
    return np.square(np.subtract(predicted_output, actual_output)) * 1 / 2


def bin_cross_entropy_loss(num_samples, predicted_output, actual_output):
    return 1 / num_samples * np.sum(-(actual_output * np.log(predicted_output)
                                      + (1 - actual_output) * np.log(1 - predicted_output)))


# Driver code main()
def main(argv=None):
    # single step of back-propagation using example from class
    example_input = [0.05, 0.10]
    example_output = [0.01, 0.99]
    example_weights = [[(0.15, 0.20), (0.25, 0.30)], [(0.40, 0.45), (0.50, 0.55)]]
    example_biases = [0.35, 0.60]

    NN = NeuralNetwork(number_layers=2, number_neurons_layer=[2, 2], loss_function='MSE',
                       activation_functions_layer=['logistic', 'logistic'], number_input_nn=2, learning_rate=0.5,
                       weights=example_weights, bias=example_biases)
    #NN.print() # mock NN.method
    NN.train(example_input, example_output)

    """
    if argv == 'example':
        NN = NeuralNetwork(number_layers=2, number_neurons_layer=[2, 2], loss_function='MSE',
                           activation_functions_layer=['logistic', 'logistic'], number_input_nn=2, learning_rate=0.5,
                           weights=None, bias=None)
        NN.print()
        a

    elif argv == 'and':
        pass

    elif argv == 'xor':
        pass
    """


if __name__ == '__main__':
    main(sys.argv)
