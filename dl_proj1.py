##############################
#### COSC 525 - Project 1 ####
#### Neural Network - OOP ####
####  Anna-Maria Nau      ####
####  Christoph Metzner   ####
####      01/24/2020      ####
##############################


import numpy as np
import sys
import copy

"""
This script contains three classes:
- Neuron
- FullyConnectedLayer
- NeuralNetwork
User of program effectively works only with the NeuralNetwork class
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

    def __init__(self, activation_function, num_input, learn_rate, weights=None, bias=None):
        self.activation_function = activation_function
        self.number_input = num_input
        self.learn_rate = learn_rate
        self.bias = bias
        self.updated_bias = None
        self.output = None
        self.delta = None
        self.updated_weights = []
        # If no tuple with weights is given then generate number of weights based on number of inputs
        # If tuple with weights is given then self.weights = weights
        if weights is None:
            self.weights = np.random.uniform(0, 1, self.number_input)
        else:
            self.weights = weights

    """
    Class Method
    Selection of activation function with input variable z
    z = bias + weights*input (self.bias + self.weights * input_vec)
    logistic (log_act(z)): 1 / 1 + exp(-z)
    linear (lin_ac(z)): z
    """

    def activate(self, z):
        if self.activation_function == "logistic":
            return log_act(z)
        elif self.activation_function == "linear":
            return lin_act(z)

    # calculate - class method - computes the weighted sum of inputs
    # for respective activation function (activate() method is called)
    def calculate(self, input_previous_layer):
        return self.activate(self.bias + np.dot(self.weights, input_previous_layer))

    # Method to calculate the delta values for the neuron if in the output layer
    def calculate_delta_output(self, actual_output_network):
        if self.activation_function == "logistic":
            return -(actual_output_network - self.output) * log_act_prime(self.output)
        elif self.activation_function == "linear":
            return -(actual_output_network - self.output) * lin_act_prime(self.output)

    # Method to calculate the delta values for the neuron if in the hidden layer
    def calculate_delta_hidden(self, delta_sum):
        if self.activation_function == "logistic":
            return delta_sum * log_act_prime(self.output)
        elif self.activation_function == "linear":
            return delta_sum * lin_act_prime(self.output)

    # Method which updates the weight and bias
    # class attributes self.updated_weights and self.updated_bias are cleared or set to None
    # for next training sample iteration
    def update_weights_bias(self):
        self.weights = copy.deepcopy(self.updated_weights)
        self.updated_weights.clear()
        self.bias = copy.deepcopy(self.updated_bias)
        self.updated_bias = None


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
            if num_neurons == 1:
                for i in range(num_neurons):
                    self.neurons.append(
                        Neuron(activation_function=activation_function, num_input=self.number_input,
                               learn_rate=self.learn_rate,
                               weights=self.weights, bias=self.bias))
            else:
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
            print("Output Neuron:", neuron.output)
        return output_vec

    def update_weights_bias(self):
        for neuron in self.neurons:
            neuron.update_weights_bias()


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

    def feed_forward(self, input_vec, actual_output_network):
        """
        Feedforward algorithm
        for-loop to compute individual (ind_loss) and total loss (total_loss) of each layers (hidden and output) of the
        network
        original input variable (input_vec; "input neurons") of network is updated to values of next layer neurons output
        values
        """
        # generating a unique vector containing original network input values
        # used to compute the deltas and weights of the final hidden layer (seen from the back-propagation algorithm)
        input_network = input_vec

        # loop through each layer of network
        print("Feed-forward algorithm:")
        for i, layer in enumerate(self.FullyConnectedLayers):
            #if i == (len(self.FullyConnectedLayers) - 1):
            #    print("Outputs at Output Layer: ")
            #else:
            #    print("Outputs at hidden layer: ", i + 1)
            # computing output of each neuron in each layer
            # calling FullyConnectedLayer method layer.calculate with variable input_vec
            input_vec = layer.calculate(input_vec)

        # computing the individual losses for each output neuron
        ind_loss = self.calculateloss(input_vec, actual_output_network)
        #print("List with individual Losses for output neurons - Output_1 and Output_2: ", ind_loss)
        # computing total loss by summing individual losses together
        total_loss = np.sum(ind_loss)
        print("Total Loss accrued in Network: ", total_loss)
        print()
        return input_network

    def back_propagation(self, vec_input_neurons, actual_output_network):
        """
        Backpropagation Algorithm
        """
        print("Back-propagation algorithm:")
        # Reversing list with layer objects for back propagation
        # back prop starts with updating weights connected to output layer
        self.FullyConnectedLayers.reverse()











        # Backpropagation for AND-Gate
        print(len(self.FullyConnectedLayers))
        if len(self.FullyConnectedLayers) == 1:
            if len(self.FullyConnectedLayers[0].neurons) == 1:
                neuron = self.FullyConnectedLayers[0].neurons[0]
                neuron.delta = neuron.calculate_delta_output(actual_output_network)
                print(neuron.delta)
                print(vec_input_neurons)
            error_weight = neuron.delta * vec_input_neurons
            for k in range(len(neuron.weights)):
                updated_weight = neuron.weights[k] - self.learn_rate * error_weight
                neuron.updated_weights.append(updated_weight)
            print("Current weights: {} --> updated weights: {}".format(neuron.weights, neuron.updated_weights))
            # Updating the bias for each neuron w.r.t. their delta
            neuron.updated_bias = neuron.bias - self.learn_rate * neuron.delta
            print("Current bias: {} --> updated bias: {}".format(neuron.bias, neuron.updated_bias))
        else:
            for i, layer in enumerate(self.FullyConnectedLayers):
                # if statement is used to check of type of layer:
                # i=0 --> output layer
                # i>0 --> hidden layer
                if i == 0:
                    # print("Updated weights connected with output neurons:")
                    # loop through layer for every neuron
                    for j, neuron in enumerate(layer.neurons):
                        # for loop which loops through the neurons of first hidden layer
                        neuron.delta = neuron.calculate_delta_output(actual_output_network[j])
                        for k, neuron_hidden in enumerate(self.FullyConnectedLayers[i + 1].neurons):
                            # compute the error of each individual weight based on output of neurons of previous hidden layer
                            error_weight = neuron.delta * neuron_hidden.output
                            # computing future weights; weights are used in next training iteration
                            updated_weight = neuron.weights[k] - self.learn_rate * error_weight
                            # store the values for the updated_weight in a vector
                            # this vector is later used to overwrite the current weight vector after computing
                            # all weights updates
                            neuron.updated_weights.append(updated_weight)
                        print("Current weights: {} --> updated weights: {}".format(neuron.weights,
                                                                                   neuron.updated_weights))
                        # Updating the bias for each neuron w.r.t. their delta
                        neuron.updated_bias = neuron.bias - self.learn_rate * neuron.delta
                        print("Current bias: {} --> updated bias: {}".format(neuron.bias, neuron.updated_bias))
                else:
                    # print("Updated weights of neurons in hidden layer {} next to output layer:".format(i))
                    for j, neuron in enumerate(layer.neurons):
                        # computing the respective sum of deltas times weight from previous layer
                        sum = 0  # setting sum to 0 for each neuron in hidden layer
                        for m in range(len(layer.neurons)):
                            sum += self.FullyConnectedLayers[i - 1].neurons[m].delta * \
                                   self.FullyConnectedLayers[i - 1].neurons[m].weights[j]
                        # computing the delta of current hidden layer
                        for k in range(len(neuron.weights)):
                            neuron.delta = neuron.calculate_delta_hidden(sum)
                            # If-statement: Determining if NN is at the input layer or not
                            # If at input layer: Use original input vector to determine the error weight
                            # If at another hidden layer: use the output of the neuron
                            if i == (self.number_layers - 1):
                                error_weight = neuron.delta * vec_input_neurons[k]
                            else:
                                error_weight = neuron.delta * self.FullyConnectedLayers[i + 1].neurons[j].output
                            # Computing the updated weights based on the current weight - learning rate * times error of weight
                            updated_weight = neuron.weights[k] - self.learn_rate * error_weight
                            # storing updated weights in vector "updated_weight" as an attribute of each neuron
                            # Storing is necessary for computing errors of all weights in successive hidden layers
                            # Later the weights are updated by setting neuron.weights = neuron.updated_weights
                            neuron.updated_weights.append(updated_weight)
                        print("Current weights: {} --> updated weights: {}".format(neuron.weights,
                                                                                   neuron.updated_weights))
                        # Updating the bias for each neuron w.r.t. their delta
                        neuron.updated_bias = neuron.bias - self.learn_rate * neuron.delta
                        print("Current bias: {} --> updated bias: {}".format(neuron.bias, neuron.updated_bias))
                print()
            print()
            print()

    def update_weights_bias(self):
        # reverse the order of the list containing the individual layer objects
        # necessary for next samples training iteration --> correct feed forward information
        self.FullyConnectedLayers.reverse()
        for layer in self.FullyConnectedLayers:
            layer.update_weights_bias()

    def train(self, num_samples, input_vec, actual_output_network):
        # set up if statement if commandline argument is "example" then execute this one"
        """
        for i in range(num_samples):
            print("Input: {}, Actual Output: {}".format(input_vec, actual_output_network))
            print()
            input_vec = self.feed_forward(input_vec, actual_output_network)
            self.back_propagation(input_vec, actual_output_network)
            self.update_weights_bias()
        """


        # elif command line argument is "and" then execute this code.

        for j in range(num_samples):
            for i, sample in enumerate(input_vec):
                print("Input: {}, Actual Output: {}".format(sample, actual_output_network[i]))
                print()
                input_vec = self.feed_forward(sample, actual_output_network[i])
                self.back_propagation(input_vec, actual_output_network[i])
                self.update_weights_bias()
        """
        
        # elif command line argument is "xor" then execute this code.
        for j in range(num_samples):
            for i, sample in enumerate(input_vec):
                print("Input: {}, Actual Output: {}".format(sample, actual_output_network[i]))
                print()
                output = self.feed_forward(sample, actual_output_network[i])
                self.back_propagation(output, actual_output_network[i])
                self.update_weights_bias()
        """

"""
Activation Functions with their respective prime functions
- logistic (log_act)
- linear (lin_act)
- z: result of the weighted sum of weights (w), biases (b), and inputs (x) - z = np.dot(w,x)-b
"""
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


def bin_cross_entropy_loss(num_samples, predicted_output, actual_output):
    return 1/num_samples * np.sum(-(actual_output*np.log(predicted_output)
                                    + (1-actual_output)*np.log(1-predicted_output)))


vec_AF = ["logistic", "logistic"]
weights_TEST = [[(0.15, 0.2), (0.25, 0.3)], [(0.4, 0.45), (0.50, 0.55)]]
bias_Test = [0.35, 0.60]


input_vec = [0.05, 0.10]
actual_output_network = [0.01, 0.99]

input_XOR_AND = [[0, 0], [0, 1], [1, 0], [1, 1]]
output_XOR = [0, 1, 1, 0]
output_AND = [0, 0, 0, 1]


# Driver code main()
def main():
    global input_vec

    """
    NN = NeuralNetwork(num_layers=2, num_neurons_layer=[2, 2], vec_activation_function=vec_AF, num_input=2,
                       loss_function="MSE", learn_rate=0.5, weights_network=weights_TEST, bias_network=bias_Test)
    # setting input_vec to global, since the variable has to be changed for feed forward information
    # output of neurons from layer will be input of neurons in following layer
    global input_vec
    NN.train(num_samples=2, input_vec=input_vec, actual_output_network=actual_output_network)
    """
    """
    #NN_xor_standard = NeuralNetwork(num_layers=1, num_neurons_layer=[1], vec_activation_function=vec_AF, num_input=2,
    #                   loss_function="MSE", learn_rate=0.5, weights_network=None, bias_network=None)
    # setting input_vec to global, since the variable has to be changed for feed forward information
    # output of neurons from layer will be input of neurons in following layer
    #NN_xor_standard.train(num_samples=10, input_vec=input_XOR_AND, actual_output_network=output_XOR)

    #NN_xor_improved = NeuralNetwork(num_layers=2, num_neurons_layer=[2, 2], vec_activation_function=vec_AF, num_input=2, loss_function="MSE", learn_rate=0.5, weights_network=None, bias_network=None)
    #NN_xor_improved.train(num_samples=1, input_vec=input_XOR_AND, actual_output_network=output_XOR)
    """

    NN = NeuralNetwork(num_layers=1, num_neurons_layer=[1], vec_activation_function=vec_AF, num_input=2,
                       loss_function="MSE", learn_rate=6, weights_network=None, bias_network=None)

    global input_vec
    NN.train(num_samples=10, input_vec=input_XOR_AND, actual_output_network=output_AND)



if __name__ == '__main__':
    main()