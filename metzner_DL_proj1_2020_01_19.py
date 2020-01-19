import numpy as np

"""
Defining the following classes:
- Neuron
- FullyConnectedLayer
- NeuralNetwork
"""


class Neuron:
    # Initializing Neuron class with an activation function (act_fun), input values (input_vals), learning rate (learn
    # rate), vector with weights based on the number of neurons of previous layer / input layer (weights), and
    # one value for the bias of a neuron. Weights and bias are randomized between 0 and 1 from a uniform distribution if
    # no optional argument is given.
    def __init__(self, activation_function, input_vals, learn_rate, weights=True, bias=True):
        self.activation_function = activation_function
        self.input_vals = input_vals
        self.learn_rate = learn_rate
        self.weights = weights if weights is not True else np.random.uniform(0, 1, len(input_vals))
        self.bias = bias if bias is not True else np.random.uniform(0, 1)


    def print_activate(self):
        print(self.bias)
        print(self.input_vals)
        print(self.weights)
        print(self.activate())
        #print(self.calculate())

    # Class method activate - phi(bias + sum of weights * input)
    def activate(self):
        return self.bias + np.dot(self.weights,self.input_vals)

    def calculate(self):
        return self.actfun(self.activate())



class FullyConnectedLayer:
    def __init__(self):
        pass


class NeuralNetwork:
    def __init__(self):
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


def mse_loss(predicted, actual):
    return np.sum(predicted - actual) ** 2 / len(actual)


def bin_cross_entropy_loss(predicted, actual):
    return predicted, actual


test_weights = [1, 2, 3, 4, 5]
test_input = [5, 4, 3, 2, 1]
test_bias = 1
a = 1

# Driver code main()
def main():
    print("This is the driver code")
    first_neuron = Neuron(a, test_input, 0.1, test_weights, test_bias)
    first_neuron.activate()
    first_neuron.print_activate()


if __name__ == '__main__':
    main()
