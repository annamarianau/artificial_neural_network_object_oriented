import numpy as np

"""
Defining the following classes:
- Neuron
- FullyConnectedLayer
- NeuralNetwork
"""


class Neuron:
    def __init__(self, act_fun):
        self.actfun = act_fun


"""
Activation Functions
- logistic (log_act)
- linear (lin_act)
- z: result of the weighted sum of weights (w), biases (b), and inputs (x) - z = np.dot(w,x)-b
"""


def log_act(z):
    return 1/1+np.exp(-z)


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
    return np.sum(predicted-actual)**2/len(actual)


def bin_cross_entropy_loss(predicted, actual):
    return predicted,actual


# Driver code main()
def main():
    print("This is the driver code")


if __name__ == '__main__':
    main()