#------------ First Sample----------------------------
import numpy as np

def sigmoid(x, derivative=False):

    if (derivative == True):
        return sigmoid(x, derivative=False) * (1 - sigmoid(x, derivative=False))
    else:
        return 1/(1+np.exp(-x))


np.random.seed(1)
alpha = 0.1
num_hidden = 3

X = np.array([
    [0, 0, 1],
    [0, 1, 1],
    [1, 0, 0],
    [1, 1, 0],
    [1, 0, 1],
    [1, 1, 1],
])

y = np.array([[0, 1, 0, 1, 1, 0]]).T

hidden_weights = 2*np.random.random((X.shape[1] + 1, num_hidden)) - 1
output_weights = 2*np.random.random((num_hidden + 1, y.shape[1])) - 1
num_iterations = 10000


# for each iteration of the gradient descent
for i in range(num_iterations):

    # forward phase
    input_layer_outputs = np.hstack((np.ones((X.shape[0], 1)), X))

    hidden_layer_outputs = np.hstack((np.ones((X.shape[0], 1)), sigmoid(
        np.dot(input_layer_outputs, hidden_weights))))

    output_layer_outputs = np.dot(hidden_layer_outputs, output_weights)

    # backward phase
    output_error = output_layer_outputs - y

    # hidden layer error term and [:, 1:] removes the bias term from the backpropagation
    hidden_error = hidden_layer_outputs[:, 1:] * (
        1 - hidden_layer_outputs[:, 1:] * np.dot(output_error, output_weights.T[:, 1:]))

    # partial derivatives

    hidden_pd = input_layer_outputs[:, :,
                                    np.newaxis] * hidden_error[:, np.newaxis, :]
    output_pd = hidden_layer_outputs[:, :,
                                     np.newaxis] * output_error[:, np.newaxis, :]

    total_hidden_gradient = np.average(hidden_pd, axis=0)
    total_output_gradient = np.average(output_pd, axis=0)

    hidden_weights += - alpha * total_hidden_gradient
    output_weights += - alpha * total_output_gradient

print("Output after Training: \n{}".format(output_layer_outputs))

#-----------------Second Sample ------------------------------------------------

def sigmoid(x):
    return 1.0/(1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1.0 - x)

class NeuralNetwork:
    def __init__(self, x, y):
        self.input = x
        self.weights1 = np.random.rand(self.input.shape[1], 4)
        self.weights2 = np.random.rand(4, 1)
        self.y = y
        self.output = np.zeros(self.y.shape)

    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))

    def backpropagation(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))

    def backpropagation(self):
        d_weights2 = np.dot(
            self.layer1.T, (2*(self.y - self.output) * sigmoid_derivative(self.output)))

        d_weights1 = np.dot(self.input.T, (np.dot(2*(self.y - self.output) * sigmoid_derivative(
            self.output), self.weights2.T) * sigmoid_derivative(self.layer1)))

     # we update the weights with the derivative of the loss function
        self.weights1 += d_weights1
        self.weights2 += d_weights2


if __name__ == "__main__":
    X = np.array([[0, 0, 1],
                  [0, 1, 1],
                  [1, 0, 1],
                  [1, 1, 1]])

    y = np.array([[0], [1], [1], [0]])

    nn = NeuralNetwork(X, y)

for i in range(1800):
    nn.feedforward()
    nn.backpropagation()

print(nn.output)
