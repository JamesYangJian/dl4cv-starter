import numpy as np

class NeuralNetwork:

    def __init__(self, layers, alpha=0.1):
        self.W = []
        # A list of integers which represents the actual architecture of the
        # feedforward network. For example, a value of [2, 2, 1] would imply
        # that our first layer has 2 nodes, our hidden layer has 2 nodes, and our
        # final output layer has one node.
        self.layers = layers
        self.alpha = alpha

        # start looping from the index of the first layers
        # but stop before we reach the last 2 layers
        for i in np.arange(0, len(layers) - 2):
            # randomly initialize a weight matrix connecting the number of nodes
            # in each respective layer together, adding an extra node for the
            # bias.
            w = np.random.randn(layers[i] + 1, layers[i + 1] + 1)
            self.W.append(w / np.sqrt(layers[i]))

        # the last 2 layers are a special case where the input connections need
        # a bias term but the output does not
        w = np.random.randn(layers[-2] + 1, layers[-1])
        self.W.append(w / np.sqrt(layers[-2]))

    def __repr__(self):
        # construct and return a string that represents the network
        # architecture
        return "NeuralNetwork: {}".format("-".join(str(l) for l in self.layers))

    def sigmoid(self, x):
        # compute and return the sigmoid activation value for a given inputs
        return 1.0 / (1 + np.exp(-x))

    def sigmoid_deriv(self, x):
        # compute the derivative of the sigmoid function ASSUMING that "x" has
        # already been passed through the sigmoid function
        return x * (1 - x)

    def fit(self, X, y, epochs=1000, display_update=100):
        # insert a column of ones as the last entry in the feature matrix --
        # this little trick allows us to treat the bias as a trainable parameter
        # within the weight matrix
        X = np.c_[X, np.ones((X.shape[0]))]

        # loop over the desired number of epochs
        for epoch in np.arange(0, epochs):
            # loop over each individual data point and train our network
            # on it
            for (x, target) in zip(X, y):
                self.fit_partial(x, target)

            # check to see if we should display a training update
            if epoch == 0 or (epoch + 1) % display_update == 0:
                loss = self.calculate_loss(X, y)
                print("[INFO] epoch={}, loss={:.7f}".format(
					epoch + 1, loss))

    def fit_partial(self, x, y):
        # construct our list of output activations for each layer as our data
        # point flows through the network; the first activation is a special
        # it's just the input feature vector itself
        A = [np.atleast_2d(x)]

        # FEEDFORWARD:
        # loop over the layers in the network
        for layer in np.arange(0, len(self.W)):
            # feedforward the activation at the current layer by taking the dot
            # product between the activation and the weight matrix. this is
            # called "net input" to the current layer.
            net = A[layer].dot(self.W[layer])

            # computing the "net output" is simply applying our nonlinear
            # activation function to the net input
            out = self.sigmoid(net)

            #once we have the net output, add it to our list of activations
            A.append(out)

        # BACKPROPAGATION:
        # the first phase of backpropagation is to compute the difference
        # between our *prediction* (the final output activation in the
        # activations list) and the true target value
        error = A[-1] - y

        # from here we need to apply the chain rule and build our list of deltas
        # 'D'; the first entry in the deltas is simply the error of the output
        # layer times the derivative of our activation function for the output
        # value.
        D = [error * self.sigmoid_deriv(A[-1])]

        # once you understand the chain rule, it becomes super easy to
        # implement with a 'for' loop -- simply loop over the layers in reverse
        # order (ignoring the last 2 since we already have taken them into
        # account)
        for layer in np.arange(len(A) - 2, 0, -1):
            # the delta for the current layer is equal to the delta of the
            # *previous layer* dotted with the weight matrix of the current
            # layer, followed by multiplying the delta by the derivative of the
            # nonlinear activation function for the activations of the current
            # layer
            delta = D[-1].dot(self.W[layer].T)
            delta = delta * self.sigmoid_deriv(A[layer])
            D.append(delta)

        # since we looped over our layers in reverse order we need to reverse
        # the deltas
        D = D[::-1]

        # WEIGHT UPDATE PHASE:
        # loop ove the layers
        for layer in np.arange(0, len(self.W)):
            # update our weights by taking the dot product of the layer
            # activations with their respective deltas, then multiplying
            # this value by some small learning rate and adding to our weight
            # matrix -- this is where the actual "learning" takes palce
            self.W[layer] += -self.alpha * A[layer].T.dot(D[layer])

    def calculate_loss(self, X, targets):
        # make predictions for the input data points then compute the loss
        targets = np.atleast_2d(targets)

        predictions = self.predict(X, add_bias=False)
        loss = 0.5 * np.sum((predictions - targets) ** 2)
        return loss

    def predict(self, X, add_bias=True):
        # initialize the output prediction as the input features -- this value
        # will be (forward) propogated through the network to obtain the final
        # prediction
        p = np.atleast_2d(X)

        # check to see if the bias column should be added
        if add_bias:
            # insert column of 1's as last entry in the feature matrix (bias)
            p = np.c_[p, np.ones((p.shape[0]))]

        # loop over our layers in the network
        for layer in np.arange(0, len(self.W)):
            p = self.sigmoid(np.dot(p, self.W[layer]))

        # return the predicted value
        return p
