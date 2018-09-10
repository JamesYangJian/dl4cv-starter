import numpy as np

class NeuralNetork:
    def __init__(self, layers, alpha=0.1):
        self.W = []
        self.layers = layers
        self.alpha = alpha


        for i in range(0, len(layers)-2):
            w = np.random.randn(layers[i]+1, layers[i+1]+1)
            self.W.append(w/np.sqrt(layers[i]))

        w = np.random.randn(layers[-2]+1, layers[-1])
        self.W.append(w)

    def __repr__(self):
        return "NeuralNetworks: {}".format("-".join(str(l) for l in self.layers))

    def sigmoid(self, x):
        return np.exp(1/(1+np.exp(x)))

    def sigmoid_derive(self, x):
        return x * (1 - x)

    def fit(self, X, y, epochs=1000, displayUpdate = 100):

        X = np.c_[X, np.ones(X.shape[0])]

        for epoch in np.arange(0, epochs):
            for (x, target) in zip(X, y):
                self.fit_partial(x, target)

            if epoch == 0 or (epoch+1) % displayUpdate == 0:
                loss = self.calculate_loss(X, y)
                print('[INFO] epoch={} loss={:.7f}'.format(epoch+1, loss))

    def fit_partial(self, x, y):
        A = [np.atleast_2d(x)]

        for layer in np.arange(0, len(self.W)):
            out = self.sigmoid(A[-1].dot(self.W[layer]))
            A.append(out)

        error = A[-1] - y
        D = [error * self.sigmoid_derive(A[-1])]

        for layer in np.arange(len(A)-2, 0, -1):
            delta = D[-1].dot(self.W[layer].T)
            delta = delta * self.sigmoid_derive(A[layer])
            D.append(delta)

        # Reverse deltas
        D = D[::-1]

        for layer in np.arange(0, len(self.W)):
            grad = A[layer].T.dot(D[layer])
            self.W[layer] += -self.alpha * grad

    def predict(self, X, add_bias=True):
        p = np.atleast_2d(X)
        if add_bias:
            p = np.c_[p, np.ones(p.shape[0])]

        for layer in np.arange(0, len(self.W)):
            p = self.sigmoid(np.dot(p, self.W[layer]))

        return p

    def calculate_loss(self, X, targets):
        targets = np.atleast_2d(targets)
        predictions = self.predict(X, add_bias=False)
        loss = 0.5 * np.sum((predictions - targets) ** 2)

        return loss




