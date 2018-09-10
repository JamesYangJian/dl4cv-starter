import numpy as np

class Perceptron:
    def __init__(self, N, alpha):
        self.W = np.random.randn(N+1)/np.sqrt(N)
        self.alpha = alpha

        print('[INFO] Initial W ={}'.format(self.W))

    def step(self, v):
        if v >= 0:
            return 1
        else:
            return 0

    def train(self, epochs, X, y):
        X = np.c_[X, np.ones(X.shape[0])]

        for i in range(epochs):
            for (x, target) in zip(X, y):
                p = self.step(np.dot(x, self.W))
                error = p - target
                if error != 0:
                    self.W += -self.alpha*error*x

        print('[INFO] Trained W ={}'.format(self.W))

    def predict(self, x):
        x = np.atleast_2d(x)
        x = np.c_[x, np.ones(x.shape[0])]

        p = self.step(np.dot(x, self.W))

        return p

