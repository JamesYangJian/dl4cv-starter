import numpy as np
from neuralnetwork import NeuralNetork


if __name__ == '__main__':
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    nn = NeuralNetork([2, 2, 1], alpha=0.5)
    nn.fit(X, y, epochs = 20000)

    for (x, target) in zip(X, y):
        step = 0
        pred = nn.predict(x)[0][0]
        if pred > 0.5:
            step = 1
        print("[INFO] data={}, ground-truth={}, pred={:.4f}, step={}".format(
            x, target[0], pred, step))
