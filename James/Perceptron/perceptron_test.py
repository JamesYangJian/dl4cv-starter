import numpy as np
from perceptron import Perceptron
import sys


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python {} method[AND|OR|XOR|AOR]')
        sys.exit(0)

    method = sys.argv[1]
    X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    if method == 'AND':
      y = np.array([[0], [0], [0], [1]])
    elif method == 'OR':
        y = np.array([[0], [1], [1], [1]])
    elif method == 'XOR':
        y = np.array([[0], [1], [1], [0]])
    elif method == 'AOR':
        y = np.array([[1], [0], [0], [1]])
    else:
        print('[WARN] Unknown method {}'.format(method))		

    p = Perceptron(X.shape[1], 0.3)
    p.train(20, X, y)

    for (x, target) in zip(X, y):
        pred = p.predict(x)
        print('[INFO] method={} data={}, ground_truth={} predict={}'.format(method, x, target[0], pred))
