import numpy as np
from scipy.sparse import spdiags
import matplotlib.pyplot as plt


def ex2a():
    G, x = generate_experiment()
    lambda_x = 80
    Gx = G @ x
    result = (lambda_x / 2) * pow((np.linalg.norm(Gx, 2), 2))
    print("result: \n %s" % result)


def generate_experiment():
    x = np.arange(0, 5, 0.01)
    n = np.size(x)
    one = int(n / 5)
    f = np.zeros(x.shape)
    f[0: one] = 0.0 + 0.5 * x[0: one]
    f[one: (2 * one)] = 0.8 - 0.2 * np.log(x[100: 200])
    f[(2 * one): (3 * one)] = 0.7 - 0.3 * x[(2 * one): 3 * one]
    f[(3 * one): (4 * one)] = 0.3
    f[(4 * one): (5 * one)] = 0.5 - 0.1 * x[(4 * one): (5 * one)]
    G = spdiags([[-np.ones(n - 1)], [np.ones(n)]], [-1, 0], n, n)
    etta = 0.1 * np.random.randn(np.size(x))
    y = f + etta
    plt.figure()
    plt.plot(x, y)
    plt.plot(x, f)
    plt.show()
    return G, x


def ex4a():
    return 0


if __name__ == '__main__':
    ex2a()
