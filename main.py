import numpy as np
from scipy.sparse import spdiags
import matplotlib.pyplot as plt


def ex2a():
    G, x, y = generate_experiment()
    lambda_x = 80

    # Since in our case: A = I, we get that:
    # x = (I + (lambda_x/2) * GtG)^-1 * y
    # mark: z = (I + (lambda_x/2) * GtG)^-1 , so: x = z @ y
    I = np.eye(np.size(x))
    GtG = G.transpose() @ G
    z = np.linalg.inv(np.array(I + ((lambda_x / 2) * GtG), dtype=int))
    x_result = z @ y
    print("x: \n %s" % x_result)
    plt.plot(x, x_result)
    plt.show()


def ex2b():
    G, x, y = generate_experiment()
    IRLS(G, y, lambda_x=1, W=np.eye(np.size(x)), epsilon=0.001, number_of_iterations=10)


def IRLS(G, y, lambda_x, W, epsilon, number_of_iterations):
    I = W
    GtWtWG = G.transpose() @ W.transpose() @ W @ G
    for i in range(number_of_iterations):
        z = np.invert(np.array(I + ((lambda_x / 2) * GtWtWG), dtype=int))
        x_result = z @ y


def generate_experiment():
    x = np.arange(0, 5, 0.01)
    n = np.size(x)
    one = int(n / 5)
    f = np.zeros(x.shape)
    f[0: one] = 0.0 + 0.5 * x[0: one]
    f[one: (2 * one)] = 0.8 - 0.2 * np.log(x[100: 200])
    f[(2 * one): (3 * one)] = 0.7 - 0.2 * x[(2 * one): 3 * one]
    f[(3 * one): (4 * one)] = 0.3
    f[(4 * one): (5 * one)] = 0.5 - 0.1 * x[(4 * one): (5 * one)]
    G = spdiags([-np.ones(n), np.ones(n)], np.array([0, 1]), n - 1, n).toarray()
    etta = 0.1 * np.random.randn(np.size(x))
    y = f + etta
    plt.figure()
    plt.plot(x, y)
    plt.plot(x, f)
    plt.show()
    return G, x, y


def ex4a():
    return 0


if __name__ == '__main__':
    ex2a()
