import numpy as np
from scipy.sparse import spdiags
import matplotlib.pyplot as plt
import math


def ex2():
    G, x, y = generate_experiment()
    x1 = ex2a(G, x, y)
    x2 = ex2b(G, x, y)
    plt.plot(x, x1, label='L2')
    plt.plot(x, x2, label='IRLS')
    plt.legend()
    plt.title("Approximated signal f")
    plt.show()


def ex2a(G, x, y):
    lambda_x = 80

    # Since in our case: A = I, we get that for the original equation:
    # arg min(x) = (x - y)t @ (x - y) + (lambda_x/2) * (G @ x)t @ (G @ x)
    # The minimization of x is obtained by:
    # x = (I + (lambda_x/2) * GtG)^-1 @ y
    # By marking: z = (I + (lambda_x/2) * GtG)^-1
    # We have: x = z @ y

    I = np.eye(np.size(x))
    GtG = G.transpose() @ G
    z = np.linalg.inv(np.array(I + ((lambda_x / 2) * GtG), dtype=int))
    x_result = z @ y
    return x_result


def ex2b(G, x, y):
    x_result = IRLS(G, y, lambda_x=1, W=np.eye(np.size(x) - 1), epsilon=0.001, number_of_iterations=10)
    return x_result


def IRLS(G, y, lambda_x, W, epsilon, number_of_iterations):
    I = np.eye(np.size(y))
    curr_x = 0

    # The minimization of x is obtained by:
    # x = (I + (lambda_x) * GtWG)^-1 @ y
    # By marking: z = I + (lambda_x) * GtWG
    # We have: x = z^-1 @ y
    # Where: W = 1/(|G @ curr_x| + epsilon), for curr_x in each iteration

    for i in range(number_of_iterations):
        GtWG = G.transpose() @ W @ G
        z = I + (lambda_x * GtWG)
        curr_x = np.linalg.inv(z) @ y
        W = calc_w(curr_x, G, epsilon)
    return curr_x


def calc_w(perv_x, G, epsilon):
    w_vector = G @ perv_x
    w_vector = np.abs(w_vector) + epsilon
    w_mat = np.diag(w_vector)
    w_ans = np.linalg.inv(w_mat)
    return w_ans


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
    plt.title("Original signal f")
    plt.show()
    return G, x, y


def ex4a(X, y):
    return Logistic_Regression(X, y, w=np.zeros(X.shape[1]))


def Logistic_Regression(X, y, w):
    number_of_labels = len(y)
    c1 = []
    c2 = []
    for i in range(number_of_labels):
        if y[i] == 0:
            c1.append(0)
            c2.append(1)
        else:
            c1.append(1)
            c2.append(0)
    c1 = np.array([c1]).transpose()
    c2 = np.array([c2]).transpose()

    # Xtw = X.transpose() @ np.array([w]).transpose()
    Xtw = xtw(X, w)
    probability = sigmoid(Xtw)

    Objective_Fw = Objective(X, c1, c2, probability)
    Gradient_Fw = Gradient(X, c1, probability)
    Hessian_Fw = Hessian(X, probability)

    # Fw = 0
    # for i in range(m):
    # xtw = np.transpose(X[:, i:i]) @ w
    # Fw += c1.transpose() @ np.log(sigmoid(xtw)) + c2.transpose() @ np.log(1 - sigmoid(xtw))
    # Fw = -1 / m * Fw

    # Gradient_Fw = np.zeros(1, m)
    # for i in range(m):
    # xtw = np.transpose(X[:, i:i]) @ w
    # Gradient_Fw[0][i] = sigmoid(xtw)
    # Gradient_Fw = 1 / m * (X @ (Gradient_Fw - c1))

    # Hessian_Fw = np.zeros(m, m)
    # for i in range(m):
    # xtw = np.transpose(X[:, i:i]) @ w
    # Hessian_Fw[i][i] = sigmoid(xtw) * (1 - sigmoid(xtw))
    # Hessian_Fw = 1 / m * (X @ Gradient_Fw @ X.transpose())
    print("4a")
    print(Objective_Fw)
    print(Gradient_Fw)
    print(Hessian_Fw)
    return Objective_Fw, Gradient_Fw, Hessian_Fw


def xtw(X, w):
    # return np.dot(X.transpose(), w)
    m = X.shape[1]
    xtw_result = np.zeros([m, 1])
    for i in range(m):
        xtw_result = xtw_result + X[:, i:i+1] * w[i]
    return xtw_result


def sigmoid(Xtw):
    return 1 / (1 + np.exp(-Xtw))


def Objective(X, c1, c2, probability):
    m = X.shape[1]
    # Objective_Fw = c1.transpose() @ np.log(probability) + c2.transpose() @ np.log(1 - probability)
    return (-1 / m) * (c1.transpose() @ np.log(probability) + c2.transpose() @ np.log(1 - probability))


def Gradient(X, c1, probability):
    m = X.shape[1]
    # Gradient_Fw = X @ (probability - c1)
    return (1 / m) * (X @ (probability - c1))


def Hessian(X, probability):
    m = X.shape[1]
    D = np.asarray(probability.transpose())[0]
    D = np.diag(D)
    # Hessian_Fw = X @ D @ X.transpose()
    return (1 / m) * (X @ D @ X.transpose())


def ex4b(X, y, w, epsilon):
    d = np.random.rand(1, X.shape[1])
    Xtw = xtw(X, w)
    probability = sigmoid(Xtw)
    X_ed = (X + (epsilon * d))
    X_edtw = xtw(X_ed, w)
    probability_new = sigmoid(X_edtw)
    Fx_ed = Objective(X_ed, y, 1 - y, probability_new)
    Fx = Objective(X, y, 1 - y, probability)
    Fx_Gradient = Gradient(X, y, probability)
    Fx_Hessian = Hessian(X, probability)
    print("4b")
    print(d)
    print(Fx_ed - Fx)
    print(np.abs(Fx_ed - Fx - (Fx_Gradient.transpose() @ (epsilon * d.transpose()))))
    print(np.linalg.norm(Fx_ed - Fx - ((epsilon * d) @ Fx_Hessian @ (epsilon * d.transpose()))))
    return True


if __name__ == '__main__':
    x = np.array([[2, 1, 1],
                  [2, 1, 2],
                  [1, 1, 2]])
    y = [1, 0, 1]
    w = [0.5, 0.5, 0.5]
    ex4a(x, y)

    #x = np.array([[1],
    #              [2],
    #              [1]])
    y = np.array([[1, 0, 1]]).transpose()
    ex4b(x, y, w, 0.1)
