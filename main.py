import random
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

    '''
    Since in our case: A = I, we get that for the original equation:
    arg min(x) = (x - y)t @ (x - y) + (lambda_x/2) * (G @ x)t @ (G @ x)
    The minimization of x is obtained by:
    x = (I + (lambda_x/2) * GtG)^-1 @ y
    By marking: z = (I + (lambda_x/2) * GtG)^-1
    We have: x = z @ y
    '''

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

    '''
    The minimization of x is obtained by:
    x = (I + (lambda_x) * GtWG)^-1 @ y
    By marking: z = I + (lambda_x) * GtWG
    We have: x = z^-1 @ y
    Where: W = 1/(|G @ curr_x| + epsilon), for curr_x in each iteration
    '''

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


def ex4a(x, y):
    w = np.zeros([x.shape[0], 1])
    return Logistic_Regression(w, x, y)


def Logistic_Regression(w, x, y, hessian_indicator=True):
    Cost_Fw = cost(w, x, y)
    Gradient_Fw = gradient(w, x, y)
    if hessian_indicator:
        Hessian_Fw = hessian(w, x, y)
        return Cost_Fw, Gradient_Fw, Hessian_Fw
    else:
        return Cost_Fw, Gradient_Fw


def net_input(x, w):
    return np.dot(x.transpose(), w)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def cost(w, x, y):
    m = x.shape[1]
    c1 = y
    c2 = 1 - y
    probability = sigmoid(net_input(x, w))
    return -(1 / m) * (c1.transpose() @ np.log(probability) + c2.transpose() @ np.log(1 - probability))


def gradient(w, x, y):
    m = x.shape[1]
    c1 = y
    probability = sigmoid(net_input(x, w))
    return (1 / m) * (x @ (probability - c1))


def hessian(w, x, y):
    m = y.shape[0]
    probability = sigmoid(net_input(x, w))
    probability_multiplication = probability * (1 - probability)
    D = np.asarray(probability_multiplication.transpose())[0]
    return (1 / m) * (np.multiply(x, D) @ x.transpose())


def ex4b():
    n = 20
    w = np.random.rand(n, 1)
    x = np.random.rand(n, 1)
    y = np.array([[random.randint(0, 1)]])
    Verification_Test(w, x, y)


def Verification_Test(w, x, y):
    d = np.random.rand(x.shape[0], 1)
    epsilon = 0.1
    n = 20
    F0 = function(w, x, y)
    G0 = grad(w, x, y)
    iterations = np.arange(0, n, 1)
    y0 = np.zeros(n)
    y1 = np.zeros(n)
    y2 = np.zeros(n)
    y3 = np.zeros(n)
    for k in range(n):
        epsilon_k = epsilon * pow(0.5, k)
        print(epsilon_k)
        ed = epsilon_k * d
        w_ed = w + ed
        Fk = function(w_ed, x, y)
        Gk = grad(w_ed, x, y)
        F1 = F0 + (epsilon_k * np.dot(d.transpose(), G0))
        G1 = G0 + JacMV(w, x, y, epsilon_k * d)
        y0[k] = np.abs(Fk - F0)
        y1[k] = np.abs(Fk - F1)
        y2[k] = np.linalg.norm(Gk - G0)
        y3[k] = np.linalg.norm(Gk - G1)
        print("Err 1 " + str(y0[k]) + " Err 2 " + str(y1[k]) + " Err 3 " + str(y2[k]) + " Err 4 " + str(y3[k]))

    plt.figure()
    plt.semilogy(iterations, y0, label="|f(w+εd)-f(w)|")
    plt.semilogy(iterations, y1, label="|f(w+εd)-f(w)-εdt*grad(w)|")
    plt.semilogy(iterations, y2, label="||grad(w+εd)-grad(w)||")
    plt.semilogy(iterations, y3, label="||grad(w+εd)-grad(w)-JacMV(w,εd)||")
    plt.xlabel("Iterations")
    plt.ylabel("Decrease Factors")
    plt.legend()
    plt.title("Gradient and Jacobian Tests")
    plt.show()


def function(w, x, y):
    return cost(w, x, y)


def grad(w, x, y):
    return gradient(w, x, y)


def JacMV(w, x, y, v):
    return np.transpose(hessian(w, x, y)) @ v


# function is running from MNIST
def ex4c_test(x1, y1, x2, y2, train_or_test):
    w1 = np.zeros((x1.shape[0], 1))
    w2 = np.zeros((x2.shape[0], 1))
    Objective_history1 = Gradient_Descent(w1, x1, y1)
    Objective_history2 = Exact_Newton(w1, x1, y1)
    Objective_history3 = Gradient_Descent(w2, x2, y2)
    Objective_history4 = Exact_Newton(w2, x2, y2)
    iterations = np.arange(0, 100, 1)

    if train_or_test == 0:
        plt.figure()
        plt.plot(iterations, Objective_history1, label="Train")
        plt.plot(iterations, Objective_history3, label="Test")
        plt.xlabel("Iterations")
        plt.ylabel("|f(w)-f(w*)|")
        plt.legend()
        plt.title("Gradient Descent: 0 vs 1")
        plt.show()

        plt.figure()
        plt.plot(iterations, Objective_history2, label="Train")
        plt.plot(iterations, Objective_history4, label="Test")
        plt.xlabel("Iterations")
        plt.ylabel("|f(w)-f(w*)||")
        plt.legend()
        plt.legend()
        plt.title("Exact Newton: 0 vs 1")
        plt.show()

    else:
        plt.figure()
        plt.plot(iterations, Objective_history1, label="Train")
        plt.plot(iterations, Objective_history3, label="Test")
        plt.xlabel("Iterations")
        plt.ylabel("|f(w)-f(w*)|")
        plt.legend()
        plt.legend()
        plt.title("Gradient Descent: 8 vs 9")
        plt.show()

        plt.figure()
        plt.plot(iterations, Objective_history2, label="Train")
        plt.plot(iterations, Objective_history4, label="Test")
        plt.xlabel("Iterations")
        plt.ylabel("|f(w)-f(w*)|")
        plt.legend()
        plt.legend()
        plt.title("Exact Newton: 8 vs 9")
        plt.show()


def ex4c():
    n = 20
    x = np.random.rand(n, 1)
    w = np.random.rand(n, 1)
    y = np.array([[1]])
    Objective_history1 = Gradient_Descent(w, x, y)
    Objective_history2 = Exact_Newton(w, x, y)
    iterations = np.arange(0, 100, 1)

    plt.figure()
    plt.plot(iterations, Objective_history1, label="GD")
    plt.plot(iterations, Objective_history2, label="EN")
    plt.xlabel("Iterations")
    plt.ylabel("|f(w)-f(w*)|")
    plt.legend()
    plt.title("Gradient And Newton Tests")
    plt.show()


def Gradient_Descent(w, x_train, y_train, x_test, y_test, alpha=1.0, iterations=100):
    train_cost_history = []
    test_cost_history = []
    d = np.zeros(w.shape)
    g_k = np.zeros(w.shape)
    f_k = 1
    for i in range(iterations):
        w = np.clip(w, -1, 1)
        if i != 0:
            alpha = Armijo_Linesearch(w, x_train, y_train, d, g_k)
        w += alpha * d
        if f_k < 0.001:
            break
        f_k, g_k = Logistic_Regression(w, x_train, y_train, hessian_indicator=False)
        d = -np.array(g_k)
        f_0 = cost(w, x_test, y_test)
        train_cost_history.append(f_k[0][0])
        test_cost_history.append(f_0[0][0])
    return w, np.array(train_cost_history), np.array(test_cost_history)


def Exact_Newton(w, x_train, y_train, x_test, y_test, alpha=1.0, iterations=100):
    train_cost_history = []
    test_cost_history = []
    d = np.zeros(w.shape)
    g_k = np.zeros(w.shape)
    f_k = 1
    for i in range(iterations):
        w = np.clip(w, -1, 1)
        if i != 0:
            alpha = Armijo_Linesearch(w, x_train, y_train, d, g_k)
        w += alpha * d
        if f_k < 0.001:
            break
        f_k, g_k, h_k = Logistic_Regression(w, x_train, y_train)
        h_k_regulated = h_k + (np.identity(h_k.shape[0]) * 0.01)
        d = -np.linalg.inv(h_k_regulated) @ g_k
        f_0 = cost(w, x_test, y_test)
        train_cost_history.append(f_k[0][0])
        test_cost_history.append(f_0[0][0])
    return w, np.array(train_cost_history), np.array(test_cost_history)


def Armijo_Linesearch(w, x, y, d, g_k, alpha=1.0, beta=0.8, c=1e-5):
    f_k = cost(w, x, y)
    for i in range(10):
        f_k_1 = cost(w + (alpha * d), x, y)
        if f_k_1 <= f_k + (alpha * c * np.dot(d.transpose(), g_k)):
            return alpha
        else:
            alpha = beta * alpha
    return alpha


def ex3f():
    theta = [1000000, 0.001, 110]
    ans = f(theta)
    y = usa_data_to_vector()
    print(ans)
    for i in range(1, 3):
        jacobi = calc_f_jacobi(theta)
        F_grad = jacobi.transpose() @ (f(theta) - y)
        theta -= F_grad * 0.000000000000001
    print("---------------------------------------------------------")
    print(f(theta))


def f(theta):
    return np.array([fi(theta, xi) for xi in range(1, 100)])


def fi(theta, xi):
    return theta[0] * math.exp(-theta[1] * ((xi - theta[2]) ** 2))


def calc_f_jacobi(theta):
    return np.array([_fi_gradient(theta, xi) for xi in range(1, 100)])


def _fi_gradient(theta, xi):
    t = (xi - theta[2]) ** 2
    g1 = math.exp(-theta[1] * t)
    g2 = -theta[0] * t * math.exp(-theta[1] * t)
    g3 = 2 * theta[0] * theta[1] * t * math.exp(-theta[1] * t ** 2)
    return [g1, g2, g3]


def usa_data_to_vector():
    with open("Covid-19-USA.txt") as file_in:
        lines = []
        for line in file_in:
            lines.append(int(line))
    return np.array(lines)


if __name__ == '__main__':
    x = np.array([[2, 1],
                  [2, 1],
                  [1, 1]])
    y = np.array([[1],
                  [0]])
    w = np.array([[0.5, 0.5]])
    # ex4a(x, y)
    ex4b()
    # ex4c()
