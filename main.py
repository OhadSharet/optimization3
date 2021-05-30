import numpy as np
from scipy.sparse import spdiags
import matplotlib.pyplot as plt


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
    return Logistic_Regression(x, y, w=np.zeros([x.shape[0], 1]))


def Logistic_Regression(x, y, w):
    c1 = y
    c2 = 1 - y
    xtw = net_input(x, w)
    probability = sigmoid(xtw)

    Objective_Fw = Objective(x, c1, c2, probability)
    Gradient_Fw = Gradient(x, c1, probability)
    Hessian_Fw = Hessian(x, probability)

    # print("4a")
    # print(Objective_Fw)
    # print(Gradient_Fw)
    # print(Hessian_Fw)
    return Objective_Fw, Gradient_Fw, Hessian_Fw


def net_input(x, w):
    return np.dot(x.transpose(), w)


def sigmoid(xtw):
    return 1 / (1 + np.exp(-xtw))


def Objective(x, c1, c2, probability):
    m = x.shape[1]
    # Objective_Fw = c1.transpose() @ np.log(probability) + c2.transpose() @ np.log(1 - probability)
    return (-1 / m) * np.sum(c1.transpose() @ np.log(probability) + c2.transpose() @ np.log(1 - probability))


def Gradient(x, c1, probability):
    m = x.shape[1]
    # Gradient_Fw = X @ (probability - c1)
    return (1 / m) * (x @ (probability - c1))


def Hessian(x, probability):
    m = x.shape[1]
    D = np.asarray(probability.transpose())[0]
    D = np.diag(D)
    # Hessian_Fw = X @ D @ X.transpose()
    return (1 / m) * (x @ D @ x.transpose())


def ex4b():
    n = 20
    x = np.random.rand(n, 1)
    w = np.random.rand(n, 1)
    y = np.array([[0]])
    verification_test(x, y, w)

    # list_of_epsilons = np.arange(0.0, 0.6, 0.1)
    # d = np.random.rand(x.shape[0], 1)
    # gradient_test_results = []
    # jacobi_test_results = []
    # for i in list_of_epsilons:
    #    epsilon = i
    #    Xtw = net_input(X, w)
    #    probability = sigmoid(Xtw)
    #    X_ed = (X + (epsilon * d))
    #    X_edtw = net_input(X_ed, w)
    #    probability_new = sigmoid(X_edtw)
    #    Fx_ed = Objective(X_ed, y, 1 - y, probability_new)
    #    Fx = Objective(X, y, 1 - y, probability)
    #    Fx_Gradient = Gradient(X, y, probability)
    #    Fx_Gradient_ed = Gradient(X_ed, y, probability_new)
    #    Fx_Hessian = Hessian(X, probability)
    #    print("4b")
    #    # print(d)
    #    print(Fx_ed - Fx)
    #    print(np.abs(Fx_ed - Fx - ((epsilon * d.transpose()) @ Fx_Gradient)))
    #    print(np.linalg.norm(Fx_Gradient_ed - Fx_Gradient))
    #    print(np.linalg.norm(Fx_Gradient_ed - Fx_Gradient - (epsilon * (Fx_Hessian @ d))))

    # gradient_test_results.append(np.abs(Fx_ed - Fx - ((epsilon * d.transpose()) @ Fx_Gradient))[0][0])
    # jacobi_test_results.append(np.abs(Fx_ed - Fx - (epsilon * epsilon * (d.transpose() @ Fx_Hessian @ d)))[0][0])


def verification_test(x, y, w):
    d = np.random.rand(x.shape[0], 1)
    epsilon = 0.1
    n = 11
    F0 = function(x, y, w)
    G0 = grand(x, y, w)
    iterations = np.arange(0, n, 1)
    y0 = np.zeros(n)
    y1 = np.zeros(n)
    y2 = np.zeros(n)
    y3 = np.zeros(n)
    for k in range(n):
        epsilon_k = epsilon * k
        print(epsilon_k)
        ed = epsilon_k * d
        x_ed = x + ed
        Fk = function(x_ed, y, w)
        Gk = grand(x_ed, y, w)
        F1 = F0 + (epsilon_k * np.dot(d.transpose(), G0))
        G1 = G0 + JacMV(x, w, epsilon_k * d)
        y0[k] = np.abs(Fk - F0)
        y1[k] = np.abs(Fk - F1)
        y2[k] = np.linalg.norm(Gk - G0)
        y3[k] = np.linalg.norm(Gk - G1)
        print("Err 1 " + str(y0[k]) + " Err 2 " + str(y1[k]) + " Err 3 " + str(y2[k]) + " Err 4 " + str(y3[k]))

    plt.figure()
    plt.plot(iterations, y0, label="|f(x+εd)-f(x)|")
    plt.plot(iterations, y1, label="|f(x+εd)-f(x)-εdt*grand(x)|")
    plt.plot(iterations, y2, label="||grand(x+εd)-grand(x)||")
    plt.plot(iterations, y3, label="||grand(x+εd)-grand(x)-JacMV(x,εd)||")
    plt.legend()
    plt.title("Gradient and Jacobian Tests")
    plt.show()


def function(x, y, w):
    xtw = net_input(x, w)
    probability = sigmoid(xtw)
    return Objective(x, y, 1-y, probability)


def grand(x, y, w):
    xtw = net_input(x, w)
    probability = sigmoid(xtw)
    return Gradient(x, y, probability)


def JacMV(x, w, v):
    xtw = net_input(x, w)
    probability = sigmoid(xtw)
    return np.transpose(Hessian(x, probability)) @ v


def ex4c():
    n = 20
    x = np.random.rand(n, 1)
    w = np.random.rand(n, 1)
    y = np.array([[0]])
    Objective_history1 = Gradient_Descent(x, y, w)
    Objective_history2 = Exact_Newton(x, y, w)
    iterations = np.arange(0, 100, 1)

    plt.figure()
    plt.plot(iterations, Objective_history1, label="GD")
    plt.plot(iterations, Objective_history2, label="EN")
    plt.legend()
    plt.title("Gradient And Newton Tests")
    plt.show()


def Gradient_Descent(x, y, w, alpha=0.01, iterations=100):
    Objective_history = np.zeros(iterations)
    for i in range(iterations):
        xtw = net_input(x, w)
        probability = sigmoid(xtw)
        f_k = Objective(x, y, 1-y, probability)
        g_k = Gradient(x, y, probability)
        d = np.array(-g_k)
        alpha = Armijo_Linesearch(x, y, w, d, g_k, alpha=alpha)
        w = np.clip(w + (alpha * d), -1, 1)
        Objective_history[i] = f_k

    return Objective_history


def Exact_Newton(x, y, w, alpha=1.0, iterations=100):
    Objective_history = np.zeros(iterations)
    for i in range(iterations):
        f_k, g_k, h_k = Logistic_Regression(x, y, w)
        h_k_regulated = h_k + np.eye(x.shape[0])
        d = -np.linalg.inv(h_k_regulated) @ g_k
        alpha = Armijo_Linesearch(x, y, w, d, g_k, alpha=alpha)
        w = np.clip(w + (alpha * d), -1, 1)
        Objective_history[i] = f_k

    return Objective_history


def Armijo_Linesearch(x, y, w, d, g_k, alpha=1.0, beta=0.5, c=1e-4):
    xtw = net_input(x, w)
    probability = sigmoid(xtw)
    f_k = Objective(x, y, 1-y, probability)
    for i in range(10):
        f_k_1 = Objective(x + (alpha * d), y, 1 - y, probability)
        if f_k_1 <= f_k - (alpha * c * np.dot(d.transpose(), g_k)):
            return alpha
        else:
            alpha = beta * alpha
    return alpha


if __name__ == '__main__':
    x = np.array([[2, 1],
                  [2, 1],
                  [1, 1]])

    w = np.array([[0.5, 0.5]])
    y = np.array([[1],
                  [0]])
    # ex4a(x, y)
    #ex4b()
    ex4c()
