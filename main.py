import os
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from os.path import join
from itertools import tee
from scipy.sparse import spdiags
from mlxtend.data import loadlocal_mnist


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


def ex4c():
    gd_0v1 = fit_train_and_test(0, 1, gradient_descent_indicator=True)
    print("Gradient Descent 0 vs 1 - CALCULATION DONE")
    en_0v1 = fit_train_and_test(0, 1, gradient_descent_indicator=False)
    print("Exact Newton 0 vs 1 - CALCULATION DONE")
    gd_8v9 = fit_train_and_test(8, 9, gradient_descent_indicator=True)
    print("Gradient Descent 8 vs 9 - CALCULATION DONE")
    en_8v9 = fit_train_and_test(8, 9, gradient_descent_indicator=False)
    print("Exact Newton 8 vs 9 - CALCULATION DONE")

    fig1, axs1 = plt.subplots()
    fig2, axs2 = plt.subplots()
    fig3, axs3 = plt.subplots()
    fig4, axs4 = plt.subplots()
    axs1.plot(gd_0v1[0], label="Train")
    axs1.plot(gd_0v1[1], label="Test")
    axs2.plot(en_0v1[0], label="Train")
    axs2.plot(en_0v1[1], label="Test")
    axs3.plot(gd_8v9[0], label="Train")
    axs3.plot(gd_8v9[1], label="Test")
    axs4.plot(en_8v9[0], label="Train")
    axs4.plot(en_8v9[1], label="Test")

    axs1.set_yscale('log')
    axs2.set_yscale('log')
    axs3.set_yscale('log')
    axs4.set_yscale('log')

    axs1.set_ylabel('|f(w) - f(w*)|')
    axs2.set_ylabel('|f(w) - f(w*)|')
    axs3.set_ylabel('|f(w) - f(w*)|')
    axs4.set_ylabel('|f(w) - f(w*)|')

    axs1.set_xlabel('Iterations')
    axs2.set_xlabel('Iterations')
    axs3.set_xlabel('Iterations')
    axs4.set_xlabel('Iterations')

    axs1.set_title('Gradient Descent: 0 vs 1')
    axs2.set_title('Exact Newton: 0 vs 1')
    axs3.set_title('Gradient Descent: 8 vs 9')
    axs4.set_title('Exact Newton: 8 vs 9')

    axs1.legend()
    axs2.legend()
    axs3.legend()
    axs4.legend()
    plt.show()

    print('Success rate in each test data per method:')
    print('Gradient Descent: 0 vs 1 = ' + str(gd_0v1[2] * 100) + '%')
    print('Exact Newton: 0 vs 1 = ' + str(en_0v1[2] * 100) + '%')
    print('Gradient Descent: 8 vs 9 = ' + str(gd_8v9[2] * 100) + '%')
    print('Exact Newton: 8 vs 9 = ' + str(en_8v9[2] * 100) + '%' + '\n')

    print('Final objective value in each test data per method:')
    print('Gradient Descent: 0 vs 1 = ' + str(gd_0v1[3]))
    print('Exact Newton: 0 vs 1 = ' + str(en_0v1[3]))
    print('Gradient Descent: 8 vs 9 = ' + str(gd_8v9[3]))
    print('Exact Newton: 8 vs 9 = ' + str(en_8v9[3]))


cwd = os.getcwd()
input_path = cwd + '/MNIST'
training_images_filepath = join(input_path, 'train-images.idx3-ubyte')
training_labels_filepath = join(input_path, 'train-labels.idx1-ubyte')
test_images_filepath = join(input_path, 't10k-images.idx3-ubyte')
test_labels_filepath = join(input_path, 't10k-labels.idx1-ubyte')

xTrain, yTrain = loadlocal_mnist(images_path=training_images_filepath, labels_path=training_labels_filepath)
xTest, yTest = loadlocal_mnist(images_path=test_images_filepath, labels_path=test_labels_filepath)
xTrain = xTrain / np.max(xTrain)
xTest = xTest / np.max(xTest)


def filter_data_for_digits(digit_1, digit_2):
    # Filter train data
    train_indexes = (i for i in range(xTrain.shape[0]) if yTrain[i] in [digit_1, digit_2])
    train_index_1, train_index_2 = tee(train_indexes)
    x_train = np.array([xTrain[i] for i in train_index_1]).transpose()
    y_train_array = np.array([yTrain[i] for i in train_index_2])
    y_train = np.array([y_train_array]).transpose() % 8

    # Filter test data
    test_indexes = (i for i in range(xTest.shape[0]) if yTest[i] in [digit_1, digit_2])
    test_index_1, test_index_2 = tee(test_indexes)
    x_test = np.array([xTest[i] for i in test_index_1]).transpose()
    y_test_array = np.array([yTest[i] for i in test_index_2])
    y_test = np.array([y_test_array]).transpose() % 8
    return x_train, y_train, x_test, y_test


def accuracy(digit_1, digit_2, x, y, w):
    # Compute accuracy rate
    y = y.transpose()[0]
    x = x.transpose()
    total_correct_digit_1 = len([i for i in range(len(x)) if (y[i] == digit_1) and (sigmoid(x[i] @ w) > 0.5)])
    total_correct_digit_2 = len([i for i in range(len(x)) if (y[i] == digit_2) and (sigmoid(x[i] @ w) <= 0.5)])
    total_correct_digits = total_correct_digit_1 + total_correct_digit_2
    return float(total_correct_digits) / len(x)


def fit_train_and_test(digit_1, digit_2, gradient_descent_indicator=True):
    # Trains the model from the training data, and tests the model from the testing data
    x_train, y_train, x_test, y_test = filter_data_for_digits(digit_1, digit_2)
    w = np.zeros((x_train.shape[0], 1))
    if gradient_descent_indicator:
        w, train_cost_history, test_cost_history = Gradient_Descent(w, x_train, 1-y_train, x_test, 1-y_test)
    else:
        w, train_cost_history, test_cost_history = Exact_Newton(w, x_train, 1-y_train, x_test, 1-y_test)
    accuracy_rate = accuracy(digit_1 % 8, digit_2 % 8, x_test, y_test, w)
    return (np.abs(train_cost_history - train_cost_history[-1]), np.abs(test_cost_history - test_cost_history[-1]),
            accuracy_rate, test_cost_history[-1], w)


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
        if f_k < 1e-3:
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
        if f_k < 1e-3:
            break
        f_k, g_k, h_k = Logistic_Regression(w, x_train, y_train)
        h_k_regulated = h_k + (np.identity(h_k.shape[0]) * 1e-2)
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


def ex3e():
    sd_ans = ex3e_sd()
    gn_ans = ex3e_gn()
    plot_3(sd_ans, gn_ans, "ex3e")


def ex3f():
    sd_ans = ex3f_sd()
    gn_ans = ex3f_gn()
    plot_3(sd_ans, gn_ans, "ex3f")


def plot_3(sd_ans, gn_ans, title):
    plt.plot(sd_ans, label='SD')
    plt.plot(gn_ans, label='GN')
    plt.legend()
    plt.title(title)
    plt.show()


def ex3e_gn():
    gn_ans = []
    theta = [1000000, 0.001, 120]
    y = usa_data_to_vector()
    gn_iter = 10
    for i in range(1, gn_iter):
        jacobi = calc_f_jacobi(theta)
        F_grad = jacobi.transpose()@(f(theta)-y)
        JtJ = jacobi.transpose()@jacobi
        d_gn = np.linalg.inv(JtJ)@(-F_grad)  # the direction of gauss-Newton
        print(d_gn)
        alpha = Armijo_Linesearch_ex3(theta, y, d_gn, F_grad)
        print(alpha)
        theta += alpha*d_gn
        gn_ans.append((F(theta, y)))
    print("---------------------------------------------------------")
    final_gn_ans = [int(abs(x-gn_ans[-1])) for x in gn_ans]
    return final_gn_ans


def ex3e_sd():
    SD_ans = []
    theta = [1000000, 0.001, 120]  # initial theta
    y = usa_data_to_vector()
    sd_iter = 100
    for i in range(1, sd_iter):
        jacobi = calc_f_jacobi(theta)
        F_grad = jacobi.transpose()@(f(theta)-y)
        F_grad = F_grad/np.linalg.norm(F_grad)  # gradient was to bit so we normalized it
        d_SD = -F_grad  # direction = - gradient
        alpha = 0.00001  # step size
        theta += alpha*d_SD
        SD_ans.append((F(theta, y)))
    final_sd_ans = [int(abs(x-SD_ans[-1])) for x in SD_ans]
    return final_sd_ans


def ex3f_sd():
    SD_ans = []
    theta = [1, 1, 1]
    y = usa_data_to_vector()
    sd_iter = 100
    for i in range(1, sd_iter):
        jacobi = calc_f_jacobi_f(theta)
        F_grad = jacobi.transpose()@(f(theta)-y)
        F_grad = F_grad/np.linalg.norm(F_grad)  # gradient was to bit so we normalized it
        d_SD = -F_grad
        print(d_SD)
        alpha = 0.0001  # Armijo_Linesearch_ex3(theta, y, d_SD, F_grad)

        print("theta : %s" % theta)
        theta += alpha*d_SD
        SD_ans.append((F(theta, y)))
    print("---------------------------------------------------------")
    print(SD_ans)
    final_sd_ans = [int(abs(x-SD_ans[-1])) for x in SD_ans]
    return final_sd_ans


def ex3f_gn():
    gn_ans = []
    theta = [1, 1, 1]
    y = usa_data_to_vector()
    gn_iter = 10
    for i in range(1, gn_iter):
        jacobi = calc_f_jacobi_f(theta)
        F_grad = jacobi.transpose()@(f(theta)-y)
        JtJ = jacobi.transpose()@jacobi
        d_gn = np.linalg.inv(JtJ)@(-F_grad)
        alpha = Armijo_Linesearch_ex3(theta, y, d_gn, F_grad)
        theta += alpha*d_gn
        gn_ans.append((F(theta, y)))
    print("---------------------------------------------------------")
    final_gn_ans = [int(abs(x-gn_ans[-1])) for x in gn_ans]
    return final_gn_ans


def f(theta):
    return np.array([fi(theta, xi) for xi in range(1, 100)])


def F(theta, y):
    return 0.5*((np.linalg.norm(f(theta) - y, 2))**2)


def fi(theta, xi):
    return theta[0]*math.exp(-theta[1]*((xi-theta[2])**2))


def calc_f_jacobi(theta):
    return np.array([_fi_gradient(theta, xi) for xi in range(1, 100)])


def _fi_gradient(theta, xi):

    '''
    calc gradient for function in ex3e
    :param theta:
    :param xi: row number
    :return: return the gradient of the i-th row
    '''

    t = (xi - theta[2]) ** 2
    t3 = (xi - theta[2])
    g1 = math.exp(-theta[1]*t)
    g2 = -t*theta[0]*math.exp(-t*theta[1])
    g3 = 2*t3*theta[0]*theta[1]*math.exp(-theta[1]*t)
    return [g1, g2, g3]


def calc_f_jacobi_f(theta):
    return np.array([_fi_gradient_f(theta, xi) for xi in range(1, 100)])


def _fi_gradient_f(theta, xi):

    '''
    calc gradient for function in ex3f
    :param theta:
    :param xi: row number
    :return: return the gradient of the i-th row
    '''

    a0 = 1000000
    a1 = -0.001
    a2 = 110
    t = (xi-a2*theta[2]) ** 2
    t3 = (xi-a2*theta[2])
    g1 = a0*math.exp(a1*theta[1]*t)
    g2 = a1*a0*t*theta[0]*math.exp(a1*t*theta[1])
    g3 = 220000*t3*theta[0]*theta[1]*math.exp(a1*theta[1]*t)
    return [g1, g2, g3]


def Armijo_Linesearch_ex3(theta, y, d, g_k, alpha=1.0, beta=0.5, c=1e-4, max_iter=100):
    f_k = F(theta, y)
    for i in range(max_iter):
        f_k_1 = F(theta + (alpha * d), y)
        if f_k_1 <= f_k + (alpha * c * np.dot(d.transpose(), g_k)):
            return alpha
        else:
            alpha = beta * alpha
    return alpha


def usa_data_to_vector():
    with open("Covid-19-USA.txt") as file_in:
        lines = []
        for line in file_in:
            lines.append(int(line))
    return np.array(lines)


if __name__ == '__main__':
    # x = np.array([[2, 1],
    #               [2, 1],
    #               [1, 1]])
    # y = np.array([[1],
    #               [0]])
    # w = np.array([[0.5, 0.5]])
    # ex4a(x, y)
    # ex4b()
    # ex4c()
    ex3e()
    ex3f()
