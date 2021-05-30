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


def Logistic_Regression(w, x, y):
    Cost_Fw = cost(w, x, y)
    Gradient_Fw = gradient(w, x, y)
    Hessian_Fw = hessian(w, x, y)

    # print("4a")
    print(Cost_Fw)
    print(Gradient_Fw)
    print(Hessian_Fw)
    return Cost_Fw, Gradient_Fw, Hessian_Fw


def net_input(x, w):
    return np.dot(x.transpose(), w)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def cost(w, x, y):
    m = x.shape[1]
    c1 = y
    c2 = 1-y
    probability = sigmoid(net_input(x, w))
    # Objective_Fw = c1.transpose() @ np.log(probability) + c2.transpose() @ np.log(1 - probability)
    return -(1 / m) * (c1.transpose() @ np.log(probability) + c2.transpose() @ np.log(1 - probability))


def gradient(w, x, y):
    m = x.shape[1]
    c1 = y
    probability = sigmoid(net_input(x, w))
    # Gradient_Fw = X @ (probability - c1)
    return (1 / m) * (x @ (probability - c1))


def hessian(w, x, y):
    m = y.shape[0]
    probability = sigmoid(net_input(x, w))
    D = np.asarray(probability.transpose())[0]
    D = np.diag(D)
    # Hessian_Fw = X @ D @ X.transpose()
    return (1 / m) * (x @ D @ x.transpose())


def ex4b():
    n = 20
    w = np.random.rand(n, 1)
    x = np.random.rand(n, 1)
    y = np.array([[0]])
    verification_test(w, x, y)

    # gradient_test_results.append(np.abs(Fx_ed - Fx - ((epsilon * d.transpose()) @ Fx_Gradient))[0][0])
    # jacobi_test_results.append(np.abs(Fx_ed - Fx - (epsilon * epsilon * (d.transpose() @ Fx_Hessian @ d)))[0][0])

def verification_test(w, x, y):
    d = np.random.rand(x.shape[0], 1)
    epsilon = 0.1
    n = 11
    F0 = function(w, x, y)
    G0 = grand(w, x, y)
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
        Fk = function(w, x_ed, y)
        Gk = grand(w, x_ed, y)
        F1 = F0 + (epsilon_k * np.dot(d.transpose(), G0))
        G1 = G0 + JacMV(w, x, y, epsilon_k * d)
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
    plt.xlabel("Iterations")
    plt.ylabel("Decrease Factor")
    plt.legend()
    plt.title("Gradient and Jacobian Tests")
    plt.show()


def function(w, x, y):
    #xtw = net_input(x, w)
    #probability = sigmoid(xtw)
    return cost(w, x, y)


def grand(w, x, y):
    #xtw = net_input(x, w)
    #probability = sigmoid(xtw)
    return gradient(w, x, y)


def JacMV(w, x, y, v):
    #xtw = net_input(x, w)
    #probability = sigmoid(xtw)
    return np.transpose(hessian(w, x, y)) @ v


# function is running from MNIST
def ex4c_test(x1, y1, x2, y2, train_or_test):
    w1 = 0.01 * np.ones((x1.shape[0], 1))
    w2 = 0.01 * np.ones((x2.shape[0], 1))
    y2 = y2
    #fitter(w1, x1, y1)
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
        plt.ylabel("|f(x+εd)-f(x)|")
        plt.legend()
        plt.title("Gradient Descent: 0 vs 1")
        plt.show()

        plt.figure()
        plt.plot(iterations, Objective_history2, label="Train")
        plt.plot(iterations, Objective_history4, label="Test")
        plt.xlabel("Iterations")
        plt.ylabel("|f(x+εd)-f(x)|")
        plt.legend()
        plt.legend()
        plt.title("Exact Newton: 0 vs 1")
        plt.show()

    else:
        plt.figure()
        plt.plot(iterations, Objective_history1, label="Train")
        plt.plot(iterations, Objective_history3, label="Test")
        plt.xlabel("Iterations")
        plt.ylabel("|f(x+εd)-f(x)|")
        plt.legend()
        plt.legend()
        plt.title("Gradient Descent: 8 vs 9")
        plt.show()

        plt.figure()
        plt.plot(iterations, Objective_history2, label="Train")
        plt.plot(iterations, Objective_history4, label="Test")
        plt.xlabel("Iterations")
        plt.ylabel("|f(x+εd)-f(x)|")
        plt.legend()
        plt.legend()
        plt.title("Exact Newton: 8 vs 9")
        plt.show()


def ex4c():
    n = 20
    x = np.random.rand(n, 1)
    w = np.random.rand(n, 1)
    y = np.array([[0]])
    Objective_history1 = Gradient_Descent(w, x, y)
    Objective_history2 = Exact_Newton(w, x, y)
    iterations = np.arange(0, 100, 1)

    plt.figure()
    plt.plot(iterations, Objective_history1, label="GD")
    plt.plot(iterations, Objective_history2, label="EN")
    plt.xlabel("Iterations")
    plt.ylabel("|f(x+εd)-f(x)|")
    plt.legend()
    plt.title("Gradient And Newton Tests")
    plt.show()


def fitter(w, x, y):
    weight = sci.fmin_tnc(function, x0=w, fprime=grand, args=(x, y))
    print(weight[0])


def Gradient_Descent(w, x, y, alpha0=0.25, iterations=100):
    Cost_history = np.zeros(iterations)
    f0 = cost(w, x, y)

    for i in range(iterations):
        f_k = cost(w, x, y)
        g_k = gradient(w, x, y)
        d = np.array(-g_k)
        alpha = Armijo_Linesearch(w, x, y, d, g_k, alpha=alpha0)
        w = np.clip(w + (alpha * d), -1, 1)
        Cost_history[i] = np.abs(f_k - f0)

    return Cost_history


def Exact_Newton(w, x, y, alpha0=1.0, iterations=100):
    Cost_history = np.zeros(iterations)
    f0 = cost(w, x, y)

    for i in range(iterations):
        f_k = cost(w, x, y)
        g_k = gradient(w, x, y)
        h_k = hessian(w, x, y)
        h_k_regulated = h_k + (0.5 * np.eye(x.shape[0]))
        d = -np.linalg.inv(h_k_regulated) @ g_k
        alpha = Armijo_Linesearch(w, x, y, d, g_k, alpha=alpha0)
        w = np.clip(w + (alpha * d), -1, 1)
        Cost_history[i] = np.abs(f_k - f0)

    return Cost_history


def Armijo_Linesearch(w, x, y, d, g_k, alpha=1.0, beta=0.5, c=1e-4):
    f_k = cost(w, x, y)
    for i in range(10):
        f_k_1 = cost(w, x + (alpha * d), y)
        if f_k_1 <= f_k - (alpha * c * np.dot(d.transpose(), g_k)):
            return alpha
        else:
            alpha = beta * alpha
    return alpha



def ex3f():
    theta = [1000000,0.001,110]
    ans = f(theta)
    y = usa_data_to_vector()
    print(ans)
    for i in range(1,3):
        jacobi = calc_f_jacobi(theta)
        F_grad = jacobi.transpose()@(f(theta)-y)
        theta -= F_grad*0.000000000000001
    print("---------------------------------------------------------")
    print (f(theta))

def f(theta):
    return np.array([fi(theta,xi) for xi in range(1,100)])

def fi(theta,xi):
    return theta[0]*math.exp(-theta[1]*((xi-theta[2])**2))

def calc_f_jacobi(theta):
    return np.array([_fi_gradient(theta, xi) for xi in range(1,100)])

def _fi_gradient(theta,xi):
    t = (xi - theta[2]) ** 2
    g1 = math.exp(-theta[1]*t)
    g2 = -theta[0]*t*math.exp(-theta[1]*t)
    g3 = 2*theta[0]*theta[1]*t*math.exp(-theta[1]*t**2)
    return [g1,g2,g3]




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
    #ex4a(x, y)
    #ex4b()
    ex4c()

