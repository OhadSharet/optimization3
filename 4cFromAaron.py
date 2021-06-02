import os
from os.path import join
from scipy.sparse import diags
import numpy as np
from mlxtend.data import loadlocal_mnist
from itertools import tee
import math
import matplotlib.pyplot as plt
import main as myMain


def get_objective(x, c, w, sigmxtw='None'):
    if (sigmxtw == 'None'):
        sigmxtw = myMain.sigmoid(myMain.net_input(x, w))
    return -(c.transpose() @ np.log(sigmxtw) + (1 - c.transpose()) @ np.log(1 - sigmxtw)) / x.shape[0]


def compute_logit(x, y, w, compute_hessian=True):
    objective = myMain.cost(w, x, y)
    deriv = myMain.gradient(w, x, y)
    if (compute_hessian):
        hessian = myMain.hessian(w, x, y)
        return (objective, deriv, hessian)
    else:
        return (objective, deriv)  # +end_src

    # +name test 4a


x = np.array([[1, 2],
              [3, 4],
              [5, 6],
              [7, 8],
              [-1, 2]])
x = x.transpose()
y = np.array([[0, 0, 1, 0, 1]]).transpose()
w = np.array([[1, 2]]).transpose()
objective, deriv, hessian = compute_logit(x, y, w)
print(hessian)

cwd = os.getcwd()
input_path = cwd + '/MNIST'
training_images_filepath = join(input_path, 'train-images.idx3-ubyte')
training_labels_filepath = join(input_path, 'train-labels.idx1-ubyte')
test_images_filepath = join(input_path, 't10k-images.idx3-ubyte')
test_labels_filepath = join(input_path, 't10k-labels.idx1-ubyte')

xtrain, ytrain = loadlocal_mnist(images_path=training_images_filepath, labels_path=training_labels_filepath)
xtest, ytest = loadlocal_mnist(images_path=test_images_filepath, labels_path=test_labels_filepath)
xtrain = xtrain / np.max(xtrain)
xtest = xtest / np.max(xtest)


def filter_data_for_digits(digita, digitb):
    train_indeces = (i for i in range(xtrain.shape[0]) if ytrain[i] in [digita, digitb])
    ti1, ti2 = tee(train_indeces)
    xtr = np.array([xtrain[i] for i in ti1]).transpose()
    ytr = np.array([ytrain[i] for i in ti2])
    y_train = np.array([ytr]).transpose()
    test_indeces = (i for i in range(xtest.shape[0]) if ytest[i] in [digita, digitb])
    ti1, ti2 = tee(test_indeces)
    xte = np.array([xtest[i] for i in ti1]).transpose()
    yte = np.array([ytest[i] for i in ti2])
    y_test = np.array([yte]).transpose()
    return (xtr, y_train, xte, y_test)


def correct_frac(digita, digitb, x, y, w):
    # Returns percentage of correct classifications
    y = y.transpose()[0]
    x = x.transpose()
    correct_0 = len([i for i in range(len(x)) if (y[i] == digita) and (myMain.sigmoid(x[i] @ w) > 0.5)])
    correct_1 = len([i for i in range(len(x)) if (y[i] == digitb) and (myMain.sigmoid(x[i] @ w) <= 0.5)])
    correct = correct_0 + correct_1
    return float(correct) / len(x)


def get_alpha_armijo(x, y, w, d, deriv):
    # Function performs line sarch according to armijo, retrns alpha
    pow = 0
    alpha = 0.8
    baseobj = get_objective(x, y, w)

    while (True):
        newalpha = math.pow(alpha, pow)
        leftop = get_objective(x, y, w + newalpha * d)
        rightop = baseobj + 0.00001 * newalpha * (d.transpose() @ deriv)

        if leftop < rightop:
            return newalpha
        else:
            pow += 1


def train(digita, digitb, xtrain, xtest, ytrain, ytest, sd=True):
    w = np.zeros((xtrain.shape[0], 1))
    oldobj = 0
    trainobjs = []
    testobjs = []
    iter = 0
    flag = True
    init = True
    deriv = np.zeros(w.shape)
    d = np.zeros(w.shape)
    obj = 1

    # Iterate maximum 100 iterations
    while (iter < 100 and flag):
        w = np.clip(w, -1, 1)
        # On first time, don't run armijo
        if (not init):
            # Running armijo
            alpha = get_alpha_armijo(xtrain, ytrain, w, d, deriv)
        else:
            alpha = 1
            init = False

        w += alpha * d

        if (obj < 0.001):
            flag = False

        oldobj = obj
        iter += 1

        if (sd):
            obj, deriv = compute_logit(xtrain, ytrain, w, False)
            # Steepest descent
            d = -deriv

        else:
            obj, deriv, hes = compute_logit(xtrain, ytrain, w)
            # Calculating according to newton
            d = -np.linalg.inv(hes + np.identity(hes.shape[0]) * 0.01) @ deriv

        trainobjs.append(obj[0][0])
        testobjs.append(myMain.cost(w, xtest, ytest)[0][0])

    return (w, np.array(trainobjs), np.array(testobjs))


def train_digits(digita, digitb, sd=True):
    xtr, ytr, xte, yte = filter_data_for_digits(digita, digitb)
    w, trainhistory, testhistory = train(digita, digitb, xtr, xte, 1-ytr, 1-yte, sd)
    print(trainhistory)
    cfrac = correct_frac(digita, digitb, xte, yte, w)
    return (np.abs(trainhistory - trainhistory[-1]), np.abs(testhistory - testhistory[-1]), cfrac, testhistory[-1], w)


def main():
    sd01 = train_digits(0, 1, sd=True)
    print("PASS")
    nw01 = train_digits(0, 1, sd=False)
    print("PASS")
    sd89 = train_digits(8, 9, sd=True)
    print("PASS")
    nw89 = train_digits(8, 9, sd=False)
    print("PASS")
    fig, axs = plt.subplots(2, figsize=(9, 8))
    axs[0].plot(sd01[0], color='blue')
    axs[0].plot(nw01[0], color='red')
    axs[0].plot(sd89[0], color='green')
    axs[0].plot(nw89[0], color='black')
    axs[0].set_yscale('log')
    axs[0].set_ylabel('|f(w^k) - f(w*)|')
    axs[0].set_xlabel('Iterations')
    axs[1].plot(sd01[1], color='blue', label='0/1, SD')
    axs[1].plot(nw01[1], color='red', label='0/1, Newton')
    axs[1].plot(sd89[1], color='green', label='8/9, SD')
    axs[1].plot(nw89[1], color='black', label='8/9, Newton')
    axs[1].set_yscale('log')
    axs[1].set_ylabel('|f(w^k) - f(w*)|')
    axs[1].set_xlabel('Iterations')
    axs[0].set_title('Training data')
    axs[1].set_title('Test data')
    handles, labels = axs[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    fig.tight_layout(pad=5)
    fig.savefig('asscd.png')
    print('_Fraction of classifications made correctly on test data_\n')
    print('0/1, SD: ' + str(sd01[2]) + '\n')
    print('0/1, Newton: ' + str(nw01[2]) + '\n')
    print('8/9, SD: ' + str(sd89[2]) + '\n')
    print('8/9, Newton: ' + str(nw89[2]) + '\n')

    print('_Final objective value on test data_\n')
    print('0/1, SD: ' + str(sd01[3]) + '\n')
    print('0/1, Newton: ' + str(nw01[3]) + '\n')
    print('8/9, SD: ' + str(sd89[3]) + '\n')
    print('8/9, Newton: ' + str(nw89[3]) + '\n')
    print('[[./asscd.png]]')


if (__name__ == '__main__'):
    main()
