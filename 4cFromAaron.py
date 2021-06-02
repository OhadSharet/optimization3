import os
from os.path import join
from scipy.sparse import diags
import numpy as np
from mlxtend.data import loadlocal_mnist
from itertools import tee
import math
import matplotlib.pyplot as plt
import main as myMain


#def compute_logit(x, y, w, compute_hessian=True):
#    objective = myMain.cost(w, x, y)
#    deriv = myMain.gradient(w, x, y)
#    if (compute_hessian):
#        hessian = myMain.hessian(w, x, y)
#        return (objective, deriv, hessian)
#    else:
#        return (objective, deriv)  # +end_src

    # +name test 4a


x = np.array([[1, 2],
              [3, 4],
              [5, 6],
              [7, 8],
              [-1, 2]])
x = x.transpose()
y = np.array([[0, 0, 1, 0, 1]]).transpose()
w = np.array([[1, 2]]).transpose()
objective, deriv, hessian = myMain.Logistic_Regression(w, x, y)
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
    y_train = np.array([ytr]).transpose() % 8
    test_indeces = (i for i in range(xtest.shape[0]) if ytest[i] in [digita, digitb])
    ti1, ti2 = tee(test_indeces)
    xte = np.array([xtest[i] for i in ti1]).transpose()
    yte = np.array([ytest[i] for i in ti2])
    y_test = np.array([yte]).transpose() % 8
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
    baseobj = myMain.cost(w, x, y)

    while (True):
        newalpha = math.pow(alpha, pow)
        leftop = myMain.cost(w + newalpha * d, x, y)
        rightop = baseobj + 0.00001 * newalpha * (d.transpose() @ deriv)

        if leftop < rightop:
            return newalpha
        else:
            pow += 1


def train(xtrain, xtest, ytrain, ytest, sd=True):
    w = np.zeros((xtrain.shape[0], 1))
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
            #alpha = get_alpha_armijo(xtrain, ytrain, w, d, deriv)
            alpha = myMain.Armijo_Linesearch(w, xtrain, ytrain, d, deriv)
        else:
            alpha = 1
            init = False

        w += alpha * d

        if (obj < 0.001):
            flag = False

        iter += 1

        if (sd):
            obj, deriv = myMain.Logistic_Regression(w, xtrain, ytrain, hessian_indicator=False)
            # Steepest descent
            d = -deriv

        else:
            obj, deriv, hes = myMain.Logistic_Regression(w, xtrain, ytrain)
            # Calculating according to newton
            d = -np.linalg.inv(hes + np.identity(hes.shape[0]) * 0.01) @ deriv

        trainobjs.append(obj[0][0])
        testobjs.append(myMain.cost(w, xtest, ytest)[0][0])

    return (w, np.array(trainobjs), np.array(testobjs))


def train_digits(digita, digitb, sd=True):
    xtr, ytr, xte, yte = filter_data_for_digits(digita, digitb)
    w = np.zeros((xtr.shape[0], 1))
    if sd:
        w, trainhistory, testhistory = myMain.Gradient_Descent(w, xtr, 1-ytr, xte, 1-yte)
    else:
        w, trainhistory, testhistory = myMain.Exact_Newton(w, xtr, 1-ytr, xte, 1-yte)
    #w, trainhistory, testhistory = train(xtr, xte, 1-ytr, 1-yte, sd)
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
    fig1, axs1 = plt.subplots(2, figsize=(9, 8))
    axs1[0].plot(sd01[0], color='red')
    axs1[0].plot(sd01[1], color='blue', label='0/1, SD')
    axs1[1].plot(nw01[0], color='red')
    axs1[1].plot(nw01[1], color='blue', label='0/1, Newton')
    fig2, axs2 = plt.subplots(2, figsize=(9, 8))
    axs2[0].plot(sd89[0], color='red')
    axs2[0].plot(sd89[1], color='blue', label='8/9, SD')
    axs2[1].plot(nw89[0], color='red')
    axs2[1].plot(nw89[1], color='blue', label='8/9, Newton')
    axs1[0].set_yscale('log')
    axs1[0].set_ylabel('|f(w^k) - f(w*)|')
    axs1[0].set_xlabel('Iterations')
    axs1[1].set_yscale('log')
    axs1[1].set_ylabel('|f(w^k) - f(w*)|')
    axs1[1].set_xlabel('Iterations')
    axs2[0].set_yscale('log')
    axs2[0].set_ylabel('|f(w^k) - f(w*)|')
    axs2[0].set_xlabel('Iterations')
    axs2[1].set_yscale('log')
    axs2[1].set_ylabel('|f(w^k) - f(w*)|')
    axs2[1].set_xlabel('Iterations')

    #axs[0].set_title('Training data')
    #axs[1].set_title('Test data')


    handles, labels = axs1[1].get_legend_handles_labels()
    fig1.legend(handles, labels, loc='upper right')
    fig1.tight_layout(pad=5)
    fig1.savefig('asscd1.png')

    handles, labels = axs2[1].get_legend_handles_labels()
    fig2.legend(handles, labels, loc='upper right')
    fig2.tight_layout(pad=5)
    fig2.savefig('asscd2.png')


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
