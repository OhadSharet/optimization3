import os
from os.path import join
import numpy as np
from mlxtend.data import loadlocal_mnist
from itertools import tee
import matplotlib.pyplot as plt
import main as myMain


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
    total_correct_digit_1 = len([i for i in range(len(x)) if (y[i] == digit_1) and (myMain.sigmoid(x[i] @ w) > 0.5)])
    total_correct_digit_2 = len([i for i in range(len(x)) if (y[i] == digit_2) and (myMain.sigmoid(x[i] @ w) <= 0.5)])
    total_correct_digits = total_correct_digit_1 + total_correct_digit_2
    return float(total_correct_digits) / len(x)


def fit_train_and_test(digit_1, digit_2, gradient_descent_indicator=True):
    # Trains the model from the training data, and tests the model from the testing data
    x_train, y_train, x_test, y_test = filter_data_for_digits(digit_1, digit_2)
    w = np.zeros((x_train.shape[0], 1))
    if gradient_descent_indicator:
        w, train_cost_history, test_cost_history = myMain.Gradient_Descent(w, x_train, 1-y_train, x_test, 1-y_test)
    else:
        w, train_cost_history, test_cost_history = myMain.Exact_Newton(w, x_train, 1-y_train, x_test, 1-y_test)
    accuracy_rate = accuracy(digit_1 % 8, digit_2 % 8, x_test, y_test, w)
    return (np.abs(train_cost_history - train_cost_history[-1]), np.abs(test_cost_history - test_cost_history[-1]),
            accuracy_rate, test_cost_history[-1], w)


def main():
    gd_0v1 = fit_train_and_test(0, 1, gradient_descent_indicator=True)
    print("PASS")
    en_0v1 = fit_train_and_test(0, 1, gradient_descent_indicator=False)
    print("PASS")
    gd_8v9 = fit_train_and_test(8, 9, gradient_descent_indicator=True)
    print("PASS")
    en_8v9 = fit_train_and_test(8, 9, gradient_descent_indicator=False)
    print("PASS")

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


if __name__ == '__main__':
    main()
