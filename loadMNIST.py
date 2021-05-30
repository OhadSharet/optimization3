#
# This is a sample Notebook to demonstrate how to read "MNIST Dataset"
#
import numpy as np
import struct
from array import array
from os.path import join
import os
import random
import matplotlib.pyplot as plt
import main as project_main


#
# MNIST Data Loader Class
#
class MnistDataloader(object):
    def __init__(self, training_images_filepath, training_labels_filepath,
                 test_images_filepath, test_labels_filepath):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath

        self.test_or_train = 0
        self.train_images_labels_0_1 = 0
        self.train_images_labels_8_9 = 0
        self.train_images_matrix_0_1 = 0
        self.train_images_matrix_8_9 = 0
        self.test_images_labels_0_1 = 0
        self.test_images_labels_8_9 = 0
        self.test_images_matrix_0_1 = 0
        self.test_images_matrix_8_9 = 0


    def read_images_labels(self, images_filepath, labels_filepath):
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())

        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())
        images = []
        size = 10000
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            #meanval = np.mean(img)
            #stdval = np.std(img)
            #img = (img - meanval) / (stdval + 0.1)
            images[i][:] = img
        #print(np.array(images[1][:]))
        #print(np.array([labels]).transpose())

        self.splitter(images, labels, size)
        return images, labels

    def splitter(self, images, labels, size):
        print(self.test_or_train)
        zeros_ones_counter = 0
        eights_nines_counter = 0
        index_1 = 0
        index_2 = 0
        if self.test_or_train == 0:
            self.test_or_train = 1
            for i in range(size):
                if labels[i] == 0 or labels[i] == 1:
                    zeros_ones_counter = zeros_ones_counter + 1
                if labels[i] == 8 or labels[i] == 9:
                    eights_nines_counter = eights_nines_counter + 1
            print(zeros_ones_counter)
            print(eights_nines_counter)
            self.train_images_labels_0_1 = []
            self.train_images_labels_8_9 = []
            self.train_images_matrix_0_1 = np.zeros((zeros_ones_counter, 28 * 28))
            self.train_images_matrix_8_9 = np.zeros((eights_nines_counter, 28 * 28))

            for i in range(size):
                if labels[i] == 0 or labels[i] == 1:
                    self.train_images_labels_0_1.append(labels[i])
                    image_i = np.array(images[i][:]).flatten()
                    self.train_images_matrix_0_1[index_1] = image_i
                    index_1 = index_1 + 1

                if labels[i] == 8 or labels[i] == 9:
                    self.train_images_labels_8_9.append(labels[i])
                    image_i = np.array(images[i][:]).flatten()
                    self.train_images_matrix_8_9[index_2] = image_i
                    index_2 = index_2 + 1
            self.train_images_matrix_0_1 = self.train_images_matrix_0_1.transpose()
            self.train_images_matrix_8_9 = self.train_images_matrix_8_9.transpose()
            self.train_images_labels_0_1 = np.array([self.train_images_labels_0_1]).transpose()
            self.train_images_labels_8_9 = np.array([self.train_images_labels_8_9]).transpose() % 8

        else:
            for i in range(size):
                if labels[i] == 0 or labels[i] == 1:
                    zeros_ones_counter = zeros_ones_counter + 1
                if labels[i] == 8 or labels[i] == 9:
                    eights_nines_counter = eights_nines_counter + 1
            print(zeros_ones_counter)
            print(eights_nines_counter)
            self.test_images_labels_0_1 = []
            self.test_images_labels_8_9 = []
            self.test_images_matrix_0_1 = np.zeros((zeros_ones_counter, 28 * 28))
            self.test_images_matrix_8_9 = np.zeros((eights_nines_counter, 28 * 28))

            for i in range(size):
                if labels[i] == 0 or labels[i] == 1:
                    self.test_images_labels_0_1.append(labels[i])
                    image_i = np.array(images[i][:]).flatten()
                    self.test_images_matrix_0_1[index_1] = image_i
                    index_1 = index_1 + 1

                if labels[i] == 8 or labels[i] == 9:
                    self.test_images_labels_8_9.append(labels[i])
                    image_i = np.array(images[i][:]).flatten()
                    self.test_images_matrix_8_9[index_2] = image_i
                    index_2 = index_2 + 1
            self.test_images_matrix_0_1 = self.test_images_matrix_0_1.transpose()
            self.test_images_matrix_8_9 = self.test_images_matrix_8_9.transpose()
            self.test_images_labels_0_1 = np.array([self.test_images_labels_0_1]).transpose()
            self.test_images_labels_8_9 = np.array([self.test_images_labels_8_9]).transpose() % 8

            # images_matrix = np.zeros((size, 28*28))
            # for i in range(size):
            #    image_i = np.array(images[i][:]).flatten()
            #    images_matrix[i] = image_i

            # return images_labels_0_1, images_labels_8_9, images_matrix_0_1, images_matrix_8_9

    def splitter_return(self):
        return self.train_images_matrix_0_1, self.train_images_labels_0_1, self.train_images_matrix_8_9, \
               self.train_images_labels_8_9, self.test_images_matrix_0_1, self.test_images_labels_0_1, \
               self.test_images_matrix_8_9, self.test_images_labels_8_9,



    def load_data(self):
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return (x_train, y_train), (x_test, y_test)

    #


# Verify Reading Dataset via MnistDataloader class
#
#
# Set file paths based on added MNIST Datasets
#
cwd = os.getcwd()
input_path = cwd + '/MNIST'
training_images_filepath = join(input_path, 'train-images.idx3-ubyte')
training_labels_filepath = join(input_path, 'train-labels.idx1-ubyte')
test_images_filepath = join(input_path, 't10k-images.idx3-ubyte')
test_labels_filepath = join(input_path, 't10k-labels.idx1-ubyte')


#
# Helper function to show a list of images with their relating titles
#
def show_images(images, title_texts):
    cols = 5
    rows = int(len(images) / cols) + 1
    plt.figure(figsize=(30, 20))
    index = 1
    for x in zip(images, title_texts):
        image = x[0]
        title_text = x[1]
        plt.subplot(rows, cols, index)
        plt.imshow(image, cmap=plt.cm.gray)
        if (title_text != ''):
            plt.title(title_text, fontsize=15);
        index += 1
    plt.show()


#
# Load MINST dataset
#
def loadMNIST_main():
    mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath,
                                       test_labels_filepath)
    (x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()
    #
    # Show some random training and test images
    #
    images_2_show = []
    titles_2_show = []
    for i in range(0, 10):
        r = random.randint(1, 10000)
        images_2_show.append(x_train[r])
        titles_2_show.append('training image [' + str(r) + '] = ' + str(y_train[r]))

    for i in range(0, 5):
        r = random.randint(1, 10000)
        images_2_show.append(x_test[r])
        titles_2_show.append('test image [' + str(r) + '] = ' + str(y_test[r]))

    show_images(images_2_show, titles_2_show)

    ### testing ###
    x1, y1, x2, y2, x3, y3, x4, y4 = mnist_dataloader.splitter_return()
    project_main.ex4c_test(x1, y1, x3, y3)
    project_main.ex4c_test(x2, y2, x4, y4)


if __name__ == "__main__":
    loadMNIST_main()
