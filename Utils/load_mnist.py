import idx2numpy
import numpy as np

def get_data():
    file1 = 'MNIST/train-images-idx3-ubyte'
    file2 = 'MNIST/t10k-images-idx3-ubyte'
    x_train = idx2numpy.convert_from_file(file1)
    x_test = idx2numpy.convert_from_file(file2)

    return x_train.reshape(x_train.shape[0],-1), x_test.reshape(x_test.shape[0], -1)

def get_labels():
    file1 = 'MNIST/train-labels-idx1-ubyte'
    file2 = 'MNIST/t10k-labels-idx1-ubyte'
    y_train = idx2numpy.convert_from_file(file1)
    y_test = idx2numpy.convert_from_file(file2)
    
    return y_train, y_test