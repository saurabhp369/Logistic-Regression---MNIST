#=======================================================
# Author : Saurabh Palande
# Package : CMSC828C Project 2
# Module : Logistic Regression implementation
#=======================================================
from Utils.load_mnist import *
from Utils.PCA import *
from Utils.MDA import *
import matplotlib.pyplot as plt


def logistic_regression(x,y, n_epochs, Y, lr):
    _, L = x.shape
    N, M = y.shape
    loss = []
    acc = []
    
    # initialize theta
    theta = np.zeros((L,M))
    for i in range(n_epochs):
        print('Starting epoch: ', i)
        # calculate the posterior
        Phi = (np.exp(x.dot(theta))/np.sum(np.exp(x.dot(theta)), axis=1).reshape(N,1))
        y_pred = np.argmax(Phi, axis=1)
        print('train accuracy', np.mean(y_pred == Y)*100)
        acc.append(np.mean(y_pred == Y)*100)
        # calculate the loss
        epoch_loss = -1*np.sum(y*np.log(Phi))
        loss.append(epoch_loss)
        # alculate the gradient
        grad = np.matmul(x.T, Phi - y)
        # update theta
        theta = theta - lr*grad

    return theta,loss, acc


def main():

    # get the MNIST data
    x_train, x_test = get_data()
    y_train, y_test = get_labels()

    # one hot encoding of train and test labels
    train_labels = np.zeros((y_train.size, y_train.max() + 1))
    train_labels[np.arange(y_train.size), y_train] = 1

    test_labels = np.zeros((y_test.size, y_test.max() + 1))
    test_labels[np.arange(y_test.size), y_test] = 1

    n_epochs = int(input('Enter the number of epochs: '))
    choice = int(input('Enter the choice 1.PCA 2.MDA: '))
    if choice == 1:
        # Perform PCA on train data
        x_train_reduced, eig_vecs = PCA(x_train)
        x_test_reduced = np.real(np.matmul(x_test, eig_vecs))
        print('PCA done on train data')
        lr = 0.0000000001 # for PCA
    else:
        # Perform MDA on train data
        x_train_reduced, t_matrix = MDA(x_train, y_train)
        x_test_reduced = np.real((np.matmul(t_matrix.T, x_test.T)).T)
        print('MDA done on train data')
        lr = 0.1 # for MD
    # find theta 
    theta_final, loss, tr_acc = logistic_regression(x_train_reduced, train_labels, n_epochs, y_train,lr)
    # Perform PCA on test data
    Phi = (np.exp(x_test_reduced.dot(theta_final)))/np.sum(np.exp(x_test_reduced.dot(theta_final)), axis=0)
    y_pred = np.argmax(Phi, axis=1)
    print('Test accuracy', np.mean(y_pred == y_test)*100)

    # plot training loss and accuracy for epochs
    plt.subplot(1,2,1)
    plt.plot(loss)
    plt.xlabel('Epochs')
    plt.ylabel('Training Loss')
    plt.subplot(1,2,2)
    plt.plot(tr_acc)
    plt.xlabel('Epochs')
    plt.ylabel('Training Accuracy')
    ch = ['PCA', 'MDA']
    plt.suptitle('Logistic Regression'+ ch[choice-1])
    # plt.savefig('Logistic Regression'+ ch[choice-1]+'.png')
    plt.show()




if __name__ == '__main__':
    main()