import matplotlib.pyplot as plt
import numpy as np

def plot_mse40():
    XY = np.genfromtxt('mse40.txt', delimiter=',')

    x = XY[:,0]
    y = XY[:,1]

    print x
    print y
    plt.ylabel('MSE')
    plt.xlabel('number of bags')
    plt.title('Testing MSE with MaxDepth set to 40')
    plt.plot(x,y)
    plt.show()

def plot_mse1():
    XY = np.genfromtxt('mse1.txt', delimiter=',')

    x = XY[:,0]
    y = XY[:,1]

    print x
    print y
    plt.ylabel('MSE')
    plt.xlabel('number of bags')
    plt.title('title 1')
    plt.plot(x,y)
    plt.show()

def plot_mse():
    XY = np.genfromtxt('mse.txt', delimiter=',')

    x = XY[:,0]
    y = XY[:,1]

    print x
    print y
    plt.ylabel('MSE')
    plt.xlabel('number of bags')
    plt.title('Generic Test Run with number of bags')
    plt.plot(x,y)
    plt.show()

def plot_compare():
    pass


plot_mse()
plot_mse1()
plot_mse40()