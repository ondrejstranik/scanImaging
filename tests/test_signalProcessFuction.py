''' tests '''

import pytest

import numpy as np
import matplotlib.pyplot as plt

@pytest.mark.GUI
def test_resetSignal():

    from scanImaging.algorithm.signalProcessFunction import resetSignal

    x = np.arange(100)
    y = 3*x + 5
    x0 = np.random.rand(len(x))>0.95

    yReset, bcg = resetSignal(y,x0)

    fig, ax = plt.subplots()

    ax.plot(x,y, 'r', label = 'signal')
    ax.plot(x[x0],y[x0], 'o', label= 'flag')
    ax.plot(x,yReset,'b', label = 'reset value')
    ax.plot(x,bcg,'g', label = 'bcg')
    ax.legend()
    plt.show()

@pytest.mark.GUI
def test_upperEdgeToCounter():
    from scanImaging.algorithm.signalProcessFunction import upperEdgeToCounter

    lastCounter = 3
    n = int(30)
    x = np.arange(n)
    y = (np.random.rand(n)> 0.5)*5    

    counter = upperEdgeToCounter(y,iniCounter=lastCounter)

    fig, ax = plt.subplots()
    ax.plot(x,y, color='red', label= 'y')
    ax.plot(x,counter, color='green', label= 'counter')
    ax.legend()
    plt.show()


