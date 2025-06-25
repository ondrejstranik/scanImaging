''' module with function to process signal '''
import numpy as np

def upperEdgeSignalToFlag(signal,threshold=0):
    ''' function returning flags, where function value went up over threshold
    input:
    signal ... 1D array
    threshold ... threshold value
    output:
    1D bool array with the events
    '''
    dy = np.diff(signal,prepend=0)
    flagY = dy>threshold
    return flagY

def flagToCounter(flag,iniCounter= 0):
    '''  function converting 1D array flag to 1D array with flag counts
    input:
    flag .. 1D array
    iniCounter .. staring value of the counter
    return:
    1D array with flag counts'''
    return iniCounter + np.cumsum(flag)

def upperEdgeToCounter(signal,threshold=0,iniCounter=0):
    ''' compound function of upperEdgeSignalToFlag and flagToCounter  '''
    return flagToCounter(
        upperEdgeSignalToFlag(signal,threshold=threshold),
        iniCounter=iniCounter)



def resetSignal(signal,flag):
    ''' function resetting signal value at given flags
    input:
    signal ... 1D array
    flag ... 1D bool array
    return:
    1D array with modified values
    1D array with background'''

    # no reset point
    if np.any(flag) == False:
        return signal
    
    s0= signal[flag]
    ds0 = np.diff(s0,prepend=0)

    s = np.zeros_like(signal)
    s[flag]= ds0
    bcg = np.cumsum(s)

    return signal-bcg, bcg

    






