'''
package to store/process multi channel flim data
'''
#%%
import numpy as np
import time
import pickle

class FlimData:
    ''' class for storing/processing flim data '''
    DEFAULT = {}


    def __init__(self,dataCube=None,timeRange=None,**kwarg):
        ''' initialization of the parameters 
        dataCube ... numpy array with dimensions time,channel,y,x 
        timeRange ... first and last time [ns]
        '''
        self.dataCube = None  
        self.timeRange = None
        self.timeAxis = None
        self.processedData = None

        self.setData(dataCube,timeRange)

    def setData(self,dataCube,timeRange=None):
        ''' set signal and (time)'''
        self.dataCube = dataCube
        
        if timeRange is not None:
            self.timeRange = timeRange

        if self.timeRange is None and self.dataCube is not None:
            self.timeRange = np.array([0,self.dataCube.shape[0]-1])
        
        self.timeAxis = np.linspace(self.timeRange[0],self.timeRange[1],int(self.dataCube.shape[0]))
        self.processData()

    def processData(self, type='wide-field'):
        ''' process Image'''
        self.processedData = np.sum(self.dataCube,axis=1)

    def getImage(self,processed=False):
        ''' get the image'''
        if processed:
            return np.sum(self.processedData,axis=0)
        else:
            return np.sum(self.dataCube,axis=0)
    
    def getTimeAxis(self, processed=False):
        ''' get axis of time vector'''
        return self.timeAxis
    
    def getTimeHistogram(self, processed=False):
        ''' get time histogram'''
        if processed:
            return np.sum(self.processedData,axis=(-1,-2))
        else:
            return np.sum(self.dataCube,axis=(-1,-2))        



        
#%%

if __name__ == "__main__":
    pass
















# %%
