'''
package to store/process multi channel flim data
'''
#%%
import numpy as np
import time
import pickle

class FlimData:
    ''' class for storing/processing flim data '''
    DEFAULT = {'timeRange': np.array([0,10]), # (time bin0 , time bin end)
               }

    def __init__(self,dataCube=None,timeRange=None,**kwarg):
        ''' initialization of the parameters 
        dataCube ... numpy array with dimensions time,channel,y,x 
        timeRange ... first and last time [ns]
        '''

        self.timeRange=timeRange if timeRange is not None else FlimData.DEFAULT['timeRange']
        self.dataCube = dataCube  
        self.timeAxis = None
        self.processedData = None

        self.setData(dataCube)

    def setData(self,dataCube,timeRange=None):
        ''' set signal and (time)'''
        self.dataCube = dataCube
        self.setTimeAxis(timeRange=timeRange)

        self.processData()


    def setTimeAxis(self,timeRange=None):
        ''' set time Axis'''
        if timeRange is not None: self.timeRange = timeRange
        if self.dataCube is not None:
            self.timeAxis = np.linspace(self.timeRange[0],self.timeRange[1],int(self.dataCube.shape[0]))

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

    def roughLifetimeFit(self, peak_offset_ns=0.3, fit_window_ns=None,
                         fit_background=True):
        ''' rough single-exponential lifetime check on the whole-image histogram.

        Sums the data over all pixels (and channels) and fits a single exponential
        past the IRF peak. Sanity check only - see algorithm.flimFit and project notes
        for the pixel-resolved Julia handoff.

        Returns the dict from flimFit.rough_single_exp_fit.
        '''
        from scanImaging.algorithm.flimFit import rough_single_exp_fit
        time_axis = self.getTimeAxis()
        histogram = self.getTimeHistogram(processed=True)
        # collapse any residual channel axis so a 1D histogram is fit
        histogram = np.asarray(histogram, dtype=float)
        while histogram.ndim > 1:
            histogram = histogram.sum(axis=-1)
        return rough_single_exp_fit(time_axis, histogram,
                                    peak_offset_ns=peak_offset_ns,
                                    fit_window_ns=fit_window_ns,
                                    fit_background=fit_background)



        
#%%

if __name__ == "__main__":
    pass
















# %%
