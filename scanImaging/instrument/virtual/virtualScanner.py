#%%
''' module for a simple virtual scanner having internal stack for the data'''

import numpy as np
import time

from viscope.instrument.base.baseADetector import BaseADetector
from viscope.virtualSystem.component.sample import Sample


class VirtualScanner(BaseADetector):
    ''' class to emulate scanner
    data are stored in a stack
    data are in the form: x,y, #number of photons
    data are scanned horizontally, single lined
    '''

    DEFAULT = {'name':'virtualScanner',
               'signalRate': 5e4, # Hz
               'imageSize' : np.array([512,512])} 

    def __init__(self, name=DEFAULT['name'], **kwargs):
        ''' detector initialization'''
        super().__init__(name=name, **kwargs)
        
        # variable to generate virtual data
        self.signalRate = self.DEFAULT['signalRate']
        self.scanPosition = 0 # current position of the scanner in linear dimension

        self.acquiring = False
        if self.acquiring is False:
            self.acquisitionStopTime = 0
            self.lastStackTime = 0

        # setting the virtual probe
        self.imageSize = self.DEFAULT['imageSize']
        
        #Probes
        '''
        # 1. constant
        self.virtualProbe = np.zeros(self.imageSize) +100 #  setup the probe
        '''
        # 2. astronaut
        _Sample = Sample()
        _Sample.setAstronaut(sampleSize=self.imageSize,photonRateMax=10)
        self.virtualProbe = _Sample.get()

        # linearise the probe
        self.virtualProbe = self.virtualProbe.reshape(-1)
        
        # set the x and y indexes
        grid = np.indices(self.imageSize)
        self.xIdx = grid[0].reshape(-1)
        self.yIdx = grid[1].reshape(-1)


    def startAcquisition(self):
        '''start detector collecting data in a stack '''
        super().startAcquisition()
        self.lastStackTime = time.time_ns() #self.acquisitionStartTime*1
        self.acquisitionStopTime = None        
        self.scanPosition = np.random.randint(int(np.prod(self.imageSize))) # set arbitrary start position of scanning

    def stopAcquisition(self):
        '''stop detector collecting data in a stack '''
        super().stopAcquisition()
        self.acquisitionStopTime = time.time_ns()

    def _calculateStack(self):
        ''' calculate the virtual stack '''

        try:
            currentTime = time.time_ns()
            
            if self.acquisitionStopTime is not None:
                currentTime = self.acquisitionStopTime

            if self.acquisitionStopTime == self.lastStackTime:
                virtualStack = None
            else:
                # number of pixels
                nPixel = int((currentTime - self.lastStackTime)*1e-9*self.signalRate)
                newScanPosition = self.scanPosition + nPixel

                _virtualPhoton = np.array([])
                _virtualXIdx = np.array([])
                _virtualYIdx = np.array([])
                # add rest of the image if scanner rolls over the end of the image
                while newScanPosition> np.prod(self.imageSize):
                    _virtualPhoton = np.append(_virtualPhoton,self.virtualProbe[self.scanPosition:-1])
                    _virtualXIdx = np.append(_virtualXIdx,self.xIdx[self.scanPosition:-1])
                    _virtualYIdx = np.append(_virtualYIdx,self.yIdx[self.scanPosition:-1])
                    self.scanPosition = 0
                    newScanPosition = newScanPosition - np.prod(self.imageSize)
                # add the rest
                _virtualPhoton = np.append(_virtualPhoton,self.virtualProbe[self.scanPosition:newScanPosition])
                _virtualXIdx = np.append(_virtualXIdx,self.xIdx[self.scanPosition:newScanPosition])            
                _virtualYIdx = np.append(_virtualYIdx,self.yIdx[self.scanPosition:newScanPosition])            
                
                self.scanPosition = newScanPosition

                # generate the signal
                virtualStack = np.vstack([_virtualXIdx,_virtualYIdx,np.random.poisson(_virtualPhoton)]).T


            self.lastStackTime = currentTime
            return virtualStack
        
        except Exception as error:
            # handle the exception
            print("An exception occurred:", error) # An exception occurred: division by zero            

    def getStack(self):
        ''' get data from the stack'''        
        #print(f'getStack from {self.DEFAULT["name"]}')
        self.stack = self._calculateStack()
        #self.stack = np.array([10,10,10])

        return self.stack

if __name__ == '__main__':

    pass





# %%