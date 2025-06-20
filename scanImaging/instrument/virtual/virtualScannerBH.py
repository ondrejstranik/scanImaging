#%%
''' module for a virtual beckel&Hickel scanner having internal stack for the data'''

import numpy as np
import time

from viscope.instrument.base.baseADetector import BaseADetector
from viscope.virtualSystem.component.sample import Sample


class VirtualBHScanner(BaseADetector):
    ''' class to emulate B&H scanner
    data are stored in a stack
    data are in the form (2 bytes): 
    bits 31-28 flags
    b31 ... =1 no photon
    b30 ... =1 macrotime overflow
    b29 ... =1 stack overflow. Data loss. Dismiss previous data
    b28 ... =1 trigger for new line
    b16-27 ... =microtime.  12 bits resolution. photon delay = 2**12 - microtime
    b12-15 ... = channel. 4 bits resolution. total 16 channels
    b0-11 ... =macrotime. 12 bit resolution
    
    data are scanned horizontally, single lined (this is given by a separate scanner)




    '''

    DEFAULT = {'name':'virtualBHScanner',
               'pixelTime': 1e-6, # in [s]
               'pixelToSignalTime': 10, # pixelToSignalTime has to be integer
               'macroToPixelTime': 100,
               'imageSize' : np.array([512,512]),
               'scanOffSet' : np.array([5,10]) # add Pixel and lines
                #to simulate return of the scanner (#lines,#horizontal pixel)
               } 

    def __init__(self, name=DEFAULT['name'], **kwargs):
        ''' detector initialization'''
        super().__init__(name=name, **kwargs)
        
        # variable to generate virtual data
        self.pixelTime = self.DEFAULT['pixelTime']
        self.signalTime = self.pixelTime/self.DEFAULT['pixelToSignalTime']
        self.macroTime = self.pixelTime*self.DEFAULT['macroToPixelTime']
        self.scanPosition = 0 # current position of the scanner in linear dimension

        self.acquiring = False
        if self.acquiring is False:
            self.acquisitionStopTime = 0
            self.lastStackTime = 0

        # setting the virtual probe
        self.imageSize = self.DEFAULT['imageSize']
        self.scanSize = self.imageSize + self.DEFAULT['imageSize']

        # maximal position of the scan in subpixels 
        self.maxScanPosition = int(np.prod(self.scanSize)*(self.DEFAULT['pixelToSamplingTime']))

        #Probes
        # 1. constant with a thick line
        _virtualProbe = np.zeros(self.imageSize) +100
        _virtualProbe[:, int(self.imageSize[0]*0.4):int(self.imageSize[0]*0.6)] = 200 

        #  add scan simulation edges
        self.virtualProbe = np.zeros(self.scanSize)
        self.virtualProbe[0:self.imageSize[0],0:self.imageSize[1]]= _virtualProbe

        # linearise the probe
        self.virtualProbe = self.virtualProbe.reshape(-1)
        
    def startAcquisition(self):
        '''start detector collecting data in a stack '''
        super().startAcquisition()
        self.lastStackTime = time.time_ns() #self.acquisitionStartTime*1
        self.acquisitionStopTime = None        
        # set arbitrary start position of scanning
        self.scanPosition = np.random.randint(int(self.maxScanPosition)) 

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
                # number of sub pixels to generate
                nSubPixel = int((currentTime - self.lastStackTime)*1e-9/self.signalTime)
                newScanPosition = self.scanPosition + nSubPixel

                _virtualPhoton = np.array([])
                _virtualXIdx = np.array([])
                _virtualYIdx = np.array([])
                
                # TODO: implement the roll over
                if newScanPosition > self.maxScanPosition:
                    print('scan roll over: not implemented yet')
                    print('setting newScanPosition = self.maxScanPosition')
                    newScanPosition = self.maxScanPosition
 
                # TODO: not finised, continue
                # generate the photons ... 1= photon was generated                
                scanRange = np.arange(self.scanPosition, newScanPosition)
                pixelIdx = int(scanRange//int(self.pixelTime/self.signalTime))
                threshold = 0.5
                _virtualPhoton = (self.virtualProbe[pixelIdx]*np.random.rand(len(pixelIdx))>threshold)*1



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