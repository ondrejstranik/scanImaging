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
               'pixelTime': 1e-5, # in [s], time per Pixel
               'maxPhotonPerPixel': 10, # maximal number of event per pixel
               'macroTimeIncrement': 1e-7, # in [s] time to increase the macrotime counter per one
               'imageSize' : np.array([512,512]),
               'scanOffSet' : np.array([5,10]) # add (horizontal,vertical) lines
                # simulate returning path of the scanner 
               } 

    def __init__(self, name=DEFAULT['name'], **kwargs):
        ''' detector initialization'''
        super().__init__(name=name, **kwargs)

        # probe, which it will scan over. 
        self.virtualProbe = None

        # variable to generate virtual data
        self.pixelTime = self.DEFAULT['pixelTime'] # dwell time on one pixel
        self.signalTime = self.pixelTime/self.DEFAULT['maxPhotonPerPixel'] # smallest time between photons
        self.macroTimeIncrement = self.DEFAULT['macroTimeIncrement']
        self.macroTimeTotal = 0 # start counting with start of the measurement, in [self.signalTime units]
        self.scanPosition = 0 # current position of the scanner in linear dimension
        self.overflowFlag = False

        self.acquiring = False
        if self.acquiring is False:
            self.acquisitionStopTime = 0
            self.lastStackTime = 0

        # setting the virtual probe
        self.imageSize = self.DEFAULT['imageSize']
        # scan size take into acount the return of scanner and double it
        # the doubling assure the roll over the new image 
        #self.scanSize = (self.imageSize + self.DEFAULT['imageSize'])*([2,1])
        self.scanSize = (self.imageSize + self.DEFAULT['scanOffSet'])

        # maximal position of the scan (for one sample) in subpixels 
        self.maxScanPosition = int(np.prod(self.scanSize)*(self.DEFAULT['maxPhotonPerPixel']))

        #Probes
        # 1. constant with a thick line
        self.virtualProbe = np.zeros(self.imageSize) +0
        self.virtualProbe[:, int(self.imageSize[0]*0.4):int(self.imageSize[0]*0.6)] = 1

        #  add scan simulation return of the scan 
        self.virtualProbeExtra = np.zeros(self.scanSize)
        self.virtualProbeExtra[0:self.imageSize[0],0:self.imageSize[1]]= self.virtualProbe

        # double the virtual probe. it overcome the indexing issue with scan rolling over 
        self.virtualProbeExtra = np.vstack((self.virtualProbeExtra,self.virtualProbeExtra))

        # linearize the probe
        self.virtualProbeExtra = self.virtualProbeExtra.reshape(-1)
        
    def startAcquisition(self):
        '''start detector collecting data in a stack '''
        super().startAcquisition()
        self.macroTimeTotal = 0 # start recording the total time
        self.lastStackTime = time.time_ns() #self.acquisitionStartTime*1
        self.acquisitionStopTime = None        
        # set arbitrary start position of scanning
        # TODO: switched off for debugging
        #self.scanPosition = np.random.randint(int(self.maxScanPosition)) 

    def stopAcquisition(self):
        '''stop detector collecting data in a stack '''
        super().stopAcquisition()
        self.acquisitionStopTime = time.time_ns()

    def _calculateStack(self):
        ''' calculate the virtual stack '''

        currentTime = time.time_ns()
        
        if self.acquisitionStopTime is not None:
            currentTime = self.acquisitionStopTime

        if self.acquisitionStopTime == self.lastStackTime:
            virtualStack = None
        else:
            # number of sub pixels to generate
            nSubPixel = int((currentTime - self.lastStackTime)*1e-9/self.signalTime)
            newScanPosition = self.scanPosition + nSubPixel

            # photon events generation
            # generate the photons ... 1= photon was generated                
            scanRange = np.arange(self.scanPosition, newScanPosition)
            pixelIdx = (scanRange//int(self.pixelTime/self.signalTime)).astype(int)
            threshold = 0.5
            
            # detect overflow 
            if pixelIdx[-1]>2*(np.prod(self.scanSize)-1):
                self.overflowFlag = True
                np.clip(pixelIdx,0,2*(np.prod(self.scanSize)-1),out=pixelIdx)
                print(f'scanner overfLow!')


            # this probability calculation already contain the the poisson noise
            _virtualPhoton = (self.virtualProbeExtra[pixelIdx]*np.random.rand(len(pixelIdx))>threshold)*1

            # macro time + macroTimeFlag generation
            #print(f'time : {(currentTime - self.lastStackTime)}')
            #print(f'nSubPixel {nSubPixel}')
            tMacro = self.macroTimeTotal + np.arange(nSubPixel+1)*self.signalTime/self.macroTimeIncrement
            tMacroSaw = (tMacro%2**12)
            tMacroFlag = (tMacroSaw[1:] - tMacroSaw[0:-1])<0

            # update macroTimeTotal
            self.macroTimeTotal = tMacro[-1]
            #print(f'len(tMacroSaw): {len(tMacroSaw)}')
            
            # new line flag generation
            newLineFlag = scanRange%(self.scanSize[1]*self.DEFAULT['maxPhotonPerPixel'])==0
            # remove new line flag from rows below sample
            # this rows imitates return of the beam
            outerRows = np.logical_and(scanRange>np.prod(self.scanSize-self.DEFAULT['scanOffSet']*[1,0])*self.DEFAULT['maxPhotonPerPixel']-1,
                                        scanRange< self.maxScanPosition)
            #outerRows = scanRange< self.maxScanPosition
            #print(f'outerRows: {outerRows}')
            newLineFlag[outerRows]= False

            # update the position, correct for roll over
            if newScanPosition > self.maxScanPosition:
                self.scanPosition = newScanPosition%self.maxScanPosition
            else:              
                self.scanPosition = newScanPosition

            # generate the signal
            # TODO: remove the events without photons (but keep the flags)
            
            
            virtualStack = np.vstack([tMacroFlag,newLineFlag,tMacroSaw[1:],_virtualPhoton]).T


        self.lastStackTime = currentTime
        return virtualStack
        
    def getStack(self):
        ''' get data from the stack'''        
        #print(f'getStack from {self.DEFAULT["name"]}')
        self.stack = self._calculateStack()
        #self.stack = np.array([10,10,10])

        return self.stack

if __name__ == '__main__':

    pass





# %%