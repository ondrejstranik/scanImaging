"""
module to process data from scanner

Created on Mon Nov 15 12:08:51 2021

@author: ostranik
"""
#%%

import os
import time
import numpy as np
from viscope.instrument.base.baseProcessor import BaseProcessor
from scanImaging.algorithm.signalProcessFunction import resetSignal, flagToCounter, upperEdgeSignalToFlag


class ScannerBHProcessor(BaseProcessor):
    ''' class to collect data from virtual scanner'''
    DEFAULT = {'name': 'ScannerProcessor',
                'pixelTime': 105, # 
                'newPageTimeFlag': 3 # threshold for new page in the macroTime 
                }

    def __init__(self, name=None, **kwargs):
        ''' initialisation '''

        if name== None: name= ScannerBHProcessor.DEFAULT['name']
        super().__init__(name=name, **kwargs)
        
        # asynchronous Detector
        self.scanner = None

        # parameters for calculation
        self.pixelTime = self.DEFAULT['pixelTime'] # dwell time on one pixel

        # data
        self.rawImage = None
        self.xIdx = 0 # quick axis
        self.yIdx = 0 # slow axis
        self.lastYIdx = -1 # at the start of scanning the y-flag is given
        self.pageIdx = 0
        self.lastPageIdx = 0
        self.macroTime = 0
        self.lastMacroSawValue = 0
        self.lastMacroTime = 0
        self.flagFullImage = False # indicate, when whole image is recorded


    def resetCounter(self):
        ''' reset value of all counting, indxing parameter. it is called when new scan start'''

        self.rawImage = 0*self.rawImage
        self.xIdx = 0 # quick axis
        self.yIdx = 0 # slow axis
        self.lastYIdx = -1 # at the start of scanning the y-flag is given
        self.pageIdx = 0
        self.lastPageIdx = 0
        self.macroTime = 0
        self.lastMacroSawValue = 0
        self.lastMacroTime = 0
        self.flagFullImage = False # indicate, when whole image is recorded

    def connect(self,scanner=None):
        ''' connect data processor with aDetector'''
        super().connect()
        if scanner is not None: self.setParameter('scanner',scanner)

    def setParameter(self,name, value):
        ''' set parameter of the spectral camera'''
        super().setParameter(name,value)

        if name== 'scanner':
            self.scanner = value
            self.flagToProcess = self.scanner.flagLoop
            self.rawImage = np.zeros(self.scanner.imageSize)

    def getParameter(self,name):
        ''' get parameter of the camera '''
        _value = super().getParameter(name)
        if _value is not None: return _value        

        if name== 'scanner':
            return self.aDetector

    def processData(self):
        ''' process newly arrived data '''

        #print(f"processing data from {self.DEFAULT['name']}")

        # calculate total macroTime
        self.macroTime = (self.lastMacroTime
                            + self.scanner.stack[:,2] -self.lastMacroSawValue
                            + np.cumsum(self.scanner.stack[:,0]*2**12)
                        )
        self.lastMacroSawValue = self.scanner.stack[-1,2]

        # reset macroTime on each new line
        self.macroTime,_ = resetSignal(self.macroTime, self.scanner.stack[:,1].astype(bool))
        self.lastMacroTime = self.macroTime[-1]

        # calculate x position
        self.xIdx = (self.macroTime//self.pixelTime).astype(int)

        # clip the x index if out of image x range
        #np.clip(self.xIdx,0,self.rawImage.shape[1]-1,out=self.xIdx)

        # calculate y position
        self.yIdx = self.lastYIdx + np.cumsum(self.scanner.stack[:,1]).astype(int)
  
        # calculate page position
        returnSignal = (self.macroTime > 
                        self.pixelTime*self.rawImage.shape[1]*self.DEFAULT['newPageTimeFlag'])
        newPageFlag = upperEdgeSignalToFlag(returnSignal,0)

        self.pageIdx = flagToCounter(newPageFlag,iniCounter=self.lastPageIdx)

        #self.yIdx = self.yIdx - (self.pageIdx-self.lastPageIdx)*self.rawImage.shape[0]
        
        self.yIdx, _ = resetSignal(self.yIdx,newPageFlag, resetValue = -1)
        
        self.lastPageIdx = self.pageIdx[-1]
        self.lastYIdx = self.yIdx[-1]

        # TODO: take into account if YIdx was full
        # check if full image is recorded
        if np.any(returnSignal):
            self.flagFullImage = True
            print('new page flag generated')

        # TODO: temporary : not in real data
        # remove the empty photons
        arrivedPhotons = self.scanner.stack[:,3]==1
        self.yIdx = self.yIdx[arrivedPhotons]
        self.xIdx = self.xIdx[arrivedPhotons]

        #remove photons which are outside image size
        insideImage = ((self.yIdx >= 0) & (self.yIdx <= self.rawImage.shape[1]-1)
                        &(self.xIdx >= 0) & (self.xIdx <= self.rawImage.shape[0]-1))
        self.yIdx = self.yIdx[insideImage]
        self.xIdx = self.xIdx[insideImage]       


        # add the photons to the image
        np.add.at(self.rawImage,(self.yIdx,self.xIdx),1)



#%%

if __name__ == '__main__':
    pass
