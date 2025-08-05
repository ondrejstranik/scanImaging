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


class BHScannerProcessor(BaseProcessor):
    ''' class to collect data from virtual scanner'''
    DEFAULT = {'name': 'ScannerProcessor',
                'pixelTime': 90, # 
                'newPageTimeFlag': 5 # threshold for new page in the macroTime 
                }

    def __init__(self, name=None, **kwargs):
        ''' initialisation '''

        if name== None: name= BHScannerProcessor.DEFAULT['name']
        super().__init__(name=name, **kwargs)
        
        # asynchronous Detector
        self.scanner = None

        # parameters for calculation
        self.pixelTime = self.DEFAULT['pixelTime'] # dwell time on one pixel

        # data
        self.rawImage = None
        self.dataCube = None
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
        ''' reset value of all counting, indexing parameter. it is called when new scan start'''

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

    def disconnect(self):
        super().disconnect()
        #self.flagToProcess.set() 


    def setParameter(self,name, value):
        ''' set parameter of the spectral camera'''
        super().setParameter(name,value)

        if name== 'scanner':
            self.scanner = value
            self.flagToProcess = self.scanner.flagLoop
            self.rawImage = np.zeros(self.scanner.imageSize)
            # TODO: set properly the dimensions, type
            self.dataCube = np.zeros((10,3,*self.scanner.imageSize), dtype=int)

    def getParameter(self,name):
        ''' get parameter of the camera '''
        _value = super().getParameter(name)
        if _value is not None: return _value        

        if name== 'scanner':
            return self.aDetector

    def processData(self):
        ''' process newly arrived data '''

        #print(f"processing data from {self.DEFAULT['name']}")

        stack = self.scanner.getStack()
        # calculate total macroTime
        self.macroTime = (self.lastMacroTime
                            + stack[:,2] -self.lastMacroSawValue
                            + np.cumsum(stack[:,0]*2**12)
                        )
        self.lastMacroSawValue = stack[-1,2]

        # reset macroTime on each new line
        self.macroTime,_ = resetSignal(self.macroTime, stack[:,1].astype(bool))
        self.lastMacroTime = self.macroTime[-1]

        # calculate x position
        self.xIdx = (self.macroTime//self.pixelTime).astype(int)

        # clip the x index if out of image x range
        #np.clip(self.xIdx,0,self.rawImage.shape[1]-1,out=self.xIdx)

        # calculate y position
        self.yIdx = self.lastYIdx + np.cumsum(stack[:,1]).astype(int)
  
        # calculate page position
        returnSignal = (self.macroTime > 
                        self.pixelTime*self.rawImage.shape[1]*self.DEFAULT['newPageTimeFlag'])
        newPageFlag = upperEdgeSignalToFlag(returnSignal,0)

        self.pageIdx = flagToCounter(newPageFlag,iniCounter=self.lastPageIdx)

        #self.yIdx = self.yIdx - (self.pageIdx-self.lastPageIdx)*self.rawImage.shape[0]
        
        self.yIdx, _ = resetSignal(self.yIdx,newPageFlag, resetValue = -1)
        
        recordingPageIdx = self.lastPageIdx
        
        self.lastPageIdx = self.pageIdx[-1]
        self.lastYIdx = self.yIdx[-1]

        # TODO: take into account if YIdx was full
        # check if full image is recorded
        if np.any(returnSignal):
            self.flagFullImage = True
            print(f'new page flag generated: {np.sum(newPageFlag)}')
            print(f'new page index {np.max(self.pageIdx)}')


        # remove flags from data
        #print(f'stack 0 \n {stack[:,0]==0}')
        arrivedPhotons = ((stack[:,0]==0) & (stack[:,1]==0))
        self.yIdx = self.yIdx[arrivedPhotons]
        self.xIdx = self.xIdx[arrivedPhotons]
        self.pageIdx = self.pageIdx[arrivedPhotons]

        #remove photons which are outside image size
        insideImage = ((self.yIdx >= 0) & (self.yIdx <= self.rawImage.shape[1]-1)
                        &(self.xIdx >= 0) & (self.xIdx <= self.rawImage.shape[0]-1))
        self.yIdx = self.yIdx[insideImage]
        self.xIdx = self.xIdx[insideImage]       


        # add the photons to the image
        # continuos viewing
        _rawImage = 0*self.rawImage
        np.add.at(_rawImage,(self.yIdx,self.xIdx),1)
        self.rawImage[_rawImage>0] =0
        self.rawImage = self.rawImage + _rawImage

        # add photons to the whole dataCube
        # TODO: add proper channel and time
        _time = np.random.randint(0,9,len(self.yIdx))
        _channel = np.random.randint(0,3,len(self.yIdx))
        if recordingPageIdx==self.lastPageIdx:
            np.add.at(self.dataCube,(_time,_channel,self.yIdx,self.xIdx),1)
        else:
            # full image recorded
            if np.any(self.yIdx== self.scanner.imageSize[0]-1):
                _idx = self.pageIdx==recordingPageIdx
                np.add.at(self.dataCube,(_time,_channel,self.yIdx,self.xIdx),1)
                self.flagFullImage = True
                print('full image recorded')
            else: # not full image was recorded
                _idx = self.pageIdx==self.lastPageIdx
                self.dataCube = 0*self.dataCube
                np.add.at(self.dataCube,(_time,_channel,self.yIdx,self.xIdx),1)





#%%

if __name__ == '__main__':
    pass
