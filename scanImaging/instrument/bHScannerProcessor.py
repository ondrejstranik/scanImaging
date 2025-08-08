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
                'pixelTime': 90*445/512, # dwell time on one pixel
                'newPageTimeFlag': 5, # threshold for new page in the macroTime
                'numberOfAccumulation': 3, # number of images to accumulate
                'generateDataCube': False, # if false only overview image is generated,
                'microTimeBin' : 2**5 # resolution of the microTime Histogram
                }

    def __init__(self, name=None, **kwargs):
        ''' initialisation '''

        if name== None: name= BHScannerProcessor.DEFAULT['name']
        super().__init__(name=name, **kwargs)
        
        # asynchronous Detector
        self.scanner = None

        # for debugging
        self.stackSize = 0

        # parameters for calculation
        self.pixelTime = self.DEFAULT['pixelTime'] # dwell time on one pixel
        self.numberOfAccumulation = self.DEFAULT['numberOfAccumulation']
        self.generateDataCube = self.DEFAULT['generateDataCube'] # if false only overview image is generated
        self.microTimeBin = self.DEFAULT['microTimeBin']

        # data
        self.rawImage = None  # 2D image - preview currently acquiring
        self.dataCube = None # 4D image - currently acquiring
        self.dataCubeFinished = None # 4D image - acquired full image
        self.accumulationIdx = 0 # index of accumulation
        self.xIdx = 0 # quick axis
        self.yIdx = 0 # slow axis
        self.lastYIdx = -1 # at the start of scanning the y-flag is given
        self.maxYIdx = -1 # maximal y-line scanned. reset to -1 after new page
        self.pageIdx = 0
        self.lastPageIdx = 0
        self.recordingPageIdx = 0
        self.macroTime = 0
        self.lastMacroSawValue = 0
        self.lastMacroTime = 0
        self.microTime = 0
        self.channel = 0
        self.flagFullImage = False # indicate, when whole image is recorded
        self.flagFullAccumulation = False # indicate, when accumulated image is recorded


    def resetCounter(self):
        ''' reset value of all counting, indexing parameter. it is called when new scan start'''

        self.rawImage = 0*self.rawImage
        self.dataCube = 0*self.dataCube
        self.dataCubeFinished = 0*self.dataCubeFinished
        self.accumulationIdx = 0
        self.xIdx = 0 # quick axis
        self.yIdx = 0 # slow axis
        self.lastYIdx = -1 # at the start of scanning the y-flag is given
        self.maxYIdx = -1 # maximal y-line scanned. reset to -1 after new page        
        self.pageIdx = 0
        self.lastPageIdx = 0
        self.recordingPageIdx = 0
        self.macroTime = 0
        self.lastMacroSawValue = 0
        self.lastMacroTime = 0
        self.microTime = 0
        self.channel = 0
        self.flagFullImage = False # indicate, when whole image is recorded
        self.flagFullAccumulation = False # indicate, when accumulated image is recorded

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
            self.dataCube = np.zeros((self.microTimeBin,self.scanner.numberOfChannel,
                                      *self.scanner.imageSize), dtype=int)
            self.dataCubeFinished = np.zeros((self.microTimeBin,self.scanner.numberOfChannel,
                                              *self.scanner.imageSize), dtype=int)

    def getParameter(self,name):
        ''' get parameter of the camera '''
        _value = super().getParameter(name)
        if _value is not None: return _value        

        if name== 'scanner':
            return self.scanner

    def processData(self):
        ''' process newly arrived data '''

        #print(f"processing data from {self.DEFAULT['name']}")

        #start = time.time()


        stack = self.scanner.getStack()
        self.stackSize = stack.shape[0]

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
        
        self.lastPageIdx = self.pageIdx[-1]
        self.lastYIdx = self.yIdx[-1]
        self.maxYIdx  = np.max((np.max(self.yIdx),self.maxYIdx))

        # get channel
        self.channel = stack[:,3].astype('int')

        # get microTime and reduce resolution
        self.microTime = (stack[:,4] * self.microTimeBin / self.scanner.timeSize).astype('int')

        # info for debugging
        if np.any(returnSignal):
            print(f'new page flag generated: {np.sum(newPageFlag)}')
            print(f'max page index {np.max(self.pageIdx)}')
            print(f'last page index {self.lastPageIdx}')
        allEventYIdx = np.copy(self.yIdx)


        # remove flags from data
        #print(f'stack 0 \n {stack[:,0]==0}')
        arrivedPhotons = ((stack[:,0]==0) & (stack[:,1]==0))
        self.yIdx = self.yIdx[arrivedPhotons]
        self.xIdx = self.xIdx[arrivedPhotons]
        self.pageIdx = self.pageIdx[arrivedPhotons]
        self.microTime = self.microTime[arrivedPhotons]
        self.channel = self.channel[arrivedPhotons]

        #remove photons which are outside image size
        insideImage = ((self.yIdx >= 0) & (self.yIdx <= self.rawImage.shape[0]-1)
                        &(self.xIdx >= 0) & (self.xIdx <= self.rawImage.shape[1]-1))
        self.yIdx = self.yIdx[insideImage]
        self.xIdx = self.xIdx[insideImage]
        self.pageIdx = self.pageIdx[insideImage]       
        self.microTime = self.microTime[insideImage]
        self.channel = self.channel[insideImage]

        # add the photons to the image
        # continuos viewing
        _rawImage = 0*self.rawImage
        np.add.at(_rawImage,(self.yIdx,self.xIdx),1)
        self.rawImage[_rawImage>0] =0
        self.rawImage = self.rawImage + _rawImage

        
        if not self.generateDataCube:
            return

        # add photons to the whole dataCube
        # TODO: add proper channel and time
        #self.microTime = np.random.randint(0,10,len(self.yIdx))
        #self.channel = np.random.randint(0,3,len(self.yIdx))
        
        if self.recordingPageIdx>=self.lastPageIdx:
            np.add.at(self.dataCube,(self.microTime,self.channel,self.yIdx,self.xIdx),1)
            #print(f'page recording. yIdx max {self.maxYIdx}')
        else:
            # full image recorded
            if (self.maxYIdx >= self.scanner.imageSize[0]-1):
                _idx = self.pageIdx<=self.recordingPageIdx
                np.add.at(self.dataCube,(self.microTime[_idx],self.channel[_idx],
                                         self.yIdx[_idx],self.xIdx[_idx]),1)
                if self.accumulationIdx == 0:
                    self.dataCubeFinished = np.copy(self.dataCube)
                else:
                    self.dataCubeFinished = self.dataCubeFinished + self.dataCube
                
                self.accumulationIdx += 1
                self.flagFullImage = True

                print(f'yIdx max {self.maxYIdx}')
                print(f'yIdx {allEventYIdx}')
                print(f'full image recorded: {self.accumulationIdx} out of {self.numberOfAccumulation}')
                
                #TODO: check if _idx is set properly
                # add to the data cube data which are on new pageIdx
                #_idx = self.pageIdx>self.recordingPageIdx
                _idx = self.pageIdx==self.lastPageIdx

                self.dataCube = 0*self.dataCube
                np.add.at(self.dataCube,(self.microTime[_idx],self.channel[_idx],
                                         self.yIdx[_idx],self.xIdx[_idx]),1)


                if self.accumulationIdx == self.numberOfAccumulation:
                    self.accumulationIdx = 0
                    self.flagFullAccumulation = True
                    print(f'full stack of images recorded')
                
                self.recordingPageIdx = self.lastPageIdx

            else: # not full image was recorded
                print(f'not full image recorded. yIdx max {self.maxYIdx}')
                print(f'yIdx {allEventYIdx}')
                _idx = self.pageIdx==self.lastPageIdx
                self.recordingPageIdx = self.lastPageIdx
                self.dataCube = 0*self.dataCube
                np.add.at(self.dataCube,(self.microTime[_idx],self.channel[_idx],
                                         self.yIdx[_idx],self.xIdx[_idx]),1)

            self.maxYIdx = -1 # reset the y counter

        #end = time.time()
        #print(f'BHScannerProcessor.processData time {(end - start)*1000} ms')
        #print(f'stack Size was {self.stackSize}')

#%%

if __name__ == '__main__':
    pass
