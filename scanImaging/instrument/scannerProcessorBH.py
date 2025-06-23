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


class ScannerProcessorBH(BaseProcessor):
    ''' class to collect data from virtual scanner'''
    DEFAULT = {'name': 'ScannerProcessor',
                'pixelTime': 3e2, # 
                'newPageTimeFlag': 3 # threshold for new page in the macroTime 
                }

    def __init__(self, name=None, **kwargs):
        ''' initialisation '''

        if name== None: name= ScannerProcessorBH.DEFAULT['name']
        super().__init__(name=name, **kwargs)
        
        # asynchronous Detector
        self.scanner = None

        # parameters for calculation
        self.pixelTime = self.DEFAULT['pixelTime'] # dwell time on one pixel

        # data
        self.rawImage = None
        self.xIdx = 0 # quick axis
        self.yIdx = 0 # slow axis
        self.lastYIdx = 0
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

    def _resetFunction(self,flagX,y):
        ''' function returning y with values reset to zero at flagX (bool) points'''
        y0= y[flagX]
        # no reset point
        if len(y0) == 0: return y
        dy0 = np.empty_like(y0)
        dy0[0] =y0[0]
        # more then one reset point
        if len(y0)>1: dy0[1:] = y0[1:]-y0[:-1]
        sy0 = np.zeros_like(y)
        sy0[flagX]= dy0
        bcg = np.cumsum(sy0)
        return y-bcg

    def processData(self):
        ''' process newly arrived data '''

        #print(f"processing data from {self.DEFAULT['name']}")

        # calculate total MacroTime
        macroSaw = self.scanner.stack[:,2]
        MacroSawIncrement = np.empty_like(macroSaw)
        MacroSawIncrement[0]= macroSaw[0] - self.lastMacroSawValue
        MacroSawIncrement[1:] = macroSaw[1:]-macroSaw[:-1]

        # add the overflow of macrotime
        MacroSawIncrement =  MacroSawIncrement + self.scanner.stack[:,0]*2**12
        self.macroTime = self.lastMacroTime + np.cumsum(MacroSawIncrement)

        print(f'processor macroTime {self.macroTime}')

        # TODO: it is wrong! Correct it!
        # reset macroTime on each new line
        #print(f'self.scanner.stack[:,1] {self.scanner.stack[:,1].astype(bool)}')
        self.macroTime = self._resetFunction(self.scanner.stack[:,1].astype(bool),self.macroTime)

        print(f'processor macroTime new line {self.macroTime}')

        # calculate x position
        self.xIdx = (self.macroTime//self.pixelTime).astype(int)

        print(f'xIdx {self.xIdx}')

        # calculate y position
        self.yIdx = self.lastYIdx + np.cumsum(self.scanner.stack[:,1]).astype(int)

        print(f'yIdx {self.yIdx}')


        # calculate page position
        self.pageIdx = self.lastPageIdx + np.cumsum(self.macroTime> (self.pixelTime*self.rawImage.shape[1]*
                        self.DEFAULT['newPageTimeFlag'])).astype(int)
        
        # check if full image is recorded
        if np.any(self.pageIdx==1):
            self.flagFullImage = True

        
        # add the photons to the image
        np.add.at(self.rawImage,(self.yIdx,self.xIdx),1)



#%%

if __name__ == '__main__':
    pass
