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


class ScannerProcessor(BaseProcessor):
    ''' class to collect data from virtual scanner'''
    DEFAULT = {'name': 'ScannerProcessor',
                }

    def __init__(self, name=None, **kwargs):
        ''' initialisation '''

        if name== None: name= ScannerProcessor.DEFAULT['name']
        super().__init__(name=name, **kwargs)
        
        # asynchronous Detector
        self.scanner = None
        
        # data
        self.rawImage = None

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
        
        try:
            #print(f'scanner stack \n {self.scanner.stack}')
            # adding the values
            #self.rawImage[(self.scanner.stack[:,1].astype(int),self.scanner.stack[:,0].astype(int))] = (
            #    self.rawImage[(self.scanner.stack[:,1].astype(int),self.scanner.stack[:,0].astype(int))] + 
            #    self.scanner.stack[:,2]) 

            self.rawImage[(self.scanner.stack[:,0].astype(int),
                           self.scanner.stack[:,1].astype(int))] = (
                self.scanner.stack[:,2]) 


        except:
            print(f'from {self.DEFAULT["name"]}: can not process the data')
            #print(f'stack {self.aDetector.stack}')
            #print(f'time {self.time}')
            #print(f'signal {self.signal}')

        # indicate that data from at ADetector were processed
        # probably not necessary it is done by the processor loop
        #self.aDetector.flagLoop.clear()


#%%

if __name__ == '__main__':
    pass
