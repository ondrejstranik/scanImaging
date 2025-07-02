#%%
''' module for a beckel&Hickel scanner having internal stack for the data'''

import numpy as np
import time

from viscope.instrument.base.baseADetector import BaseADetector
from scanImaging.algorithm.dataCoding import BHData

from bh_spc import spcm


class BHScanner(BaseADetector):
    ''' class to control B&H scanner
    '''

    DEFAULT = {'name':'BHScanner',
               'bufferSize':32768,  # Max number of 16-bit words in a single read.
               'configFile': r'C:\Users\localxueweikai\Desktop\copy of spcm.ini\spcm_Georg.ini',
               'modeNumber' : 0, # default is zero => use hardware 
               'imageSize': np.array([512,512])
               } 

    def __init__(self, name=DEFAULT['name'], **kwargs):
        ''' detector initialization'''
        super().__init__(name=name, **kwargs)

        # variable to initiates BHScanner
        self.bufferSize = self.DEFAULT['bufferSize'] 
        self.configFile = self.DEFAULT['configFile']
        self.modeNumber = self.DEFAULT['modeNumber']
        self.imageSize = self.DEFAULT['imageSize']

        self.bhData = BHData()

    def connect(self):
        ''' connect to the instrument '''
        super().connect()

        spcm.init(self.configFile)
        print(f'mod number {self.modeNumber} ' + str(spcm.get_init_status(self.modeNumber)))
        print(f'parameter ID \n')
        for e in spcm.ParID:
            print(e.value, e.name, e.type.__name__)
        print('current parameters \n')
        params = spcm.get_parameters(self.modeNumber)
        for par, val in params.items():
            print(f"{par} = {val}")
        res2 = spcm.get_adjust_parameters(self.modeNumber)
        print(f' adjust parameters \n')
        for par, val in res2.items():
            print(f"{par} = {val}")

    def disconnect(self):
        ''' disconnect the instrument '''
        super().disconnect()
        spcm.close()


    def startAcquisition(self):
        '''start detector collecting data in a stack '''
        super().startAcquisition()
        spcm.start_measurement(self.modeNumber)

    def stopAcquisition(self):
        '''stop detector collecting data in a stack '''
        super().stopAcquisition()
        spcm.stop_measurement(self.modeNumber)

    def getStack(self):
        ''' get data from the stack'''        

        data = []  # Collect arrays of data into a list.
        collectBuffer = True
        while collectBuffer:
            buffer =  spcm.read_fifo_to_array(self.modeNumber, self.bufferSize)
            if len(buffer):
                data.append(buffer)
            if len(buffer) < self.bufferSize: # Buffer is not full
                collectBuffer = False
        
        # TODO: move this to the processor 
        # convert the stream to stack further process by bhScannerProcessor
        if len(data) > 0:
            self.bhData.streamToData(np.concatenate(data).view(np.uint32))
            self.stack = np.vstack([self.bhData.newMacroTimeFlag,self.bhData.newLineFlag,self.bhData.macroTime]).T
        else:
            self.stack = None
        return self.stack

if __name__ == '__main__':

    pass





# %%