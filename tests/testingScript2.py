''''
script to test  HBdata decoding and encoding
'''
#%%
import numpy as np

#%%

class BHData():
    
    newLineFlag = None
    newMacroTimeFlag = None
    stackOverflowFlag = None
    macroTime = None
    microTime = None
    channel = None
    nData = 0

    def streamToData(self,stream):
        ''' convert 4 bytes compressed stream into BH data'''
        nStream = len(stream)

        self.macroTime= (stream&0x0FFF).astype('uint16')
        self.channel = (stream>> 12 & 0xF).astype('uint8')
        self.microTime = (stream>> 16 &0x0FFF).astype('uint16')
        self.noPhotonFlag = (stream>>31&0b1).astype('bool')
        self.newLineFlag = (stream>> 28&0b1).astype('bool')
        self.stackOverflowFlag = (stream >> 29 &0b1).astype('bool')
        self.newMacroTimeFlag = (stream>> 30 &0b1).astype('bool')
        self.nData = len(self.channel)

    def dataToStream(self):
        ''' convert BH data to 4 bytes compressed stream
        return int32 numpy 1D array 
        '''
        stream= np.empty(self.nData, dtype='uint32')
        stream = ((self.noPhotonFlag.astype('uint32')<<31) 
                + (self.newMacroTimeFlag.astype('uint32')<<30) 
                + (self.stackOverflowFlag.astype('uint32')<<29)
                + (self.newLineFlag.astype('uint32') <<28)
                + (self.microTime.astype('uint32')<<16)
                + (self.channel.astype('uint32')<<12) 
                + self.macroTime.astype('uint32'))
        return stream

    def _setArbitraryData(self,length=100):
        ''' set arbitrary data of length length. for testing purposes'''
        self.nData = length
        self.newLineFlag = np.random.randint(2,size= self.nData,dtype='bool')
        self.newMacroTimeFlag = np.random.randint(2,size= self.nData,dtype='bool')
        self.stackOverflowFlag = np.random.randint(2,size= self.nData,dtype='bool')
        self.noPhotonFlag = (self.newLineFlag | self.newMacroTimeFlag 
                             | self.stackOverflowFlag)
        self.macroTime= np.random.randint(2**12, size=self.nData, dtype='uint16')
        self.microTime= np.random.randint(2**12, size=self.nData, dtype='uint16')
        self.channel = np.random.randint(2**4, size=self.nData, dtype='uint8')

#%%

myBHData = BHData()
myBHData._setArbitraryData(5)

myStream = myBHData.dataToStream()
myBHData2 = BHData()
myBHData2.streamToData(myStream)

print(f'stream \n {myStream}')
print(f'microtime original \n {myBHData.microTime}')
print(f'microtime recovered \n {myBHData2.microTime}')

print(np.all(myBHData.microTime==myBHData2.microTime))






# %%
