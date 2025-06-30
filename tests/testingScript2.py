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
        self.channel = (stream&0xF000 >> 12).astype('uint8')
        self.microTime = (stream&0x0FFF_0000 >> 16).astype('uint16')
        self.noPhotonFlag = (stream&0b1000_0000_0000_0000 >>31).astype('bool')
        self.newLineFlag = (stream&0b0001_0000_0000_0000 >> 28).astype('bool')
        self.stackOverflowFlag = (stream&0b0010_0000_0000_0000 >> 29).astype('bool')
        self.newMacroTimeFlag = (stream&0b0100_0000_0000_0000 >> 30).astype('bool')
        self.nData = len(self.channel)

    def dataToStream(self):
        ''' convert BH data to 4 bytes compressed stream
        return int32 numpy 1D array 
        '''
        stream= np.empty(self.nData, dtype='uint32')
        stream = ((self.noPhotonFlag<<31).astype('uint32') 
                + (self.newMacroTimeFlag<<30).astype('uint32') 
                + (self.stackOverflowFlag<<29).astype('uint32')
                + (self.newLineFlag <<28).astype('uint32')
                + (self.microTime<<16).astype('uint32')
                + (self.channel<<12).astype('uint32') 
                + (self.macroTime).astype('uint32'))
        return stream

    def _setArbitraryData(self,length=100):
        ''' set arbitrary data of length length. for testing purposes'''
        self.nData = length
        self.newLineFlag = np.random.randint(2,size= self.nData,dtype='bool')
        self.newMacroTimeFlag = np.random.randint(2,size= self.nData,dtype='bool')
        self.stackOverflowFlag = np.random.randint(2,size= self.nData,dtype='bool')
        self.noPhotonFlag = (self.newLineFlag | self.newMacroTimeFlag 
                             | self.stackOverflowFlag)
        self.macroTime= np.random.randint(2**12, size=self.nData)
        self.microTime= np.random.randint(2**12, size=self.nData)
        self.channel = np.random.randint(2**4, size=self.nData)

#%%

myBHData = BHData()
myBHData._setArbitraryData(5)

myStream = myBHData.dataToStream()
myBHData2 = BHData()
myBHData2.streamToData(myStream)

print(f'stream \n {myStream}')
print(myBHData.microTime)
print(myBHData2.microTime)

print(np.all(myBHData.microTime==myBHData2.microTime))






# %%
