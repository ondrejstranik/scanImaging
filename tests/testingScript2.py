''''
script to test  HBdata decoding and encoding
'''
#%%
import numpy as np

#%%

nData = 10

newLineFlag = np.random.randint(2,size= nData)
newMacroTimeFlag = np.random.randint(2,size= nData)
stackOverflowFlag = np.random.randint(2,size= nData)
macroTime= np.random.randint(2**12, size=nData)
microTime= np.random.randint(2**12, size=nData)
channel = np.random.randint(2**4, size=nData)

#%%

def decode():


    data= np.empty(nData, dtype='int32')
    noPhotonFlag = newLineFlag or newMacroTimeFlag or stackOverflowFlag

    data = ((noPhotonFlag<<31) + (newMacroTimeFlag<<30) 
            + (stackOverflowFlag<<29) + (newLineFlag <<28) +
            (microTime<<16) + (channel<<12) + (macroTime))
    return data

def encode(data):

    nTag = len(data)

    nLTag = np.empty(nTag,dtype='uint8')
    nMTTag = np.empty(nTag,dtype='uint8')
    sOTag = np.empty(nTag,dtype='uint8')
    nPTag = np.empty(nTag,dtype='uint8')
    maT = np.empty(nTag,dtype='uint16')
    miT = np.empty(nTag,dtype='uint16')
    ch = np.empty(nTag,dtype='uint8')

    macroT= (data&0x0FFF).astype('uint16')








# %%
