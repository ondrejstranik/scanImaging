''' testing script'''
#%%
from scanImaging.instrument.virtual.virtualScannerBH import VirtualBHScanner
from scanImaging.instrument.scannerProcessorBH import ScannerProcessorBH
import numpy as np
#import napari
import time
import matplotlib.pyplot as plt


#%%
bhScanner = VirtualBHScanner()

bhScanner.startAcquisition()
time.sleep(0.01)

myStack = bhScanner.getStack()
bhScanner.stopAcquisition()


#%%

bhPro = ScannerProcessorBH()
bhPro.connect(bhScanner)

bhPro.processData()

print(f'rawImage {bhPro.rawImage}')

# %%
'''
fig, ax = plt.subplots()
ax.imshow(bhScanner.virtualProbe)


fig, ax = plt.subplots()
ax.plot(myStack[:,0],color = 'g', label = 'macro tag')
ax.plot(myStack[:,1],color = 'r', label = 'new line tag')
ax.plot(myStack[:,2],color = 'b', label = 'macro time')
ax.plot(myStack[:,3],color = 'y', label = 'photons')
ax.legend()

fig, ax = plt.subplots()
ax.imshow(bhPro.rawImage)





'''
fig, ax = plt.subplots()
ax.plot(bhPro.macroTime,color = 'g', label = 'macro time')
ax.plot(bhPro.xIdx,color = 'r', label = 'x')
ax.plot(bhPro.yIdx,color = 'b', label = 'y')
ax.legend()

plt.show()
# %%
