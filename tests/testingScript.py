''' testing script'''
#%%
from scanImaging.instrument.virtual.virtualScannerBH import VirtualBHScanner
from scanImaging.instrument.scannerProcessorBH import ScannerProcessorBH
import numpy as np
import napari
import time
import matplotlib.pyplot as plt


#%%
bhScanner = VirtualBHScanner()

bhPro = ScannerProcessorBH()
bhPro.connect(bhScanner)

bhScanner.startAcquisition()

myStackList = []

for ii in range(5):
    print(f'acquisition {ii}')
    time.sleep(0.1)
    myStackList.append(bhScanner.getStack())

bhScanner.stopAcquisition()

#%%

imStack = np.zeros((5,*bhPro.rawImage.shape))

for ii in range(5):
    bhScanner.stack = myStackList[ii]
    bhPro.processData()
    imStack[ii,...] = bhPro.rawImage

#%%

viewer = napari.Viewer()
viewer.add_image(imStack, colormap='turbo')
napari.run()
#print(f'rawImage {bhPro.rawImage}')



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


fig, ax = plt.subplots()
ax.plot(bhPro.macroTime,color = 'g', label = 'macro time')
ax.legend()

plt.show()




fig, ax = plt.subplots()
ax.plot(bhPro.macroTime,color = 'g', label = 'macro time')
ax.plot(bhPro.xIdx,color = 'r', label = 'x')
ax.plot(bhPro.yIdx,color = 'b', label = 'y')
ax.legend()

plt.show()



viewer = napari.Viewer()
viewer.add_image(bhPro.rawImage, colormap='turbo')
napari.run()

'''
# %%


