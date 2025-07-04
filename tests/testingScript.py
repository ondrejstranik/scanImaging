''' testing script'''
from scanImaging.instrument.virtual.virtualScannerBH import VirtualBHScanner
from scanImaging.instrument.bHScannerProcessor import BHScannerProcessor
import numpy as np
import napari
import time

bhScanner = VirtualBHScanner()

bhPro = BHScannerProcessor()
bhPro.connect(bhScanner)

# generate and process data
bhScanner.startAcquisition()
imStack = np.zeros((5,*bhPro.rawImage.shape))

for ii in range(5):
    print(f'acquisition {ii}')
    time.sleep(0.1)
    bhScanner.updateStack()
    bhPro.processData()
    imStack[ii,...] = bhPro.rawImage

bhScanner.stopAcquisition()

# show data - visual check
viewer = napari.Viewer()
viewer.add_image(imStack, colormap='turbo')
napari.run()

