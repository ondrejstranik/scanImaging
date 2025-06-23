''' testing script'''
#%%
from scanImaging.instrument.virtual.virtualScannerBH import VirtualBHScanner
import numpy as np
import napari


#%%
bhScanner = VirtualBHScanner()

bhScanner.startAcquisition()

bhScanner.getStack()

bhScanner.startAcquisition()

