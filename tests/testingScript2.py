''''
for small scripts tests
'''
#%%
import numpy as np
import scanImaging
from scanImaging.algorithm.dataCoding import BHData
from pathlib import Path
from ismdmdao.bhdataconverter import (translate_to_pixels,
    filter_clean_photon_events,get_microtimes, get_channel, 
    get_trucated_macrotime, get_image_data, ImageParameter, get_raw_image_data
)
import napari
from scanImaging.instrument.virtual.virtualScannerBH import VirtualBHScanner
from scanImaging.instrument.scannerBHProcessor import ScannerBHProcessor


#%%
myP = Path(scanImaging.__file__)
folder =  myP.parent / 'DATA'
file = 'voltag_one_image_mirror.npy'

streamData = np.load(str(folder / file))


# %%

#print("Clean Photon Events:", filter_clean_photon_events(data))
#print("Microtimes:", get_microtimes(data))
#print("Channel:", get_channel(data))
#print("Truncated Macrotime:", get_trucated_macrotime(data))
pixel_x, pixel_y, micro_time = translate_to_pixels(streamData)
#print("Translated Pixels X:", pixel_x)
#print("Translated Pixels Y:", pixel_y)
#print("Microtimes:", micro_time)

imageP = ImageParameter()
imageP.set_maxy_from_py(pixel_y)


imageG = get_raw_image_data(pixel_x,pixel_y,imageP)
#imageG = get_image_data(pixel_x,pixel_y,micro_time,imageP)


#%%

data = BHData()

data.streamToData(streamData)

myS = VirtualBHScanner()

myS.stack = np.vstack([data.newMacroTimeFlag,data.newLineFlag, data.macroTime]).T

myP = ScannerBHProcessor()
myP.setParameter('scanner', myS)

myP.processData()

image = myP.rawImage



#%%
viewer = napari.Viewer()
viewer.add_image(image)
viewer.add_image(imageG)
napari.run()

# %%
