''''
script to test  function processing flags
'''
#%%

from scanImaging.instrument.virtual.virtualScannerBH import VirtualBHScanner
from scanImaging.instrument.scannerBHProcessor import ScannerBHProcessor
from viscope.main import viscope
from viscope.gui.aDetectorGUI import ADetectorGUI
from viscope.gui.cameraViewGUI import CameraViewGUI

bhScanner = VirtualBHScanner(name='BHScanner')
bhScanner.connect()
bhScanner.setParameter('threadingNow', True)

bhPro = ScannerBHProcessor(name='ScannerProcessor')
bhPro.connect(scanner=bhScanner)
bhPro.setParameter('threadingNow', True)

adGui  = ADetectorGUI(viscope)
adGui.setDevice(bhScanner,processor=bhPro)

cvGui  = CameraViewGUI(viscope)
cvGui.setDevice(bhPro)

#bhScanner.startAcquisition()
viscope.run()

bhScanner.stopAcquisition()
bhPro.disconnect()
bhScanner.disconnect()




