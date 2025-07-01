''' testing script'''
''' check the scanner data are processed live '''
from scanImaging.instrument.virtual.virtualScannerBH import VirtualBHScanner
from scanImaging.instrument.scannerBHProcessor import ScannerBHProcessor
from viscope.main import viscope
from scanImaging.gui.scannerBHGUI import ScannerBHGUI
from viscope.gui.cameraViewGUI import CameraViewGUI
import time

bhScanner = VirtualBHScanner(name='BHScanner')
bhScanner.connect()
bhScanner.setParameter('threadingNow', True)

bhPro = ScannerBHProcessor(name='ScannerProcessor')
bhPro.connect(scanner=bhScanner)
bhPro.setParameter('threadingNow', True)

adGui  = ScannerBHGUI(viscope)
adGui.setDevice(bhScanner,processor=bhPro)

cvGui  = CameraViewGUI(viscope,vWindow='new')
cvGui.setDevice(bhPro)

viscope.run()
time.sleep(1)

bhPro.disconnect()
bhScanner.disconnect()
