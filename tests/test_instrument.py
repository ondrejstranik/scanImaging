''' tests '''

import pytest

def test_bHScanner():
    ''' check the data are generated'''
    from scanImaging.instrument.bHScanner import BHScanner
    import time

    bhScanner = BHScanner()
    bhScanner.connect()
    bhScanner.startAcquisition()
    i = 0
    while i<30: 
        time.sleep(0.01)
        bhScanner.updateStack()
        myStack = bhScanner.getStack()
        print(myStack)
        i += 1
    bhScanner.stopAcquisition()
    bhScanner.disconnect()


def test_bHScanner2():
    ''' check the scanner data are processed live '''
    from scanImaging.instrument.bHScanner   import BHScanner
    from scanImaging.instrument.bHScannerProcessor import BHScannerProcessor
    from viscope.main import viscope
    from scanImaging.gui.scannerBHGUI import ScannerBHGUI
    from viscope.gui.cameraViewGUI import CameraViewGUI
    import time

    bhScanner = BHScanner(name='BHScanner')
    bhScanner.connect()
    bhScanner.setParameter('threadingNow', True)

    bhPro = BHScannerProcessor(name='ScannerBHProcessor')
    bhPro.connect(scanner=bhScanner)
    bhPro.setParameter('threadingNow', True)

    adGui  = ScannerBHGUI(viscope)
    adGui.setDevice(bhScanner,processor=bhPro)

    cvGui  = CameraViewGUI(viscope,vWindow='new')
    cvGui.setDevice(bhPro)

    viscope.run()
    bhPro.disconnect()
    bhScanner.disconnect()
