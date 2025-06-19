''' tests '''

import pytest

import scanImaging.instrument

@pytest.mark.GUI
def test_VirtualScanner_2():

    from viscope.instrument.virtual.virtualADetector import VirtualADetector
    from viscope.instrument.aDetectorProcessor import ADetectorProcessor
    from viscope.main import viscope
    from viscope.gui.aDetectorGUI import ADetectorGUI
    from viscope.gui.aDetectorViewGUI import ADetectorViewGUI

    aDet = VirtualADetector(name='ADetector')
    aDet.connect()
    aDet.setParameter('threadingNow',True)

    aDProc = ADetectorProcessor(name='ADetectorProcessor')
    aDProc.connect(aDetector=aDet)
    aDProc.setParameter('threadingNow',True)

    adGui  = ADetectorGUI(viscope)
    adGui.setDevice(aDet)

    advGui  = ADetectorViewGUI(viscope)
    advGui.setDevice(aDProc)

    viscope.run()

    aDet.disconnect()
    aDProc.disconnect()


def test_VirtualScanner():
    ''' check if the data are obtained'''
    import scanImaging

    #from scanImaging.instrument.virtual.virtualScanner import VirtualScanner
    import time
    import numpy as np

    scanner = scanImaging.instrument.virtual.virtualScanner.VirtualScanner()
    scanner.connect()
    scanner.setParameter('threadingNow',True)
    scanner.startAcquisition()

    cTime = time.time()
    while time.time()-cTime < 3:    
        if scanner.flagLoop.is_set():
            data = scanner.getStack()
            print(f'stack \n {data}')
            scanner.flagLoop.clear()

    scanner.startAcquisition()
    scanner.disconnect()
    #%%