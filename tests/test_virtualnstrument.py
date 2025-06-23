''' tests '''

import pytest

import scanImaging.instrument

@pytest.mark.GUI
def test_VirtualScanner_2():

    from scanImaging.instrument.scannerProcessor import ScannerProcessor
    from scanImaging.instrument.virtual.virtualScanner import VirtualScanner
    from viscope.main import viscope
    from viscope.gui.aDetectorGUI import ADetectorGUI
    from viscope.gui.cameraViewGUI import CameraViewGUI


    scanner = VirtualScanner()
    scanner.connect()
    scanner.setParameter('threadingNow',True)
    scanner.startAcquisition()

    sProc = ScannerProcessor(name='ScannerProcessor')
    sProc.connect(scanner=scanner)
    sProc.setParameter('threadingNow',True)

    adGui  = ADetectorGUI(viscope)
    adGui.setDevice(scanner)

    cvGui  = CameraViewGUI(viscope)
    cvGui.setDevice(sProc)

    viscope.run()

    sProc.disconnect()
    scanner.disconnect()


def test_VirtualScanner():
    ''' check if the data are obtained'''
    #import scanImaging

    from scanImaging.instrument.virtual.virtualScanner import VirtualScanner
    import time
    import numpy as np

    scanner = VirtualScanner()
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


def test_VirtualScannerBH():
    ''' check the data are generated'''
    from scanImaging.instrument.virtual.virtualScannerBH import VirtualBHScanner
    import time
    import matplotlib.pyplot as plt

    bhScanner = VirtualBHScanner()
    bhScanner.startAcquisition()
    time.sleep(0.01)
    myStack = bhScanner.getStack()
    bhScanner.stopAcquisition()

    fig, ax = plt.subplots()
    ax.imshow(bhScanner.virtualProbe)

    fig, ax = plt.subplots()
    ax.plot(myStack[:,0],color = 'g', label = 'macro tag')
    ax.plot(myStack[:,1],color = 'r', label = 'new line tag')
    ax.plot(myStack[:,2],color = 'b', label = 'macro time')
    ax.plot(myStack[:,3],color = 'y', label = 'photons')
    ax.legend()

    plt.show()
