''' tests '''

import pytest

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

    scanner.stopAcquisition()
    scanner.disconnect()
    #%%

@pytest.mark.GUI
def test_VirtualScannerBH():
    ''' check the data are generated'''

    from scanImaging.instrument.virtual.virtualScannerBH import VirtualBHScanner
    import time
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print('to run this test install matplotlib')
        assert False


    bhScanner = VirtualBHScanner()
    bhScanner.connect()
    bhScanner.setParameter('threadingNow',True)    
    bhScanner.startAcquisition()
    
    while bhScanner.isEmptyStack():
        #print('waiting')
        time.sleep(0.1)
   
    myStack = bhScanner.getStack()

    bhScanner.stopAcquisition()
    bhScanner.disconnect()

    fig, ax = plt.subplots()
    ax.imshow(bhScanner.virtualProbe)

    fig, ax = plt.subplots()
    ax.plot(myStack[:,0],color = 'g', label = 'macro tag')
    ax.plot(myStack[:,1],color = 'r', label = 'new line tag')
    ax.plot(myStack[:,2],color = 'b', label = 'macro time')
    ax.legend()

    plt.show()

@pytest.mark.GUI
def test_ScannerBHProcessor():
    ''' check the scanner data are processed '''
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


@pytest.mark.GUI
def test_ScannerBHProcessor2():
    ''' check the scanner data are processed live '''
    from scanImaging.instrument.virtual.virtualScannerBH import VirtualBHScanner
    from scanImaging.instrument.bHScannerProcessor import BHScannerProcessor
    from viscope.main import viscope
    from scanImaging.gui.scannerBHGUI import ScannerBHGUI
    from viscope.gui.cameraViewGUI import CameraViewGUI
    import time

    bhScanner = VirtualBHScanner(name='BHScanner')
    bhScanner.connect()
    bhScanner.setParameter('threadingNow', True)

    bhPro = BHScannerProcessor(name='BHScannerProcessor')
    bhPro.connect(scanner=bhScanner)
    bhPro.setParameter('threadingNow', True)

    adGui  = ScannerBHGUI(viscope)
    adGui.setDevice(bhScanner,processor=bhPro)

    cvGui  = CameraViewGUI(viscope,vWindow='new')
    cvGui.setDevice(bhPro)

    viscope.run()
    bhPro.disconnect()
    bhScanner.disconnect()


