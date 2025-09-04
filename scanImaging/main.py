
from viscope.main import viscope

class ScanImaging():
    ''' base top class for control'''

    DEFAULT = {}

    @classmethod
    def runReal(cls):
        ''' check the scanner data are processed live '''
        from scanImaging.instrument.bHScanner.bHScanner   import BHScanner
        from scanImaging.instrument.bHScannerProcessor import BHScannerProcessor
        from scanImaging.gui.flimViewerGUI import FlimViewerGUI
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

        fvGui  = FlimViewerGUI(viscope,vWindow='new')
        fvGui.setDevice(bhPro)

        viscope.run()
        bhPro.disconnect()
        bhScanner.disconnect()

   


    @classmethod
    def runVirtual(cls):
        ''' check the scanner data are processed live '''
        from scanImaging.instrument.virtual.virtualScannerBH import VirtualBHScanner
        from scanImaging.instrument.bHScannerProcessor import BHScannerProcessor
        from scanImaging.gui.flimViewerGUI import FlimViewerGUI
        from scanImaging.instrument.virtual.virtualDMBmc import VirtualDMBmc
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

        VirtualDMBmc=VirtualDMBmc(name='DMBmc')
        VirtualDMBmc.connect()

        adGui  = ScannerBHGUI(viscope)
        adGui.setDevice(bhScanner,processor=bhPro)

        cvGui  = CameraViewGUI(viscope,vWindow='new')
        cvGui.setDevice(bhPro)

        fvGui  = FlimViewerGUI(viscope,vWindow='new')
        fvGui.setDevice(bhPro)

        viscope.run()
        bhPro.disconnect()
        bhScanner.disconnect()

if __name__ == "__main__":
    ScanImaging.runVirtual()

