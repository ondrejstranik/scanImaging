
from json import scanner
from viscope.main import viscope
import sys
# Add the current directory to sys.path
#sys.path.append(r'C:\Users\localxueweikai\Documents\GitHub\scanImaging\scanImaging\instrument\dmc')

import os
#if sys.version_info >= (3, 8):
#    os.add_dll_directory(r"C:\Program Files\Boston Micromachines\Bin64")

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
        from scanImaging.instrument.DMBmc import DMBmc
        from scanImaging.gui.dmGui import DMGui
        from viscope.gui.cameraViewGUI import CameraViewGUI
        from scanImaging.instrument.adaptiveOpticsSequencer import AdaptiveOpticsSequencer
        from scanImaging.instrument.adaptiveOpticsSequencer import ScannerImageProvider
        import time

        bhScanner = BHScanner(name='BHScanner')
        bhScanner.connect()
        bhScanner.setParameter('threadingNow', True)

        bhPro = BHScannerProcessor(name='ScannerBHProcessor')
        bhPro.connect(scanner=bhScanner)
        bhPro.setParameter('threadingNow', True)

        dmDevice = DMBmc(name='DMBmc')
        dmDevice.connect()

        scannerImageProvider=ScannerImageProvider(scanner=bhScanner,processor=bhPro)
        aoSequencer=AdaptiveOpticsSequencer(viscope=viscope,name='AdaptiveOpticsSequencer')
        aoSequencer.connect(deformable_mirror=dmDevice,image_provider=scannerImageProvider)

        adGui  = ScannerBHGUI(viscope)
        adGui.setDevice(bhScanner,processor=bhPro)

        cvGui  = CameraViewGUI(viscope,vWindow='new')
        cvGui.setDevice(bhPro)

        fvGui  = FlimViewerGUI(viscope,vWindow='new')
        fvGui.setDevice(bhPro)

        dmGui=DMGui(viscope,vWindow='new')
        dmGui.setDevice(dmDevice)
        dmGui.setAdaptiveOpticsSequenceser(aoSequencer)

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
        from scanImaging.gui.dmGui import DMGui
        from scanImaging.instrument.virtual.virtualISM import VirtualISM
        from scanImaging.gui.virtualISMGUI import VirtualISMGui
        from scanImaging.instrument.adaptiveOpticsSequencer import AdaptiveOpticsSequencer
        from scanImaging.instrument.adaptiveOpticsSequencer import ScannerImageProvider
        bhScanner = VirtualBHScanner(name='BHScanner')
        bhScanner.connect()
        bhScanner.setParameter('threadingNow', True)

        virtualDMBmc=VirtualDMBmc(name='DMBmc')
        virtualDMBmc.connect()

        virtualISM=VirtualISM(name='VirtualISM')
        virtualISM.connect(virtualScanner=bhScanner,virtualAdaptiveOptics=virtualDMBmc)
        bhPro = BHScannerProcessor(name='BHScannerProcessor')
        bhPro.connect(scanner=bhScanner)
        bhPro.setParameter('threadingNow', True)



        adGui  = ScannerBHGUI(viscope,vWindow='new')
        adGui.setDevice(bhScanner,processor=bhPro)

        cvGui  = CameraViewGUI(viscope,vWindow='new')
        cvGui.setDevice(bhPro)

        fvGui  = FlimViewerGUI(viscope,vWindow='new')
        fvGui.setDevice(bhPro)

        scannerImageProvider=ScannerImageProvider(scanner=bhScanner,processor=bhPro)
        aoSequencer=AdaptiveOpticsSequencer(viscope=viscope,name='AdaptiveOpticsSequencer')
        aoSequencer.connect(deformable_mirror=virtualDMBmc,image_provider=scannerImageProvider)

        dmGui=DMGui(viscope,vWindow='new')
        dmGui.setDevice(virtualDMBmc)
        dmGui.setAdaptiveOpticsSequenceser(aoSequencer)

         # register the virtualISM as dependent to the aoSequencer

        virtualISM.updateImage()

        virtualISMGui=VirtualISMGui(viscope,vWindow='new')
        virtualISMGui.setDevice(virtualISM)


        viscope.run()
        bhPro.disconnect()
        bhScanner.disconnect()

if __name__ == "__main__":
    ScanImaging.runVirtual()
    # choose to run virtual or real
    choice=input("Run virtual (v) or real (r) scanner? (v/r): ").strip().lower()
    if choice == 'v':
        ScanImaging.runVirtual()
    elif choice == 'r':
         ScanImaging.runReal()
    else:
        print("Invalid choice. Please enter 'v' for virtual or 'r' for real.")

