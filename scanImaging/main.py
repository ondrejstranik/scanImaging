
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
        import time

        bhScanner = BHScanner(name='BHScanner')
        bhScanner.connect()
        bhScanner.setParameter('threadingNow', True)

        bhPro = BHScannerProcessor(name='ScannerBHProcessor')
        bhPro.connect(scanner=bhScanner)
        bhPro.setParameter('threadingNow', True)

        dmDevice = DMBmc(name='DMBmc')
        dmDevice.connect()

        adGui  = ScannerBHGUI(viscope)
        adGui.setDevice(bhScanner,processor=bhPro)

        cvGui  = CameraViewGUI(viscope,vWindow='new')
        cvGui.setDevice(bhPro)

        fvGui  = FlimViewerGUI(viscope,vWindow='new')
        fvGui.setDevice(bhPro)

        dmGui=DMGui(viscope,vWindow='new')
        dmGui.setDevice(dmDevice)

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
        import time
        from PIL import Image
        import numpy as np
        
 
        bhScanner = VirtualBHScanner(name='BHScanner')
        bhScanner.connect()
        bhScanner.setParameter('threadingNow', True)

        from pathlib import Path
        base_path = Path(__file__).resolve().parent
        p = rf"{base_path}/instrument/virtual/images/Gemini_Generated_Image_saa5ihsaa5ihsaa5.png"

                # path to your gray PNG
 #       p = "/home/georg/0_work/projects_IPHT/adaptive_optics_psf_analysis/ondra_example/scanImaging/scanImaging/instrument/virtual/images/radial-sine-144.png"
        
        # load as grayscale
        img = Image.open(p).convert("L")
        
        # get target size from scanner (rows, cols)
        rows, cols = bhScanner.imageSize  # numpy array or tuple
        img = img.resize((int(cols), int(rows)), resample=Image.BILINEAR)
        
        # to numpy float32 and normalize to [0,1]
        arr = np.asarray(img, dtype=np.float32)
        if arr.max() != 0:
            arr /= arr.max()
        
        # if you need multiple channels (must equal scanner.numberOfChannel)
        # arr = np.repeat(arr[:, :, None], scanner.numberOfChannel, axis=2)
        
        # set as virtual probe
        bhScanner.setVirtualProbe(arr)

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
#    ScanImaging.runReal()

