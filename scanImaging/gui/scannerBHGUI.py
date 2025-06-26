'''
module for setting ADetector parameters
'''
from magicgui import magicgui
from typing import Annotated
from qtpy.QtCore import Qt
from viscope.gui.baseGUI import BaseGUI

class ScannerBHGUI(BaseGUI):
    ''' class for ADetector gui'''

    DEFAULT = {'nameGUI': 'ScannerBH'}

    def __init__(self, viscope, **kwargs):
        ''' initialise the class '''
        super().__init__(viscope, **kwargs)

        # widget
        self.parameterADetectorGui = None

        # prepare the gui of the class
        ScannerBHGUI.__setWidget(self)        

    def __setWidget(self):
        ''' prepare the gui '''
        @magicgui(auto_call=True)
        def parameterADetectorGui(
            acquisition = False,
            ):
            if acquisition: 
                 self.processor.resetCounter()
                 self.device.startAcquisition()
            else:
                 self.device.stopAcquisition()

        # add widget parameterCameraGui 
        self.parameterADetectorGui = parameterADetectorGui
        self.dw =self.vWindow.addParameterGui(self.parameterADetectorGui,name=self.DEFAULT['nameGUI'])

    def setDevice(self,device,processor=None):
        ''' set the laser '''

        super().setDevice(device)
        self.processor = processor

        # set gui parameters
        self.parameterADetectorGui.acquisition.value = self.device.acquiring
        self.dw.setWindowTitle(self.device.name)

if __name__ == "__main__":
        pass


