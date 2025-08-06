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
        self.parameter2ADetectorGui = None

        # prepare the gui of the class
        ScannerBHGUI.__setWidget(self)        

    def __setWidget(self):
        ''' prepare the gui '''
        @magicgui(auto_call=True)
        def parameterADetectorGui(
            acquisition = False
            ):
            if acquisition: 
                 self.processor.resetCounter()
                 self.device.startAcquisition()
            else:
                 self.device.stopAcquisition()
        
        @magicgui(auto_call=True)
        def parameter2ADetectorGui(
            continuous = True,
            numberOfAccumulation = 5
            ):
            self.processor.numberOfAccumulation = numberOfAccumulation


        # add widget parameterCameraGui 
        self.parameterADetectorGui = parameterADetectorGui
        self.parameter2ADetectorGui = parameter2ADetectorGui

        self.dw =self.vWindow.addParameterGui(self.parameterADetectorGui,name=self.DEFAULT['nameGUI'])
        self.dw =self.vWindow.addParameterGui(self.parameter2ADetectorGui,name=self.DEFAULT['nameGUI']+'_2')


    def setDevice(self,device,processor=None):
        ''' set the laser '''

        super().setDevice(device)
        self.processor = processor

        # set gui parameters
        self.parameterADetectorGui.acquisition.value = self.device.acquiring
        self.parameter2ADetectorGui.numberOfAccumulation.value = self.processor.numberOfAccumulation

        self.dw.setWindowTitle(self.device.name)

        # connect the signals
        self.processor.worker.yielded.connect(self.guiUpdateTimed)

    def updateGui(self):
        ''' update the data in gui '''
        if (self.processor.flagFullAccumulation and 
            not self.parameter2ADetectorGui.continuous.value):
             #self.device.stopAcquisition()
             self.parameterADetectorGui.acquisition.value = False
             
if __name__ == "__main__":
        pass


