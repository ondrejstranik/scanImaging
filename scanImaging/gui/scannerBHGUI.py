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
        self.parameterTimingGui = None

        # prepare the gui of the class
        ScannerBHGUI.__setWidget(self)        

    def __setWidget(self):
        ''' prepare the gui '''
        @magicgui(auto_call=True,
                    stackSize = {'label':'stackSize',
                                 'widget_type':'Label'})
        def parameterADetectorGui(
            acquisition = False,
            stackSize = ''
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
            self.processor.generateDataCube = False if continuous else True

        @magicgui(auto_call=True)
        def parameterTimingGui(
            line_period_ms: Annotated[float, {"label": "Line Period (ms)", "min": 0.1, "max": 1000.0, "step": 0.1}] = 50.0,
            active_fraction: Annotated[float, {"label": "Active Fraction", "min": 0.1, "max": 1.0, "step": 0.01}] = 0.801
            ):
            """Scanner timing parameters

            Line Period: Total time for one scan line including flyback [ms]
            Active Fraction: Fraction of line period used for active scanning (rest is flyback)

            These parameters control how photon arrival times are mapped to pixel positions.
            Adjust if scan speed changes or signal appears misaligned.
            """
            self.processor.setParameter('line_period_ms', line_period_ms)
            self.processor.setParameter('active_fraction', active_fraction)

        @magicgui(call_button='Restart Scanner')
        def restartScannerGui():
            """Manual scanner restart for locked/stalled scanner

            Use this button when the scanner appears locked:
            - Photon counts still received but image not progressing
            - Line triggers still sent but scan stuck at one position
            - Only detectable by user observation
            """
            print("Manual scanner restart initiated...")
            try:
                import time

                # Stop acquisition
                self.device.stopAcquisition()
                print("  Scanner stopped")
                time.sleep(0.5)

                # Reset processor state
                if hasattr(self.processor, 'reset_accumulation'):
                    self.processor.reset_accumulation()
                    print("  Processor accumulation reset")
                else:
                    self.processor.resetCounter()
                    print("  Processor counter reset")
                time.sleep(0.5)

                # Restart acquisition
                self.device.startAcquisition()
                print("  Scanner restarted")
                time.sleep(1.0)

                # Update GUI state
                self.parameterADetectorGui.acquisition.value = True

                print("Scanner restart complete!")
            except Exception as e:
                print(f"ERROR during manual scanner restart: {e}")

        # add widget parameterCameraGui
        self.parameterADetectorGui = parameterADetectorGui
        self.parameter2ADetectorGui = parameter2ADetectorGui
        self.parameterTimingGui = parameterTimingGui
        self.restartScannerGui = restartScannerGui

        self.dw =self.vWindow.addParameterGui(self.parameterADetectorGui,name=self.DEFAULT['nameGUI'])
        self.dw =self.vWindow.addParameterGui(self.parameter2ADetectorGui,name=self.DEFAULT['nameGUI']+'_2')
        self.dw =self.vWindow.addParameterGui(self.parameterTimingGui,name=self.DEFAULT['nameGUI']+'_timing')
        self.dw =self.vWindow.addParameterGui(self.restartScannerGui,name=self.DEFAULT['nameGUI']+'_restart')


    def setDevice(self,device,processor=None):
        ''' set the laser '''

        super().setDevice(device)
        self.processor = processor

        # set gui parameters
        self.parameterADetectorGui.acquisition.value = self.device.acquiring
        self.parameter2ADetectorGui.numberOfAccumulation.value = self.processor.numberOfAccumulation
        self.parameter2ADetectorGui.continuous.value = self.processor.generateDataCube
        self.parameterTimingGui.line_period_ms.value = self.processor.line_period_ms
        self.parameterTimingGui.active_fraction.value = self.processor.active_fraction

        self.dw.setWindowTitle(self.device.name)

        # connect the signals
        self.processor.worker.yielded.connect(self.guiUpdateTimed)

    def updateGui(self):
        ''' update the data in gui '''
        if (self.processor.flagFullAccumulation and 
            not self.parameter2ADetectorGui.continuous.value):
             #self.device.stopAcquisition()
             self.parameterADetectorGui.acquisition.value = False
        self.parameterADetectorGui.stackSize.value = str(self.processor.stackSize)
             
if __name__ == "__main__":
        pass


