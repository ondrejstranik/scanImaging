'''
package to show processed flim data
'''
#%%
#from spectralCamera.gui.xywViewerGUI import XYWViewerGui
from viscope.gui.baseGUI import BaseGUI
from scanImaging.gui.flimViewer.flimViewer import FlimViewer
from plim.gui.spectralViewer.plasmonViewer import PlasmonViewer
from qtpy.QtCore import Signal


class FlimViewerGUI(BaseGUI):
    ''' main class to show  flimData'''

    DEFAULT = {'nameGUI': 'FlimViewer'}

    def __init__(self, viscope, **kwargs):
        ''' initialise the class '''
        super().__init__(viscope, **kwargs)

        # prepare the gui of the class
        FlimViewerGUI.__setWidget(self) 

    def __setWidget(self):
        ''' prepare the gui '''

        self.flimViewer = FlimViewer(show=False)
        self.viewer = self.flimViewer.viewer

        self.vWindow.addMainGUI(self.viewer.window._qt_window, name=self.DEFAULT['nameGUI'])

    def setDevice(self,device):
        super().setDevice(device)
        # connect data container
        self.plasmonViewer.pF = self.device.pF
        self.plasmonViewer.spotSpectra = self.device.spotSpectra
        # connect signals
        self.device.worker.yielded.connect(self.guiUpdateTimed)

    def updateGui(self):
        ''' update the data in gui '''
        # napari
        #self.plasmonViewer.setWavelength(self.device.wavelength)
        self.plasmonViewer.xywImage = self.device.spotSpectra.wxyImage
        self.plasmonViewer.wavelength = self.device.pF.wavelength
        self.plasmonViewer.redraw()




if __name__ == "__main__":
    pass
