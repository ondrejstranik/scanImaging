'''
package to show processed flim data
'''
#%%
#from spectralCamera.gui.xywViewerGUI import XYWViewerGui
from viscope.gui.baseGUI import BaseGUI
from scanImaging.gui.flimViewer.flimViewer import FlimViewer
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
        # connect signals
        self.device.worker.yielded.connect(self.guiUpdateTimed)

    def updateGui(self):
        ''' update the data in gui '''
        
        if self.device.flagFullImage:
            print('updating flimViewerGUI')
            self.flimViewer.flimData.setData(self.device.dataCube)
            self.flimViewer.updateViewer()
            self.device.flagFullImage = False

if __name__ == "__main__":
    pass
