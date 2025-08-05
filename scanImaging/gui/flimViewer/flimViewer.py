'''
package for viewing ism flim data
'''
import napari
import pyqtgraph as pg
from PyQt5.QtGui import QColor, QPen
from qtpy.QtWidgets import QLabel, QSizePolicy, QWidget, QMainWindow
from qtpy.QtCore import Qt
from qtpy.QtCore import Signal

import numpy as np
from scanImaging.algorithm.flimData import FlimData
import traceback


class FlimViewer(QWidget):
    ''' main class for viewing ism flim data'''
    sigUpdateData = Signal()

    def __init__(self,flimData=None,**kwargs):
        ''' initialise the class '''
    
        super().__init__()

        # data parameter
        if flimData is None:
            self.flimData = FlimData(np.zeros((2,2,2,2)))
            self.flimData.processData()
        else:
            self.flimData = flimData

        # napari
        if 'show' in kwargs:
            self.viewer = napari.Viewer(show=kwargs['show'])
        else:
            self.viewer = napari.Viewer()

        self.imageLayer = None

        # pyqt
        if not hasattr(self, 'dockWidgetParameter'):
            self.dockWidgetParameter = None 
        if not hasattr(self, 'dockWidgetData'):
            self.dockWidgetData = None 

        self.flimGraph = None

        # set this qui of this class
        FlimViewer._setWidget(self)

    def _setWidget(self):
        ''' prepare the gui '''

        window_height = self.viewer.window._qt_window.sizeHint().height()
        window_width = self.viewer.window._qt_window.sizeHint().width()

        # add image layer
        self.imageLayer = self.viewer.add_image(self.flimData.getImage(), rgb=False, colormap="gray", 
                                            name='flimData', blending='additive')
        self.processedImageLayer = self.viewer.add_image(
            self.flimData.getImage(processed=True), rgb=False, colormap="gray", 
            name='processed Image', blending='additive')


        # set some parameters of napari
        #self.spectraLayer._keep_auto_contrast = True
        self.viewer.layers.selection.active = self.imageLayer

        # add widget spectraGraph
        self.timeGraph = pg.plot()
        self.timeGraph.setTitle(f'time Histogram')
        styles = {'color':'r', 'font-size':'20px'}
        self.timeGraph.setLabel('left', 'Count', units='a.u.')
        self.timeGraph.setLabel('bottom', 'Time ', units= 'ns')
        dw = self.viewer.window.add_dock_widget(self.timeGraph, name = 'lifetime')
        # tabify the widget
        if self.dockWidgetData is not None:
            self.viewer.window._qt_window.tabifyDockWidget(self.dockWidgetData,dw)
        self.dockWidgetData = dw
        self.viewer.window._qt_window.resizeDocks([dw], [500], Qt.Vertical)

        self.drawTimeGraph()

        # connect events 
        # connect changes of the slicer in the viewer
        self.viewer.dims.events.current_step.connect(self.drawTimeGraph)
        self.viewer.layers.selection.events.changed.connect(self.drawTimeGraph)

    def drawTimeGraph(self):
        ''' draw time lines in the spectraGraph '''
        # remove all lines
        self.timeGraph.clear()

        try:
            #mypen = QPen()
            #mypen.setWidth(0)
            #lineplot = self.timeGraph.plot(pen= mypen)
            lineplot = self.timeGraph.plot()

            if self.viewer.layers.selection.active == self.processedImageLayer:
                lineplot.setData(self.flimData.getTimeAxis(),
                                self.flimData.getTimeHistogram(processed=True))
            else:
                lineplot.setData(self.flimData.getTimeAxis(),
                                self.flimData.getTimeHistogram()[:,int(self.viewer.dims.point[0])])             
        except:
            print('error occurred in FlimViewer - drawTimeGraph')
            traceback.print_exc()

    def updateImage(self):
        ''' update image in viewer'''
        self.imageLayer.data = self.flimData.getImage()
        self.processedImageLayer.data = self.flimData.getImage(processed=True)

    def updateViewer(self):
        ''' update images/graphs in the viewer'''
        self.updateImage()
        self.drawTimeGraph()

    def setData(self, flimData):
        ''' set the data and update image '''
        self.flimData = flimData
        self.updateViewer()

if __name__ == "__main__":
    pass

        














