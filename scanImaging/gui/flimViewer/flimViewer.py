'''
package for viewing ism flim data
'''
import napari
import pyqtgraph as pg
from PyQt5.QtGui import QColor, QPen
from qtpy.QtWidgets import QLabel, QSizePolicy, QWidget, QMainWindow
from qtpy.QtCore import Qt
from qtpy.QtCore import Signal
from viscope.gui.napariViewer.napariViewer import NapariViewer

import numpy as np
from scanImaging.algorithm.flimData import FlimData
from scanImaging.algorithm.flimFit import rough_single_exp_fit
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
            self.viewer = NapariViewer(show=kwargs['show'])
        else:
            self.viewer = NapariViewer()

        self.imageLayer = None

        # pyqt
        if not hasattr(self, 'dockWidgetParameter'):
            self.dockWidgetParameter = None 
        if not hasattr(self, 'dockWidgetData'):
            self.dockWidgetData = None 

        self.flimGraph = None

        # rough single-exponential fit overlaid on the time histogram
        self.fitEnabled = kwargs.get('fitEnabled', True)
        self.fitPeakOffsetNs = kwargs.get('fitPeakOffsetNs', 0.3)

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
        ''' draw time histogram (+ rough single-exp fit) in the time graph '''
        # remove all lines
        self.timeGraph.clear()

        try:
            timeAxis = self.flimData.getTimeAxis()
            if self.viewer.layers.selection.active == self.processedImageLayer:
                histogram = self.flimData.getTimeHistogram(processed=True)
            else:
                histogram = self.flimData.getTimeHistogram()[:, int(self.viewer.dims.point[0])]
            histogram = np.asarray(histogram, dtype=float)

            lineplot = self.timeGraph.plot()
            lineplot.setData(timeAxis, histogram)

            # overlay a rough single-exponential fit (sanity check on tau range)
            self._drawRoughFit(timeAxis, histogram)
        except:
            print('error occurred in FlimViewer - drawTimeGraph')
            traceback.print_exc()

    def _drawRoughFit(self, timeAxis, histogram):
        ''' fit a single exponential to the displayed histogram and overlay it,
        showing the estimated lifetime in the plot title. Failures are silent
        (e.g. empty/initial data) so the viewer keeps working. '''
        if not self.fitEnabled:
            self.timeGraph.setTitle('time Histogram')
            return
        try:
            fit = rough_single_exp_fit(timeAxis, histogram,
                                       peak_offset_ns=self.fitPeakOffsetNs)
            if fit['success']:
                fitline = self.timeGraph.plot(pen=pg.mkPen('r', width=2))
                fitline.setData(fit['t_fit'], fit['y_fit'])
                self.timeGraph.setTitle(
                    f"time Histogram   (rough τ = {fit['tau_ns']:.2f} ns, "
                    f"fit from {fit['t_start_ns']:.2f} ns)")
            else:
                self.timeGraph.setTitle('time Histogram   (no fit)')
        except Exception:
            self.timeGraph.setTitle('time Histogram')
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

        














