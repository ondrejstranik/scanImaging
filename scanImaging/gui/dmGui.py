'''
class for live viewing spectral images
'''
#%%
from viscope.gui.baseGUI import BaseGUI
import napari
#from hmflux.gui.slmViewer import SLMViewer
from magicgui import magicgui
import numpy as np
from qtpy.QtWidgets import QLineEdit, QSizePolicy
from viscope.gui.napariViewer.napariViewer import NapariViewer



class DMGui(BaseGUI):
    ''' main class to define and view SLM images'''
    DEFAULT = {'nameGUI': 'DM image',
               'liveUpdate': 'Live Update DM',
               'zernikeGui': 'DM Zernike',
               'updateNow': 'Update DM Now'
               }

    def __init__(self, viscope, **kwargs):
        ''' initialise the class '''
        super().__init__(viscope, **kwargs)

        self.liveUpdate = False
        self.device=None
        self.imager=None

        # fake data parameter
        self.sizeX = 20
        self.sizeY = 20
        self.imageDM= np.zeros((self.sizeX,self.sizeY))
        self.zernikeCoeffs = np.zeros(15)
        self.active_aperture = 0
        # initiate napari viewer
        self.viewer = NapariViewer(show=False)
        # napari can not work in a dock Window,
        # therefore it must run in the main Window
        #self.vWindow.addMainGUI(self.viewer.window._qt_window, name=self.DEFAULT['nameGUI'])
        #self.viewer.window._qt_window.setWindowTitle('DM image')
        self.imageLayer = self.viewer.add_image(
            self.imageDM, name='Setting Image', colormap='red',
            contrast_limits=(0.0, 255.0))
        self.imageLayerDevice = self.viewer.add_image(
            self.imageDM, name='On Device Image', colormap='green',
            contrast_limits=(0.0, 255.0))
        self.dockWidgetParameter = None
        # set gui of this class
        self.__setWidget()

    def generateImage(self):
        if self.imager is None:
            return
        self.imager.set_active_aperture(self.active_aperture)
        self.imager.set_phase_map_from_zernike(self.zernikeCoeffs)
        # preprocess raw image to 0..255 and replace NaNs
        self.imageDM = self._preprocess_image(self.imager.image)
        self.imageLayer.data = self.imageDM
        if self.liveUpdate:
            self._updateDeviceSurface()
    

    def _preprocess_image(self, arr, nan_fill=None):
        """Return a GL-safe image array.

        - Never returns None.
        - Replaces NaN/inf with nan_fill or median of finite values.
        - Ensures shape is (rows, cols) for 2D data (tries to auto-fix swapped X/Y).
        - Returns a C-contiguous uint8 array in 0..255.
        """

        a = np.asarray(arr)

        # work in float for nan handling / scaling
        try:
            a_float = a.astype(np.float64, copy=False)
        except Exception:
            a_float = np.array(a, dtype=np.float64, copy=True)

        finite_mask = np.isfinite(a_float)
        # all non-finite -> zeros of same shape
        if not np.any(finite_mask):
            return np.zeros_like(a_float, dtype=np.uint8)

        # pick fill value
        if nan_fill is None:
            fill = float(np.nanmedian(a_float[finite_mask]))
        else:
            fill = float(nan_fill)

        a_filled = a_float.copy()
        a_filled[~finite_mask] = fill

        vmin = float(np.nanmin(a_filled))
        vmax = float(np.nanmax(a_filled))

        if not np.isfinite(vmin) or not np.isfinite(vmax):
            return np.zeros_like(a_filled, dtype=np.uint8)

        # constant image -> mid-gray
        if vmin == vmax:
            out = np.full_like(a_filled, 128, dtype=np.uint8)
            return np.ascontiguousarray(out)

        # scale to 0..255 per-image
        norm = (a_filled - vmin) / (vmax - vmin)
        norm = np.clip(norm * 255.0, 0.0, 255.0)
        out = norm.astype(np.uint8)

        # ensure contiguous memory and supported dtype for GL upload
        out = np.ascontiguousarray(out)
        return out

    def __setWidget(self):
        ''' prepare the gui '''

        @magicgui(auto_call=True)
        def parameterDMGUI(liveUpdate = self.liveUpdate):
            self.liveUpdate = liveUpdate 
            self._updateDeviceSurface()

        # add widget parameterCameraGui 
        self.parameterDMGUI = parameterDMGUI
        # - os - self.dw = self.vWindow.addParameterGui(self.parameterDMGUI,name=self.DEFAULT['liveUpdate'])
        self.dw = self.viewer.window.add_dock_widget(self.parameterDMGUI, name=self.DEFAULT['liveUpdate'], area='bottom')
        if self.dockWidgetParameter is not None:
            self.viewer.window._qt_window.tabifyDockWidget(self.dockWidgetParameter,self.dw)
        # remember this dock immediately so subsequent tabify uses a valid dock
        self.dockWidgetParameter = self.dw

        # new: "Update now" button placed below the liveUpdate checkbox
        @magicgui(call_button="Update now")
        def updateNow():
            # run a single update when the button is pressed
            self.generateImage()
            if not self.liveUpdate:
                self._updateDeviceSurface()

        self.updateNow = updateNow
        # add the button to the same parameters area so it appears under the checkbox
        # - os - self.vWindow.addParameterGui(self.updateNow, name=self.DEFAULT['updateNow'])
        self.dw = self.viewer.window.add_dock_widget(self.updateNow, name=self.DEFAULT['updateNow'], area='bottom')
        if self.dockWidgetParameter is not None:
            self.viewer.window._qt_window.tabifyDockWidget(self.dockWidgetParameter,self.dw)
        self.dockWidgetParameter = self.dw

        ''' prepare the qui '''
        @magicgui(auto_call=False,
                  zernike={"widget_type": "LineEdit", "label": "Zernike coeffs"})
        def zernikeGui(zernike: str = ','.join(map(str, self.zernikeCoeffs))):
            try:
                # parse comma/space separated floats
                parts = [p.strip() for p in zernike.replace(';', ',').split(',') if p.strip() != ""]
                vals = [float(p) for p in parts]
                self.zernikeCoeffs = np.array(vals, dtype=float)
            except Exception:
                # keep previous values on parse error
                print("Error:Could not parse zernike coefficients")
                pass
            self.generateImage()
        self.zernikeGui = zernikeGui
        # self.dw =self.vWindow.addParameterGui(self.zernikeGui,name=self.DEFAULT['zernikeGui'])
        self.dw = self.viewer.window.add_dock_widget(self.zernikeGui,name=self.DEFAULT['zernikeGui'], area='bottom')
        if self.dockWidgetParameter is not None:
            self.viewer.window._qt_window.tabifyDockWidget(self.dockWidgetParameter,self.dw)
        self.dockWidgetParameter = self.dw
        self.vWindow.addMainGUI(self.viewer.window._qt_window, name=self.DEFAULT['nameGUI']) 


    def run(self):
        ''' start napari engine '''
        napari.run()

    def _updateDeviceSurface(self):
        self.device.display_surface()
        self.imageLayerDevice.data = self._preprocess_image(self.device.image)

    def setDevice(self,device):
        super().setDevice(device)
        # copy parameters
        self.sizeX = self.device.sizeX
        self.sizeY = self.device.sizeY        
        self.imager = device
        # preprocess raw image to 0..255 and replace NaNs
        self.imageDM = self._preprocess_image(self.imager.image)
        self.active_aperture = self.imager.active_aperture
        self.imageLayer.data = self.imageDM



if __name__ == "__main__":
    pass

