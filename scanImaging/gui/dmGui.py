'''
class for live viewing spectral images
'''
#%%
from viscope.gui.baseGUI import BaseGUI
import napari
#from hmflux.gui.slmViewer import SLMViewer
from magicgui import magicgui
import numpy as np
from qtpy.QtWidgets import QLineEdit, QSizePolicy, QPushButton, QVBoxLayout
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
        self.dw = self.viewer.window.add_dock_widget(self.parameterDMGUI, name=self.DEFAULT['liveUpdate'], area='right')
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
        self.dw = self.viewer.window.add_dock_widget(self.updateNow, name=self.DEFAULT['updateNow'], area='right')
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
        self.dw = self.viewer.window.add_dock_widget(self.zernikeGui,name=self.DEFAULT['zernikeGui'], area='right')
        if self.dockWidgetParameter is not None:
            self.viewer.window._qt_window.tabifyDockWidget(self.dockWidgetParameter,self.dw)
        self.dockWidgetParameter = self.dw
        # add an alternative gui to provide sliders for the most common zernike modes
        # a button is added to set to default 0 values
        @magicgui(
    auto_call=True,
    zernike_4={"widget_type": "FloatSlider", "label": "Zernike 4 (Defocus)", "min": -1000.0, "max": 1000.0},
    zernike_5={"widget_type": "FloatSlider", "label": "Zernike 5 (Astigmatism 45)", "min": -1000.0, "max": 1000.0},
    zernike_6={"widget_type": "FloatSlider", "label": "Zernike 6 (Astigmatism 0)", "min": -1000.0, "max": 1000.0},
    zernike_7={"widget_type": "FloatSlider", "label": "Zernike 7 (Coma Y)", "min": -500.0, "max": 500.0},
    zernike_8={"widget_type": "FloatSlider", "label": "Zernike 8 (Coma X)", "min": -500.0, "max": 500.0},
    zernike_9={"widget_type": "FloatSlider", "label": "Zernike 9 (Trefoil Y)", "min": -500.0, "max": 500.0},
    zernike_10={"widget_type": "FloatSlider", "label": "Zernike 10 (Trefoil X)", "min": -500.0, "max": 500.0},
    zernike_11={"widget_type": "FloatSlider", "label": "Zernike 11 (Spherical)", "min": -500.0, "max": 500.0},
)
        def zernikeSlidersGui(
                zernike_4: float = 0.0,
                zernike_5: float = 0.0,
                zernike_6: float = 0.0,
                zernike_7: float = 0.0,
                zernike_8: float = 0.0,
                zernike_9: float = 0.0,
                zernike_10: float = 0.0,
                zernike_11: float = 0.0,
            ):
            # if the "Set Defaults" button was pressed, call this function again
            # with all sliders set to zero and set_defaults=False to avoid recursion.
            # ensure zernikeCoeffs large enough
            max_index = 11
            if len(self.zernikeCoeffs) < max_index + 1:
                new_coeffs = np.zeros(max_index + 1)
                new_coeffs[:len(self.zernikeCoeffs)] = self.zernikeCoeffs
                self.zernikeCoeffs = new_coeffs

            # write slider values into the coeff array
            self.zernikeCoeffs[4] = zernike_4
            self.zernikeCoeffs[5] = zernike_5
            self.zernikeCoeffs[6] = zernike_6
            self.zernikeCoeffs[7] = zernike_7
            self.zernikeCoeffs[8] = zernike_8
            self.zernikeCoeffs[9] = zernike_9
            self.zernikeCoeffs[10] = zernike_10
            self.zernikeCoeffs[11] = zernike_11

            # update image
            self.generateImage()

        self.zernikeSlidersGui = zernikeSlidersGui
        self.dw = self.viewer.window.add_dock_widget(self.zernikeSlidersGui, name="Zernike Sliders", area='right')
        if self.dockWidgetParameter is not None:
            self.viewer.window._qt_window.tabifyDockWidget(self.dockWidgetParameter, self.dw)
        self.dockWidgetParameter = self.dw

        # add a native Qt "Set Defaults" button into the same widget (same tab)
        try:
            native = self.zernikeSlidersGui.native  # the QWidget that magicgui created
            # create the button as a child of the native widget
            btn = QPushButton("Set Defaults", native)

            def _reset_sliders():
                # Try to update magicgui widget values (this updates the visible sliders)
                names = (
                    "zernike_4", "zernike_5", "zernike_6", "zernike_7",
                    "zernike_8", "zernike_9", "zernike_10", "zernike_11",
                )
                updated = False
                for n in names:
                    try:
                        w = getattr(self.zernikeSlidersGui, n)  # magicgui exposes parameter widgets as attributes
                        # some widget wrappers use .value, others .value or .native.value — try common patterns
                        if hasattr(w, "value"):
                            w.value = 0.0
                        elif hasattr(w, "set_value"):
                            w.set_value(0.0)
                        else:
                            # try underlying native widget (Qt)
                            native = getattr(w, "native", None)
                            if native is not None and hasattr(native, "setValue"):
                                native.setValue(0.0)
                        updated = True
                    except Exception:
                        # ignore missing/unsupported widget and continue
                        pass

                if not updated:
                    # fallback: call the magicgui callable with kwargs (may update internal values)
                    try:
                        self.zernikeSlidersGui(
                            zernike_4=0.0,
                            zernike_5=0.0,
                            zernike_6=0.0,
                            zernike_7=0.0,
                            zernike_8=0.0,
                            zernike_9=0.0,
                            zernike_10=0.0,
                            zernike_11=0.0,
                        )
                    except Exception:
                        pass

                # ensure internal coeffs and image are updated
                # write zeros to coeff array explicitly (keeps internal state consistent)
                max_index = 11
                if len(self.zernikeCoeffs) < max_index + 1:
                    new_coeffs = np.zeros(max_index + 1)
                    new_coeffs[:len(self.zernikeCoeffs)] = self.zernikeCoeffs
                    self.zernikeCoeffs = new_coeffs
                self.zernikeCoeffs[4:12] = 0.0

                # regenerate image and update device preview if needed
                self.generateImage()

            btn.clicked.connect(_reset_sliders)

            # try to append the button to the existing layout
            layout = native.layout()
            if layout is not None:
                layout.addWidget(btn)
            else:
                # if no layout exists, create a simple vertical layout and add button
                vlay = QVBoxLayout(native)
                vlay.addWidget(btn)
                native.setLayout(vlay)
        except Exception:
            # non-Qt backend or unexpected widget structure: ignore and continue
            pass

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

