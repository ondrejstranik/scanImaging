'''
class for live viewing spectral images
'''
#%%
from viscope.gui.baseGUI import BaseGUI
import napari
#from hmflux.gui.slmViewer import SLMViewer
from magicgui import magicgui
import numpy as np
from qtpy.QtWidgets import QLineEdit, QSizePolicy, QPushButton, QVBoxLayout, QWidget, QLabel, QSpinBox, QHBoxLayout, QComboBox
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
        self.aoSequencer=None

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
    
    def generateDisplayImage(self):
        if self.imager is None:
            return
        self.imager.set_active_aperture(self.active_aperture)
        self.imager.set_phase_map_from_zernike(self.zernikeCoeffs)
        # preprocess raw image to 0..255 and replace NaNs
        self.imageDM = self._preprocess_image(self.imager.image)
        self.imageLayer.data = self.imageDM

    def generateImage(self):
        self.generateDisplayImage()
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
    def _update_zernike_display(self):
        try:
            text = ', '.join(f'{v:.2f}' for v in np.asarray(self.zernikeCoeffs, dtype=float))
            if hasattr(self, 'zernikeLabel'):
                self.zernikeLabel.setText(text)
        except Exception:
            pass

    def __setWidget(self):
        ''' prepare the gui '''

        # create the two magicgui controls as before (parameterDMGUI and updateNow)
        @magicgui(auto_call=True)
        def parameterDMGUI(liveUpdate = self.liveUpdate):
            self.liveUpdate = liveUpdate
            self._updateDeviceSurface()

        @magicgui(call_button="Update now")
        def updateNow():
            self.generateImage()
            if not self.liveUpdate:
                self._updateDeviceSurface()

        # combine into one dock
        container = QWidget()
        vlay = QVBoxLayout(container)
        for w in (parameterDMGUI, updateNow):
            native = getattr(w, "native", None)
            if native is not None:
                vlay.addWidget(native)
            else:
                # fallback: add the callable as a dock (rare)
                self.viewer.window.add_dock_widget(w, name=w.__name__, area='right')
        # add combined dock and keep it tabified with existing parameter docks
        self.dw = self.viewer.window.add_dock_widget(container, name=self.DEFAULT['liveUpdate'], area='right')
        if self.dockWidgetParameter is not None:
            self.viewer.window._qt_window.tabifyDockWidget(self.dockWidgetParameter, self.dw)
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
            self._update_zernike_display()

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
                self._update_zernike_display()

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

        # --- Adaptive Optics controls (single tab with Start/Stop) ---
        def _ao_start_clicked():
            if self.aoSequencer is None:
                print("No AO sequencer set")
                return
            try:
                if hasattr(self.aoSequencer, 'register_dependent'):
                    self.aoSequencer.register_dependent(self)
                if hasattr(self.aoSequencer, 'start'):
                    self.aoSequencer.start()
            except Exception as e:
                print("AO start error:", e)

        def _ao_stop_clicked():
            if self.aoSequencer is None:
                return
            try:
                if hasattr(self.aoSequencer, 'stop'):
                    self.aoSequencer.stop()
                if hasattr(self.aoSequencer, 'unregister_dependent'):
                    self.aoSequencer.unregister_dependent(self)
            except Exception as e:
                print("AO stop error:", e)

        ao_control_widget = QWidget()
        vlay = QVBoxLayout(ao_control_widget)
        btn_start = QPushButton("Start AO", ao_control_widget)
        btn_stop = QPushButton("Stop AO", ao_control_widget)
        btn_start.clicked.connect(_ao_start_clicked)
        btn_stop.clicked.connect(_ao_stop_clicked)
        vlay.addWidget(btn_start)
        vlay.addWidget(btn_stop)

        # iterations
        h = QHBoxLayout()
        lbl_it = QLabel("Optim iterations", ao_control_widget)
        spin_iter = QSpinBox(ao_control_widget)
        spin_iter.setRange(1, 1000)
        spin_iter.setValue(getattr(self.aoSequencer, 'optim_iterations', 3))
        def _on_iter(v):
            try:
                if self.aoSequencer is not None:
                    self.aoSequencer.optim_iterations = int(v)
            except Exception:
                pass
        spin_iter.valueChanged.connect(_on_iter)
        h.addWidget(lbl_it); h.addWidget(spin_iter)
        vlay.addLayout(h)

        # steps per mode
        h2 = QHBoxLayout()
        lbl_steps = QLabel("Steps per mode", ao_control_widget)
        spin_steps = QSpinBox(ao_control_widget)
        spin_steps.setRange(1, 500)
        spin_steps.setValue(getattr(self.aoSequencer, 'num_steps_per_mode', 3))
        def _on_steps(v):
            try:
                if self.aoSequencer is not None:
                    self.aoSequencer.num_steps_per_mode = int(v)
            except Exception:
                pass
        spin_steps.valueChanged.connect(_on_steps)
        h2.addWidget(lbl_steps); h2.addWidget(spin_steps)
        vlay.addLayout(h2)

        # optimization method
        h3 = QHBoxLayout()
        lbl_method = QLabel("Optim method", ao_control_widget)
        cb_method = QComboBox(ao_control_widget)
        methods = ['simple_interpolation', 'steepest_descent']
        cb_method.addItems(methods)
        cb_method.setCurrentText(getattr(self.aoSequencer, 'optim_method', methods[0]))
        def _on_method(txt):
            try:
                if self.aoSequencer is not None:
                    self.aoSequencer.optim_method = str(txt)
            except Exception:
                pass
        cb_method.currentTextChanged.connect(_on_method)
        h3.addWidget(lbl_method); h3.addWidget(cb_method)
        vlay.addLayout(h3)
        # initial zernike indices (comma-separated)
        h4 = QHBoxLayout()
        lbl_idxs = QLabel("Initial Zernike indices", ao_control_widget)
        le_idxs = QLineEdit(ao_control_widget)
        default_idxs = getattr(self.aoSequencer, 'initial_zernike_indices', [4,11,2])
        le_idxs.setText(','.join(map(str, default_idxs)))
        def _on_idxs():
            print("AOGui: setting initial zernike indices")
            txt = le_idxs.text()
            try:
                parts = [p.strip() for p in txt.replace(';', ',').split(',') if p.strip() != ""]
                idxs = [int(p) for p in parts]
            except Exception:
                idxs = []
            try:
                if self.aoSequencer is not None:
                    self.aoSequencer.setInitialZernikeModes(indices=idxs)
            except Exception:
                pass
        le_idxs.editingFinished.connect(_on_idxs)
        h4.addWidget(lbl_idxs); h4.addWidget(le_idxs)
        vlay.addLayout(h4)
        # set sizes, comma separated
        h5 = QHBoxLayout()
        lbl_sizes = QLabel("Zernike step sizes in nm", ao_control_widget)
        le_sizes = QLineEdit(ao_control_widget)
        default_sizes = getattr(self.aoSequencer, 'zernike_amplitude_scan_nm', [20,15,10])
        le_sizes.setText(','.join(map(str, default_sizes)))
        def _on_sizes():
            txt = le_sizes.text()
            try:
                parts = [p.strip() for p in txt.replace(';', ',').split(',') if p.strip() != ""]
                sizes = [float(p) for p in parts]
            except Exception:
                sizes = []
            try:
                if self.aoSequencer is not None:
                    self.aoSequencer.setAmplitudeScan(amplitude_scan_nm=sizes)
            except Exception:
                pass
        le_sizes.editingFinished.connect(_on_sizes)
        h5.addWidget(lbl_sizes); h5.addWidget(le_sizes)
        vlay.addLayout(h5)

        self.dw = self.viewer.window.add_dock_widget(ao_control_widget, name="Adaptive Optics", area='right')
        if self.dockWidgetParameter is not None:
            self.viewer.window._qt_window.tabifyDockWidget(self.dockWidgetParameter, self.dw)
        self.dockWidgetParameter = self.dw

        self.zernikeLabel = QLabel(', '.join(f'{v:.2f}' for v in self.zernikeCoeffs))
        self.zernikeLabel.setWordWrap(True)
        self.zernikeLabel.setMinimumWidth(150)
        self.dw = self.viewer.window.add_dock_widget(self.zernikeLabel, name="Zernike Coeffs", area='right')
        if self.dockWidgetParameter is not None:
            self.viewer.window._qt_window.tabifyDockWidget(self.dockWidgetParameter, self.dw)
        self.dockWidgetParameter = self.dw

        self.vWindow.addMainGUI(self.viewer.window._qt_window, name=self.DEFAULT['nameGUI']) 


    def run(self):
        ''' start napari engine '''
        napari.run()

    def _updateDeviceSurface(self):
        self.device.display_surface()
        self.imageLayerDevice.data = self._preprocess_image(self.device.image)

    def setAdaptiveOpticsSequenceser(self,aoSequencer):
        self.aoSequencer=aoSequencer

    def on_sequencer_update(self, params=None):
        ''' callback from the adaptive optics sequencer when a new image is available'''
        print("DMGui: received update from AO sequencer")
        print(f" Params: {params}")
        if params is None:
            return
        zernikecoeff = params.get('current_coefficients', None)
        if zernikecoeff is not None:
            print("DMGui: received new zernike coefficients from AO sequencer")
            print(zernikecoeff)
            self.zernikeCoeffs = zernikecoeff
            self.generateDisplayImage()
            self._updateDeviceSurface()
            self._update_zernike_display()

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

