'''
class for live viewing spectral images
'''
#%%
from viscope.gui.baseGUI import BaseGUI
import napari
#from hmflux.gui.slmViewer import SLMViewer
from magicgui import magicgui
import numpy as np
from qtpy.QtWidgets import QLineEdit, QSizePolicy, QPushButton, QVBoxLayout, QWidget, QLabel, QSpinBox, QHBoxLayout, QComboBox, QCheckBox
from viscope.gui.napariViewer.napariViewer import NapariViewer
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


# Hierarchical optimization preset: recommended iterations per Zernike mode
# Balanced compromise between shallow and deep tissue imaging
# Based on empirical experience and Gibson-Lanni theoretical models
HIERARCHICAL_ITERATIONS = {
    # Tier 1: Defocus - always critical (shallow AND deep)
    4: 6,   # Defocus - easiest to fit, establishes focal reference

    # Tier 2: Astigmatism - always relevant (shallow AND deep)
    3: 5,   # Oblique astigmatism - from tissue/coverslip tilt
    5: 5,   # Vertical astigmatism - from tissue/coverslip tilt

    # Tier 3: Spherical - increasingly important with depth
    11: 4,  # Spherical aberration - critical >30 μm, moderate <30 μm

    # Tier 4: Coma - secondary aberrations
    6: 4,   # Vertical coma - from tilted interfaces
    7: 4,   # Horizontal coma - from tilted interfaces

    # Tier 5: Trefoil and higher-order (use global default)
    # 8: use global default (typically 3)
    # All others: use global optim_iterations value (typically 3)
}


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

        # AO metrics plotting
        self.ao_metric_figure = Figure(figsize=(5, 4), dpi=100)
        self.ao_metric_canvas = FigureCanvas(self.ao_metric_figure)
        self.ao_metric_ax = self.ao_metric_figure.add_subplot(111)
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

        # Save/Load coefficients buttons with simple text input (avoids Qt dialog crashes)
        def _save_coeffs_clicked():
            if self.aoSequencer is None:
                print("No AO sequencer set")
                return
            try:
                from qtpy.QtWidgets import QInputDialog
                filepath, ok = QInputDialog.getText(
                    ao_control_widget,
                    "Save AO Coefficients",
                    "Enter filename (will add .npz):",
                    text="ao_coefficients"
                )
                if ok and filepath:
                    # Add .npz extension if not present
                    if not filepath.endswith('.npz'):
                        filepath = filepath + '.npz'
                    self.aoSequencer.save_coefficients(filepath)
            except Exception as e:
                print(f"Error saving coefficients: {e}")

        def _load_coeffs_clicked():
            if self.aoSequencer is None:
                print("No AO sequencer set")
                return
            try:
                from qtpy.QtWidgets import QInputDialog
                filepath, ok = QInputDialog.getText(
                    ao_control_widget,
                    "Load AO Coefficients",
                    "Enter filename to load:",
                    text="ao_coefficients.npz"
                )
                if ok and filepath:
                    data = self.aoSequencer.load_coefficients(filepath)
                    loaded_coeffs = data['coefficients']

                    # Apply to DM immediately
                    if hasattr(self.aoSequencer, 'apply_coefficients_to_dm'):
                        self.aoSequencer.apply_coefficients_to_dm(loaded_coeffs)

                    # Update GUI display
                    self.zernikeCoeffs = loaded_coeffs.copy()
                    self.generateDisplayImage()
                    self._updateDeviceSurface()
                    self._update_zernike_display()

                    # Update sliders if applicable (for indices 4-11)
                    try:
                        if len(self.zernikeCoeffs) > 11:
                            names = [
                                "zernike_4", "zernike_5", "zernike_6", "zernike_7",
                                "zernike_8", "zernike_9", "zernike_10", "zernike_11"
                            ]
                            for i, name in enumerate(names):
                                idx = i + 4
                                if hasattr(self.zernikeSlidersGui, name):
                                    widget = getattr(self.zernikeSlidersGui, name)
                                    if hasattr(widget, 'value'):
                                        widget.value = float(self.zernikeCoeffs[idx])
                    except Exception as e:
                        print(f"Note: Could not update slider values: {e}")

                    print(f"AO coefficients loaded from {filepath} and applied to GUI")

            except Exception as e:
                print(f"Error loading coefficients: {e}")

        h_saveload = QHBoxLayout()
        btn_save = QPushButton("Save Coeffs", ao_control_widget)
        btn_load = QPushButton("Load Coeffs", ao_control_widget)
        btn_save.clicked.connect(_save_coeffs_clicked)
        btn_load.clicked.connect(_load_coeffs_clicked)
        h_saveload.addWidget(btn_save)
        h_saveload.addWidget(btn_load)
        vlay.addLayout(h_saveload)

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

        # continuous scan checkbox
        h2b = QHBoxLayout()
        lbl_continuous = QLabel("Continuous scan", ao_control_widget)
        chk_continuous = QCheckBox(ao_control_widget)
        chk_continuous.setChecked(getattr(self.aoSequencer, 'continuous_scan', False))
        def _on_continuous(state):
            try:
                if self.aoSequencer is not None:
                    self.aoSequencer.continuous_scan = bool(state)
                    print(f"Continuous scan set to: {bool(state)}")
            except Exception as e:
                print(f"Error setting continuous scan: {e}")
        chk_continuous.stateChanged.connect(_on_continuous)
        h2b.addWidget(lbl_continuous); h2b.addWidget(chk_continuous)
        vlay.addLayout(h2b)

        # use current DM state checkbox
        h2c = QHBoxLayout()
        lbl_use_dm = QLabel("Init from DM state", ao_control_widget)
        chk_use_dm = QCheckBox(ao_control_widget)
        chk_use_dm.setChecked(getattr(self.aoSequencer, 'use_current_dm_state', True))
        def _on_use_dm(state):
            try:
                if self.aoSequencer is not None:
                    self.aoSequencer.use_current_dm_state = bool(state)
                    print(f"Initialize from DM state set to: {bool(state)}")
            except Exception as e:
                print(f"Error setting use_current_dm_state: {e}")
        chk_use_dm.stateChanged.connect(_on_use_dm)
        h2c.addWidget(lbl_use_dm); h2c.addWidget(chk_use_dm)
        vlay.addLayout(h2c)

        # verbose logging checkbox
        h2d = QHBoxLayout()
        lbl_verbose = QLabel("Verbose logging", ao_control_widget)
        chk_verbose = QCheckBox(ao_control_widget)
        chk_verbose.setChecked(getattr(self.aoSequencer, 'verbose', True))
        def _on_verbose(state):
            try:
                if self.aoSequencer is not None:
                    self.aoSequencer.verbose = bool(state)
                    print(f"Verbose logging set to: {bool(state)}")
            except Exception as e:
                print(f"Error setting verbose: {e}")
        chk_verbose.stateChanged.connect(_on_verbose)
        h2d.addWidget(lbl_verbose); h2d.addWidget(chk_verbose)
        vlay.addLayout(h2d)

        # optimization metric selector
        h3b = QHBoxLayout()
        lbl_metric = QLabel("Optimization metric", ao_control_widget)
        cb_metric = QComboBox(ao_control_widget)
        metrics = ['laplacian_variance', 'brenner', 'normalized_variance', 'tenengrad', 'gradient_squared']
        metric_descriptions = {
            'laplacian_variance': 'Laplacian (sensitive)',
            'brenner': 'Brenner (robust)',
            'normalized_variance': 'Variance (simple)',
            'tenengrad': 'Tenengrad (standard)',
            'gradient_squared': 'Gradient (fast)'
        }
        cb_metric.addItems([f"{m} - {metric_descriptions.get(m, '')}" for m in metrics])
        current_metric = getattr(self.aoSequencer, 'selected_metric', 'laplacian_variance')
        try:
            current_idx = metrics.index(current_metric)
            cb_metric.setCurrentIndex(current_idx)
        except (ValueError, AttributeError):
            cb_metric.setCurrentIndex(0)

        def _on_metric(idx):
            try:
                if self.aoSequencer is not None:
                    selected = metrics[idx]
                    self.aoSequencer.selected_metric = selected
                    print(f"Optimization metric set to: {selected}")
            except Exception as e:
                print(f"Error setting metric: {e}")
        cb_metric.currentIndexChanged.connect(_on_metric)
        h3b.addWidget(lbl_metric); h3b.addWidget(cb_metric)
        vlay.addLayout(h3b)

        # optimization algorithm selector
        h3c = QHBoxLayout()
        lbl_algorithm = QLabel("Optimization algorithm", ao_control_widget)
        cb_algorithm = QComboBox(ao_control_widget)
        algorithms = ['simple_interpolation', 'weighted_fit', 'spgd', 'random_search']
        algorithm_descriptions = {
            'simple_interpolation': '3-point fit (fast)',
            'weighted_fit': 'All-points weighted (robust)',
            'spgd': 'SPGD (simultaneous, very robust)',
            'random_search': 'Random search (baseline)'
        }
        cb_algorithm.addItems([f"{alg} - {algorithm_descriptions.get(alg, '')}" for alg in algorithms])
        current_algorithm = getattr(self.aoSequencer, 'optim_method', 'simple_interpolation')
        try:
            current_idx = algorithms.index(current_algorithm)
            cb_algorithm.setCurrentIndex(current_idx)
        except (ValueError, AttributeError):
            cb_algorithm.setCurrentIndex(0)

        def _on_algorithm(idx):
            try:
                if self.aoSequencer is not None:
                    selected = algorithms[idx]
                    self.aoSequencer.optim_method = selected
                    print(f"Optimization algorithm set to: {selected}")
            except Exception as e:
                print(f"Error setting algorithm: {e}")
        cb_algorithm.currentIndexChanged.connect(_on_algorithm)
        h3c.addWidget(lbl_algorithm); h3c.addWidget(cb_algorithm)
        vlay.addLayout(h3c)

        # initial zernike indices (comma-separated)
        h4 = QHBoxLayout()
        lbl_idxs = QLabel("Initial Zernike indices", ao_control_widget)
        le_idxs = QLineEdit(ao_control_widget)
        default_idxs = getattr(self.aoSequencer, 'initial_zernike_indices', [4,3,5,11,6,7])
        le_idxs.setText(','.join(map(str, default_idxs)))

        # Store reference for later access
        self.le_idxs = le_idxs

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
        default_sizes = getattr(self.aoSequencer, 'zernike_amplitude_scan_nm', [80,60,60,200,50,50])
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

        # iterations per mode (hierarchical optimization)
        h6 = QHBoxLayout()
        lbl_iters = QLabel("Iterations per mode", ao_control_widget)
        le_iters = QLineEdit(ao_control_widget)
        default_iters = getattr(self.aoSequencer, 'optim_iterations_per_mode', [])
        le_iters.setText(','.join(map(str, default_iters)) if default_iters else "")
        le_iters.setPlaceholderText("e.g., 12,12,8,8,5 or leave empty for global default")
        def _on_iters():
            txt = le_iters.text().strip()
            try:
                if txt == "":
                    iters = []
                else:
                    parts = [p.strip() for p in txt.replace(';', ',').split(',') if p.strip() != ""]
                    iters = [int(p) for p in parts]
                if self.aoSequencer is not None:
                    self.aoSequencer.optim_iterations_per_mode = iters
                    print(f"Iterations per mode set to: {iters}")
            except Exception as e:
                print(f"Error setting iterations per mode: {e}")
        le_iters.editingFinished.connect(_on_iters)
        h6.addWidget(lbl_iters); h6.addWidget(le_iters)
        vlay.addLayout(h6)

        # hierarchical optimization checkbox
        h7 = QHBoxLayout()
        lbl_hierarchical = QLabel("Use Hierarchical", ao_control_widget)
        chk_hierarchical = QCheckBox(ao_control_widget)
        chk_hierarchical.setChecked(False)
        chk_hierarchical.setToolTip("Auto-set iterations based on mode importance\n"
                                    "Order: Z4(6) → Z3,Z5(5) → Z11(4) → Z6,Z7(4) → others(default)\n"
                                    "Balanced for both shallow and deep tissue imaging")
        def _on_hierarchical(state):
            try:
                if state and self.aoSequencer is not None:
                    # Build hierarchical iterations vector based on current modes
                    indices_txt = le_idxs.text()
                    parts = [p.strip() for p in indices_txt.replace(';', ',').split(',') if p.strip() != ""]
                    modes = [int(p) for p in parts]

                    # Build iterations vector using HIERARCHICAL_ITERATIONS preset
                    iterations = []
                    for mode in modes:
                        mode_iters = HIERARCHICAL_ITERATIONS.get(mode, self.aoSequencer.optim_iterations)
                        iterations.append(mode_iters)

                    # Update the iterations field and sequencer
                    le_iters.setText(','.join(map(str, iterations)))
                    self.aoSequencer.optim_iterations_per_mode = iterations
                    print(f"Hierarchical mode enabled - iterations set to: {iterations}")
                elif not state:
                    # Clear iterations when unchecked
                    le_iters.setText("")
                    if self.aoSequencer is not None:
                        self.aoSequencer.optim_iterations_per_mode = []
                    print("Hierarchical mode disabled - using global iterations")
            except Exception as e:
                print(f"Error toggling hierarchical mode: {e}")
        chk_hierarchical.stateChanged.connect(_on_hierarchical)
        h7.addWidget(lbl_hierarchical); h7.addWidget(chk_hierarchical)
        vlay.addLayout(h7)

        # convergence detection
        h8 = QHBoxLayout()
        lbl_convergence = QLabel("Enable Convergence Detection", ao_control_widget)
        chk_convergence = QCheckBox(ao_control_widget)
        chk_convergence.setChecked(False)
        chk_convergence.setToolTip("Auto-stop when metric improvement < threshold\n"
                                    "Saves time and reduces bleaching")
        def _on_convergence(state):
            try:
                if self.aoSequencer is not None:
                    self.aoSequencer.enable_convergence_detection = bool(state)
                    status = "enabled" if state else "disabled"
                    print(f"Convergence detection {status}")
            except Exception as e:
                print(f"Error toggling convergence detection: {e}")
        chk_convergence.stateChanged.connect(_on_convergence)
        h8.addWidget(lbl_convergence); h8.addWidget(chk_convergence)
        vlay.addLayout(h8)

        # convergence parameters
        h9 = QHBoxLayout()
        lbl_conv_params = QLabel("Convergence (threshold, window)", ao_control_widget)
        le_conv_threshold = QLineEdit(ao_control_widget)
        le_conv_threshold.setText("0.01")
        le_conv_threshold.setToolTip("Relative improvement threshold (e.g., 0.01 = 1%)")
        le_conv_threshold.setMaximumWidth(60)
        le_conv_window = QLineEdit(ao_control_widget)
        le_conv_window.setText("2")
        le_conv_window.setToolTip("Number of iterations to check for improvement")
        le_conv_window.setMaximumWidth(40)

        def _on_conv_threshold():
            try:
                threshold = float(le_conv_threshold.text())
                if self.aoSequencer is not None:
                    self.aoSequencer.convergence_threshold = threshold
                    print(f"Convergence threshold set to: {threshold}")
            except ValueError:
                print("Invalid convergence threshold")
        le_conv_threshold.editingFinished.connect(_on_conv_threshold)

        def _on_conv_window():
            try:
                window = int(le_conv_window.text())
                if self.aoSequencer is not None:
                    self.aoSequencer.convergence_window = window
                    print(f"Convergence window set to: {window}")
            except ValueError:
                print("Invalid convergence window")
        le_conv_window.editingFinished.connect(_on_conv_window)

        h9.addWidget(lbl_conv_params)
        h9.addWidget(le_conv_threshold)
        h9.addWidget(le_conv_window)
        h9.addStretch()
        vlay.addLayout(h9)

        # SPGD parameters
        h10 = QHBoxLayout()
        lbl_spgd = QLabel("SPGD (gain, delta, iters)", ao_control_widget)
        le_spgd_gain = QLineEdit(ao_control_widget)
        le_spgd_gain.setText("0.05")
        le_spgd_gain.setToolTip("Step size for coefficient updates (typically 0.01-0.1)")
        le_spgd_gain.setMaximumWidth(60)

        le_spgd_delta = QLineEdit(ao_control_widget)
        le_spgd_delta.setText("10")
        le_spgd_delta.setToolTip("Perturbation size in nm (typically 5-50)")
        le_spgd_delta.setMaximumWidth(50)

        le_spgd_iters = QLineEdit(ao_control_widget)
        le_spgd_iters.setText("100")
        le_spgd_iters.setToolTip("Number of SPGD iterations")
        le_spgd_iters.setMaximumWidth(60)

        def _on_spgd_gain():
            try:
                gain = float(le_spgd_gain.text())
                if self.aoSequencer is not None:
                    self.aoSequencer.spgd_gain = gain
                    print(f"SPGD gain set to: {gain}")
            except ValueError:
                print("Invalid SPGD gain")
        le_spgd_gain.editingFinished.connect(_on_spgd_gain)

        def _on_spgd_delta():
            try:
                delta = float(le_spgd_delta.text())
                if self.aoSequencer is not None:
                    self.aoSequencer.spgd_delta = delta
                    print(f"SPGD delta set to: {delta}")
            except ValueError:
                print("Invalid SPGD delta")
        le_spgd_delta.editingFinished.connect(_on_spgd_delta)

        def _on_spgd_iters():
            try:
                iters = int(le_spgd_iters.text())
                if self.aoSequencer is not None:
                    self.aoSequencer.spgd_iterations = iters
                    print(f"SPGD iterations set to: {iters}")
            except ValueError:
                print("Invalid SPGD iterations")
        le_spgd_iters.editingFinished.connect(_on_spgd_iters)

        h10.addWidget(lbl_spgd)
        h10.addWidget(le_spgd_gain)
        h10.addWidget(le_spgd_delta)
        h10.addWidget(le_spgd_iters)
        h10.addStretch()
        vlay.addLayout(h10)

        # Random search parameters
        h11 = QHBoxLayout()
        lbl_random = QLabel("Random Search (iters, range)", ao_control_widget)

        le_random_iters = QLineEdit(ao_control_widget)
        le_random_iters.setText("100")
        le_random_iters.setToolTip("Number of random samples")
        le_random_iters.setMaximumWidth(60)

        le_random_range = QLineEdit(ao_control_widget)
        le_random_range.setText("200")
        le_random_range.setToolTip("Search range in nm (±range)")
        le_random_range.setMaximumWidth(60)

        def _on_random_iters():
            try:
                iters = int(le_random_iters.text())
                if self.aoSequencer is not None:
                    self.aoSequencer.random_search_iterations = iters
                    print(f"Random search iterations set to: {iters}")
            except ValueError:
                print("Invalid random search iterations")
        le_random_iters.editingFinished.connect(_on_random_iters)

        def _on_random_range():
            try:
                range_val = float(le_random_range.text())
                if self.aoSequencer is not None:
                    self.aoSequencer.random_search_range = range_val
                    print(f"Random search range set to: {range_val}")
            except ValueError:
                print("Invalid random search range")
        le_random_range.editingFinished.connect(_on_random_range)

        h11.addWidget(lbl_random)
        h11.addWidget(le_random_iters)
        h11.addWidget(le_random_range)
        h11.addStretch()
        vlay.addLayout(h11)

        # Store references for callback access
        self.le_iters = le_iters
        self.chk_hierarchical = chk_hierarchical

        self.dw = self.viewer.window.add_dock_widget(ao_control_widget, name="Adaptive Optics", area='right')
        if self.dockWidgetParameter is not None:
            self.viewer.window._qt_window.tabifyDockWidget(self.dockWidgetParameter, self.dw)
        self.dockWidgetParameter = self.dw

        # Add AO metrics plot widget
        self.ao_metric_ax.set_xlabel('Parameter value')
        self.ao_metric_ax.set_ylabel('Metric value')
        self.ao_metric_ax.set_title('AO Optimization Metric')
        self.ao_metric_ax.grid(True)
        self.ao_metric_figure.tight_layout()

        self.dw = self.viewer.window.add_dock_widget(self.ao_metric_canvas, name="AO Metrics Plot", area='right')
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
        """
        Update GUI visualization of DM surface.

        TWO MODES OF OPERATION:
        1. Manual mode (user adjusting GUI controls):
           - This method updates BOTH hardware and visualization
           - User → GUI → Hardware → Visualization
        2. AO optimization mode (sequencer running):
           - Sequencer already updated hardware
           - This method's display_surface() call is redundant but harmless
           - Sequencer → Hardware → Notification → GUI → (redundant hardware) → Visualization

        The redundant hardware call in mode 2 is intentional - it keeps the code
        simple by avoiding mode-dependent logic. Both modes use the same update path.
        """
        self.device.display_surface()
        self.imageLayerDevice.data = self._preprocess_image(self.device.image)

    def setAdaptiveOpticsSequenceser(self,aoSequencer):
        self.aoSequencer=aoSequencer

    def on_sequencer_update(self, params=None):
        """
        Callback from adaptive optics sequencer when coefficients change (AO mode).

        CONTEXT: This is called ONLY during AO optimization mode.
        During manual mode, GUI updates itself directly without this callback.

        RESPONSIBILITY: Update GUI visualizations in response to sequencer updates.
        Note: The sequencer has already updated the DM hardware before calling this.

        Updates:
        - DM surface plot (via generateDisplayImage + _updateDeviceSurface)
        - Zernike coefficient display (via _update_zernike_display)
        - Convergence plot is handled separately via updateAOMetrics()

        The _updateDeviceSurface() call may redundantly update hardware - this is
        intentional for code simplicity (see _updateDeviceSurface docstring).
        """
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
    
    def updateAOMetrics(self, metric_values, parameter_stack=None, mode='scan', iteration=None):
        '''
        Callback from the adaptive optics sequencer when new metric values are available.

        Parameters:
        -----------
        metric_values : list or array
            Metric values to plot
        parameter_stack : list or array, optional
            Parameter values (for scan-based methods) or iteration numbers (for SPGD)
        mode : str
            'scan' for scan-based methods, 'iteration' for SPGD
        iteration : int, optional
            Current iteration number (for cumulative SPGD plotting)
        '''
        # Get verbose flag (use sequencer's verbose if available)
        verbose = getattr(self.aoSequencer, 'verbose', False) if hasattr(self, 'aoSequencer') and self.aoSequencer else False
        if verbose:
            print("DMGui: received AO metrics update")
            print(f" Mode: {mode}, Metric values: {metric_values}")
            if parameter_stack is not None:
                print(f" Parameter stack: {parameter_stack}")

        # Update the plot
        try:
            print(f"DEBUG: updateAOMetrics called with mode={mode}, len(metric_values)={len(metric_values) if metric_values is not None else 0}")

            if mode == 'scan':
                # Scan-based methods: plot parameter vs metric
                self.ao_metric_ax.clear()
                self.ao_metric_ax.plot(parameter_stack, metric_values, 'o-', linewidth=2, markersize=6)
                self.ao_metric_ax.set_xlabel('Parameter value (nm)')
                self.ao_metric_ax.set_ylabel('Metric value')
                self.ao_metric_ax.set_title('AO Optimization Metric (Scan)')
                self.ao_metric_ax.grid(True, alpha=0.3)

                # Highlight the maximum
                max_idx = np.argmax(metric_values)
                self.ao_metric_ax.plot(parameter_stack[max_idx], metric_values[max_idx],
                                       'r*', markersize=15, label=f'Max: {metric_values[max_idx]:.2f}')
                self.ao_metric_ax.legend()

            elif mode == 'iteration':
                # SPGD: cumulative plot of iterations vs metric
                # Don't clear - accumulate points
                if iteration is not None and len(metric_values) > 0:
                    # Just add the latest point
                    self.ao_metric_ax.plot([iteration], [metric_values[-1]],
                                          'o-', color='blue', markersize=4)
                else:
                    # Full replot
                    self.ao_metric_ax.clear()
                    iterations = list(range(len(metric_values)))
                    self.ao_metric_ax.plot(iterations, metric_values, 'o-', linewidth=2, markersize=4)

                self.ao_metric_ax.set_xlabel('Iteration')
                self.ao_metric_ax.set_ylabel('Metric value')
                self.ao_metric_ax.set_title('AO Optimization Metric (SPGD)')
                self.ao_metric_ax.grid(True, alpha=0.3)

                # Show current best
                if len(metric_values) > 0:
                    best_val = max(metric_values)
                    best_iter = metric_values.index(best_val) if hasattr(metric_values, 'index') else np.argmax(metric_values)
                    self.ao_metric_ax.plot([best_iter], [best_val],
                                          'r*', markersize=15, label=f'Best: {best_val:.2f}')
                    self.ao_metric_ax.legend()

            self.ao_metric_figure.tight_layout()
            self.ao_metric_canvas.draw()
            self.ao_metric_canvas.flush_events()  # Force immediate GUI update
            print(f"DEBUG: Plot updated successfully for mode={mode}")
        except Exception as e:
            print(f"Error updating AO metrics plot: {e}")
            import traceback
            traceback.print_exc()




if __name__ == "__main__":
    pass

