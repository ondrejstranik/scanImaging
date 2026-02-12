#%%
from viscope.gui.baseGUI import BaseGUI
import napari
#from hmflux.gui.slmViewer import SLMViewer
from magicgui import magicgui
import numpy as np
from qtpy.QtWidgets import QLineEdit, QSizePolicy, QPushButton, QVBoxLayout
from enum import Enum

class Mode(Enum):
    WIDEFIELD = "widefield"
    CONFOCAL = "confocal"
    ISM = "ISM"

class BaseImage(Enum):
    IMAGE1 = "image1"
    IMAGE2 = "image2"
    IMAGE3 = "image3"


class VirtualISMGui(BaseGUI):
    ''' main class to define and view SLM images'''
    DEFAULT = {'nameGUI': 'Virtual ISM GUI',
               'updateNow': 'Update DM Now'
               }

    def __init__(self, viscope, **kwargs):
        ''' initialise the class '''
        super().__init__(viscope, **kwargs)

        self.device=None # virtual ISM device
        self.virtualISM_parameters = {
            'lambda': 520,  # in nm
            'numberOfChannel': 16,
            'NA': 1.4,
            'pixelSize': 5,  # in nm
        }
        self.virtualProbe_parameter={
            "photons_per_pixel": 10, # This parameter is actually a property that needs to be passed to the scanner.
            "background_level": 0.01, # relative to max
            "dark_noise": 0.0, # Poisson to be implemented later
            "read_noise": 0.0, # Gaussian

        }
        self.__setWidget()
    def __setWidget(self):
        """ prepare the gui """

        # -------------------------
        # ISM PARAMETER TAB
        # -------------------------
        @magicgui(call_button='Update ISM Image', layout='vertical')
        def parameter_widget(
            mode: Mode = Mode.WIDEFIELD,
            base_image: BaseImage = BaseImage.IMAGE1,
            lambda_val: float = 520,
            emission_wavelength_val: float = 520,
            na_val: float = 1.4,
            pinhole_diameter_in_AU_val: float = 1.0,
            channels_valx: int = 4,
            channels_valy: int = 4,
            pixel_size_val: float = 5,
            system_aberrations: str = "0, 0, 0",
        ):
            self.virtualISM_parameters['mode'] = mode.value
            self.virtualISM_parameters['baseImage'] = base_image.value
            self.virtualISM_parameters['lambda'] = float(lambda_val)
            self.virtualISM_parameters['emission_wavelength'] = float(emission_wavelength_val)
            self.virtualISM_parameters['NA'] = float(na_val)
            self.virtualISM_parameters['pinhole_diameter_in_AU'] = float(pinhole_diameter_in_AU_val)
            self.virtualISM_parameters['numberOfChannelX'] = int(channels_valx)
            self.virtualISM_parameters['numberOfChannelY'] = int(channels_valy)
            self.virtualISM_parameters['numberOfChannel'] = int(channels_valx * channels_valy)
            self.virtualISM_parameters['pixelSize'] = float(pixel_size_val)

            # Parse system aberrations
            s = str(system_aberrations)
            s = s.replace(';', ' ').replace(',', ' ').replace('[', ' ')\
                .replace(']', ' ').replace('(', ' ').replace(')', ' ')
            parts = [p for p in s.split() if p]
            coeffs = []
            for p in parts:
                try:
                    coeffs.append(float(p))
                except ValueError:
                    pass

            self.virtualISM_parameters['systemAberrations'] = coeffs

            if self.device is not None:
                self.device.setParameter('microscopeType', self.virtualISM_parameters['mode'])
                self.device.setParameter('baseImage', self.virtualISM_parameters['baseImage'])
                self.device.setParameter('lambda', self.virtualISM_parameters['lambda'])
                self.device.setParameter('emissionWavelength', self.virtualISM_parameters['emission_wavelength'])
                self.device.setParameter('numberOfChannelX', self.virtualISM_parameters['numberOfChannelX'])
                self.device.setParameter('numberOfChannelY', self.virtualISM_parameters['numberOfChannelY'])
                self.device.setParameter('NA', self.virtualISM_parameters['NA'])
                self.device.setParameter('pinholeSize', self.virtualISM_parameters['pinhole_diameter_in_AU'])
                self.device.setParameter('pixelSize', self.virtualISM_parameters['pixelSize'])

                try:
                    self.device.setParameter('systemAberrations', coeffs)
                except Exception:
                    pass

                self.device.updateImage()

        # -------------------------
        # PROBE PARAMETER TAB
        # -------------------------
        @magicgui(call_button='Update Probe Parameters', layout='vertical')
        def probe_widget(
            photons_per_pixel: float = 10,
            background_level: float = 0.01,
            dark_noise: float = 0.0,
            read_noise: float = 0.0,
        ):
            self.virtualProbe_parameter['photons_per_pixel'] = float(photons_per_pixel)
            self.virtualProbe_parameter['background_level'] = float(background_level)
            self.virtualProbe_parameter['dark_noise'] = float(dark_noise)
            self.virtualProbe_parameter['read_noise'] = float(read_noise)

            if self.device is not None:
                try:
                    self.device.setProbeParameter(self.virtualProbe_parameter)
                    self.device.updateImage()
                except Exception:
                    pass

        # Store references
        self.parameter_widget = parameter_widget
        self.probe_widget = probe_widget

        # -------------------------
        # ADD TABS
        # -------------------------
        from qtpy.QtWidgets import QTabWidget, QWidget, QVBoxLayout

        tab_widget = QTabWidget()

        # ISM tab
        tab1 = QWidget()
        layout1 = QVBoxLayout()
        layout1.addWidget(self.parameter_widget.native)
        tab1.setLayout(layout1)

        # Probe tab
        tab2 = QWidget()
        layout2 = QVBoxLayout()
        layout2.addWidget(self.probe_widget.native)
        tab2.setLayout(layout2)

        tab_widget.addTab(tab1, "ISM Parameters")
        tab_widget.addTab(tab2, "Probe Parameters")

        self.dw = self.vWindow.addParameterGui(tab_widget, name=self.DEFAULT['nameGUI'])



    def __setWidgetOld(self):
        ''' prepare the gui '''
        @magicgui(call_button='Update ISM Image', layout='vertical')
        def parameter_widget(
            mode: Mode = Mode.WIDEFIELD,
            lambda_val: float = 520,
            emission_wavelength_val: float = 520,
            na_val: float = 1.4,
            pinhole_diameter_in_AU_val: float = 1.0,
            channels_valx: int = 4,
            channels_valy: int = 4,
            pixel_size_val: float = 5,
            system_aberrations: str = "0, 0, 0",
        ):
            self.virtualISM_parameters['mode'] = mode.value
            self.virtualISM_parameters['lambda'] = float(lambda_val)
            self.virtualISM_parameters['emission_wavelength'] = float(emission_wavelength_val)
            self.virtualISM_parameters['NA'] = float(na_val)
            self.virtualISM_parameters['pinhole_diameter_in_AU'] = float(pinhole_diameter_in_AU_val)
            self.virtualISM_parameters['numberOfChannelX'] = int(channels_valx)
            self.virtualISM_parameters['numberOfChannelY'] = int(channels_valy)
            self.virtualISM_parameters['numberOfChannel'] = int(channels_valx * channels_valy)
            self.virtualISM_parameters['pixelSize'] = float(pixel_size_val)
            # parse system aberrations from a textual vector input (e.g. "0, 0.1, -0.02")
            s = str(system_aberrations)
            s = s.replace(';', ' ').replace(',', ' ').replace('[', ' ').replace(']', ' ').replace('(', ' ').replace(')', ' ')
            parts = [p for p in s.split() if p]
            coeffs = []
            for p in parts:
                try:
                    coeffs.append(float(p))
                except ValueError:
                    # ignore non-numeric tokens
                    pass
            self.virtualISM_parameters['systemAberrations'] = coeffs
            if self.device is not None:
                self.device.setParameter('microscopeType', self.virtualISM_parameters['mode'])
                self.device.setParameter('lambda', self.virtualISM_parameters['lambda'])
                self.device.setParameter('emissionWavelength', self.virtualISM_parameters['emission_wavelength'])
                self.device.setParameter('numberOfChannelX', self.virtualISM_parameters['numberOfChannelX'])
                self.device.setParameter('numberOfChannelY', self.virtualISM_parameters['numberOfChannelY'])
                self.device.setParameter('NA', self.virtualISM_parameters['NA'])
                self.device.setParameter('pinholeSize', self.virtualISM_parameters['pinhole_diameter_in_AU'])
                self.device.setParameter('pixelSize', self.virtualISM_parameters['pixelSize'])
                # send parsed Zernike coefficients vector to the device
                try:
                    self.device.setParameter('systemAberrations', self.virtualISM_parameters.get('systemAberrations', []))
                except Exception:
                    pass
                self.device.updateImage()
        self.parameter_widget = parameter_widget
        self.dw = self.vWindow.addParameterGui(self.parameter_widget, name=self.DEFAULT['nameGUI'])
 
    def setDevice(self, device):
        ''' set the device to control '''
        self.device = device


