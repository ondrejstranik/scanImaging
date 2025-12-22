# VirtualISM class for simulating an image scanning microscope instrument

import numpy as np

class VirtualISM:
    ''' Virtual Image Scanning Microscope instrument for testing and simulation '''

    DEFAULT = {'name': 'VirtualISM'}

    def __init__(self, name=None, **kwargs):
        ''' initialisation '''
        if name is None:
            name = VirtualISM.DEFAULT['name']
        self.name = name
        self.parameters = {}
        self.is_connected = False
        self.base_image = kwargs.get('base_image', np.zeros((256, 256)))
        self.virtualScanner = kwargs.get('virtualScanner', None)
        self.virtualAdaptiveOptics = kwargs.get('virtualAdaptiveOptics', None)
        self.parameters['lambda'] = kwargs.get('wavelength', 640)  # in nm

    def connect(self, virtualScanner=None,virtualAdaptiveOptics=None):
        ''' simulate connecting to the virtual ISM instrument '''
        if virtualScanner is not None:
            self.virtualScanner = virtualScanner
            if hasattr(self.virtualScanner, 'virtualProbe'):
                self.base_image = self.virtualScanner.virtualProbe.reshape(self.virtualScanner.imageSize)
        if virtualAdaptiveOptics is not None:
            self.virtualAdaptiveOptics = virtualAdaptiveOptics
        self.is_connected = True
        print(f"{self.name} connected.")

    def updateImage(self):
        '''Calculate the ISM image based on the current base image and virtual scanner settings'''
        if (not self.is_connected) or self.virtualScanner is None or self.virtualAdaptiveOptics is None:
            raise Exception("VirtualISM is not connected.")
        # For simplicity, we just return the base image modified by a factor
        # In a real implementation, this would involve complex calculations
        wavelength = self.getParameter('lambda')
        ism_image = self.base_image * (wavelength / 640)  # Simple scaling
        self.virtualScanner.setVirtualProbe(ism_image)
        return ism_image

    def disconnect(self):
        ''' simulate disconnecting from the virtual ISM instrument '''
        self.is_connected = False
        print(f"{self.name} disconnected.")

    def setParameter(self, param_name, param_value):
        ''' set a parameter for the virtual ISM '''
        self.parameters[param_name] = param_value
        print(f"Parameter {param_name} set to {param_value}.")

    def getParameter(self, param_name):
        ''' get a parameter value '''
        return self.parameters.get(param_name, None)