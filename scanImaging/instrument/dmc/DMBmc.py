

#import bmc
import sys
# Add the current directory to sys.path
sys.path.append(r'C:\Users\localxueweikai\Documents\GitHub\scanImaging\scanImaging\instrument\dmc')

import os
import sys

if sys.version_info >= (3, 8):
    os.add_dll_directory(r"C:\Program Files\Boston Micromachines\Bin64")

import bmc
import numpy as np
from viscope.instrument.base.baseSLM import BaseSLM

class DMBmc(BaseSLM):
    ''' Light modulator class for Boston Micromachines deformable mirror'''
    
    DEFAULT = {'name':'DMBmc',
             'monitor': 1} 
    
    def __init__(self,name=DEFAULT['name'], **kwargs):
        super().__init__(name=name, **kwargs)

        self.monitor = kwargs['monitor'] if 'monitor' in kwargs else DMBmc.DEFAULT['monitor']
        self.dm = None        
        self.image = None
        # this is an array of actuator values. Min=0, Max=1.
        self.actuator_array = None
        # For Zernike polynomials, one can specify the radius of the active aperture
        self.active_aperture=0
        self.err_code = None
        self.width = 0
        self.sizeX = 0
        self.sizeY = 0
        # surface in contrast to image, is a cpp array with the phase map
        self.surface=None
        self.downsampled_surface=None

    def connect(self, serial_number='MultiUSB000', **kwargs):
        ''' connect to the instrument '''
        super().connect()
        self.init(serial_number=serial_number)
        self.sizeX, self.sizeY = self.width, self.width
        # set image
        self.setImage(np.zeros((self.sizeY,self.sizeX)))

    def disconnect(self):
        super().disconnect()
        self.close()
    
    def setImage(self,image):
        self.image=image
        self.set_surface_from_image()
        self.display_surface()

# The rest is specific functionality for the Boston Micromachines deformable mirror
    def _set_error_code(self, code):
        """Set the error code if there is an error. 
           Keep old error code if no error."""
        if code:
            self.err_code = code
    def has_error(self):
        return self.err_code
    
    def report_error(self):
        if self.err_code:
            return self.dm.error_string(self.err_code)
        else:
            return "No error."
        
    def init(self, serial_number='MultiUSB000'):
        self.dm = bmc.BmcDm()
        self._set_error_code(self.dm.open_dm(serial_number))
        if self.err_code:
            raise Exception(self.dm.error_string(self.err_code))
        self.actuator_array = np.zeros(self.dm.num_actuators(), dtype=float)
        self.active_aperture=0
        self.width = self.dm.num_actuators_width()
        self.surface=bmc.DoubleVector(self.width*self.width)

    def close(self):
        self._set_error_code(self.dm.close_dm())

    def __del__(self):
        self.close()

    def load_matlab_calibration(self, filename):
        self._set_error_code(self.dm.load_calibration_file(filename))

    def zero_actuators(self):
        self.actuator_array.fill(0.0)
        self.update_actuators()

    def update_actuators(self):
         self._set_error_code(self.dm.send_data(self.actuator_array.tolist()))

    def set_actuators(self, actuator_array):
        if len(actuator_array) != self.dm.num_actuators():
            raise ValueError("Actuator array length does not match number of actuators.")
        self.actuator_array = actuator_array
        self.update_actuators()

    def set_active_aperture(self, radius):
#        width = dm.num_actuators_width()
#        recommended_diameter = width - 3 # default (used if set to 0)
#        full_diameter = width - 1
        self.active_aperture = radius

    def set_surface_from_image(self):
        self.sizeX, self.sizeY = self.image.shape
        # convert it to the C++ type
        surface = bmc.DoubleVector(self.sizeX*self.sizeY)
        for i in range(self.sizeX):
            for j in range(self.sizeY):
                surface[i*self.sizeY + j] = self.image[i,j]

    def get_downsampled_surface(self):
        err_code,command_map, downsampled_surface = self.dm.calculate_surface(self.surface, self.sizeX, self.sizeY, self.active_aperture)
        self._set_error_code(err_code)
        return command_map,downsampled_surface

    def display_surface(self):
        self._set_error_code(self.dm.set_surface(self.surface, self.sizeX, self.sizeY, self.active_aperture))

    def set_phase_map_nm(self,surface_phase_nm):
        self.image=surface_phase_nm
        self.set_surface_from_image()
        self.display_surface()

    def report_status(self):
        return self.dm.error_string(self.dm.get_status())

    def set_zernike(self, rms_zernike_nm):
        self.surface=bmc.DoubleVector(self.width*self.width)
        err_code, self.surface = self.dm.zernike_surface(rms_zernike_nm, self.active_aperture, 0)
        self._set_error_code(err_code)
        self.image=np.zeros((self.width,self.width))
        for i in range(self.width):
            for j in range(self.width):
                self.image[i,j] = self.surface[i*self.width + j]
        self._set_error_code(self.dm.set_surface(self.surface, self.sizeX, self.sizeY, 0))


if __name__ == '__main__':
    dm=DMBmc()
    dm.connect(serial_number='17DW008#094')
    print('BMC error status:', dm.report_status())
    print(dm.get_downsampled_surface())
    dm.set_zernike([0,0,0,0,10])
    import matplotlib.pyplot as plt
    plt.imshow(dm.image)
    plt.show()
    pass
