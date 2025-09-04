

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

def image_from_surface(surface,size_x,size_y,image:np.ndarray):
        ''' Convert a bmc.DoubleVector surface (C++ vector from SWIG) to a 2D numpy array,
            be careful with size_x and size_y and ordering of indices'''
        image=np.zeros((size_x,size_y))
        for i in range(size_x):
            for j in range(size_y):
                image[i,j] = surface[i*size_y + j]
        return image
    
def surface_from_image(image:np.ndarray,surface):
        ''' Convert a 2D numpy array to a bmc.DoubleVector surface (C++ vector from SWIG)'''
        size_x,size_y=image.shape
        surface=bmc.DoubleVector(size_x*size_y)
        for i in range(size_x):
            for j in range(size_y):
                surface[i*size_y + j] = image[i,j]
        return surface

class DMBmc(BaseSLM):
    ''' Light modulator class for Boston Micromachines deformable mirror'''
    
    DEFAULT = {'name':'DMBmc',
             'monitor': 1} 
    
    def __init__(self,name=DEFAULT['name'], **kwargs):
        super().__init__(name=name, **kwargs)

        self.monitor = kwargs['monitor'] if 'monitor' in kwargs else DMBmc.DEFAULT['monitor']
        self.dm = None        
        self.image = None
        # For Zernike polynomials, one can specify the radius of the active aperture
        self.active_aperture=0
        self.err_code = None
        self.width = 0
        self.sizeX = 0
        self.sizeY = 0
        # surface in contrast to image, is a cpp array with the phase map
        self.surface=None
        self.is_connected = False

    def connect(self, serial_number='MultiUSB000', **kwargs):
        ''' connect to the instrument '''
        super().connect()
        self.dm = bmc.BmcDm()
        if "log_level" in kwargs and "log_file" in kwargs:
            self.dm.configure_log(kwargs["log_file"],kwargs["log_level"])
        self._set_error_code(self.dm.open_dm(serial_number))
        if self.err_code:
            raise Exception(self.dm.error_string(self.err_code))
        self.active_aperture=0
        self.width = self.dm.num_actuators_width()
        self.surface=bmc.DoubleVector(self.width*self.width)
        self.sizeX, self.sizeY = self.width, self.width
        # set image
        self.setImage(np.zeros((self.sizeY,self.sizeX)))
        self.is_connected = True

    def disconnect(self):
        if not self.is_connected:
            return
        super().disconnect()
        self._set_error_code(self.dm.close_dm())
        self.is_connected = False

    def __del__(self):
        self.disconnect()
    

    def _set_error_code(self, code):
        """Set the error code if there is an error. 
           Keep old error code if no error."""
        if code:
            self.err_code = code

    def get_last_error_code(self):
        return self.err_code
    
    def report_last_error(self):
        if self.err_code:
            return self.dm.error_string(self.get_last_error_code())
        else:
            return "No error."
        
    def report(self):
        return self.dm.error_string(self.dm.get_status())

    def load_matlab_calibration(self, filename):
        self._set_error_code(self.dm.load_calibration_file(filename))


    def update_actuators(self):
         self._set_error_code(self.dm.send_data(self.actuator_array.tolist()))

    def set_actuators(self, actuator_array):
        ''' Set the actuator commands from a numpy array '''
        if len(actuator_array) != self.dm.num_actuators():
            raise ValueError("Actuator array length does not match number of actuators.")
        self._set_error_code(self.dm.send_data(actuator_array.tolist()))

    def get_actuator_commands(self):
        actuator_commands = bmc.DoubleVector(self.dm.num_actuators())
        self._set_error_code(self.dm.get_data(actuator_commands))
        return np.array(actuator_commands)    

    def set_active_aperture(self, radius):
#        width = dm.num_actuators_width()
#        recommended_diameter = width - 3 # default (used if set to 0)
#        full_diameter = width - 1
        self.active_aperture = radius

    def _update_surface_from_image(self):
        self.sizeX, self.sizeY = self.image.shape
        surface_from_image(self.image,self.surface)

    def get_downsampled_surface(self):
        err_code,command_map, downsampled_surface = self.dm.calculate_surface(self.surface, self.sizeX, self.sizeY, self.active_aperture)
        self._set_error_code(err_code)
        return command_map,downsampled_surface

    def display_surface(self):
        ''' Send the current surface to the DM and display it. Convention: other functions (except setImage()) will only update the stored surface, this function will actually send it to the DM.'''
        self._set_error_code(self.dm.set_surface(self.surface, self.sizeX, self.sizeY, self.active_aperture))


    def set_phase_map_nm(self,phase_map_nm:np.ndarray):
        self.image=phase_map_nm
        self._update_surface_from_image()

    def setImage(self,image):
        ''' Alias for consistency with SLM class in other projects, combines set_phase_map_nm and display_surface'''
        self.set_phase_map_nm(image)
        self.display_surface()

    def set_phase_map_from_zernike(self, rms_zernike_nm):
        self.surface=bmc.DoubleVector(self.width*self.width)
        # TODO: active_aperature is zero here since it will be handled in dispay_surface. Is this consistent?
        err_code, self.surface = self.dm.zernike_surface(rms_zernike_nm, 0, 0)
        self._set_error_code(err_code)
        image_from_surface(self.surface,self.width,self.width,self.image)


if __name__ == '__main__':
    dm=DMBmc()
    dm.connect(serial_number='17DW008#094')
    print('BMC report:', dm.report())
    dm.set_phase_map_from_zernike([0,0,0,0,10])
    dm.display_surface()
    print('BMC report after setting Zernike:', dm.report())
    print('Current error:', dm.report_last_error() )
    import matplotlib.pyplot as plt
    plt.imshow(dm.image)
    plt.show()
    pass
