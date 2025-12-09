
#import sys
# Add the current directory to sys.path
#sys.path.append(r'C:\Users\localxueweikai\Documents\GitHub\scanImaging\scanImaging\instrument\dmc')

#import sys
# Add the current directory to sys.path
#sys.path.append(r'C:\Users\localxueweikai\Documents\GitHub\scanImaging\scanImaging\instrument\dmc')

#import os
#import sys

#if sys.version_info >= (3, 8):
#    os.add_dll_directory(r"C:\Program Files\Boston Micromachines\Bin64")



#import bmc

import os
import importlib
import sys
import numpy as np
from viscope.instrument.base.baseSLM import BaseSLM

def image_from_surface(surface,size_x,size_y):
        ''' Convert a bmc.DoubleVector surface (C++ vector from SWIG) to a 2D numpy array,
            be careful with size_x and size_y and ordering of indices'''
        image=np.zeros((size_x,size_y))
        for i in range(size_x):
            for j in range(size_y):
                image[i,j] = surface[i*size_y + j]
        return image
    
def surface_from_image(image:np.ndarray,bmc):
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
    # library loading management
    # class level variables to ensure DLLs are loaded only once
    dll_path = None       
    _dll_loaded = False
    bmc = None

    @classmethod
    def _load_dll(cls):
        if not sys.version_info >= (3, 8):
            print("Warning: DLL directory not added automatically for Python versions below 3.8. Make sure the DLLs are accessible.")
        else:
            if cls.dll_path and cls.dll_path != ".":
                if not os.path.isdir(cls.dll_path):
                    raise FileNotFoundError(f"DMBMC Error: specified DLL directory {cls.dll_path} does not exist. Use the 'dll_path' argument to specify a different path. ")
                os.add_dll_directory(cls.dll_path)

    @classmethod
    def set_dll_path(cls, path):
        if cls._dll_loaded:
            raise RuntimeError("DLL path must be set before loading the extension module")
        cls.dll_path = path

    @classmethod
    def _ensure_loaded(cls):
        if cls._dll_loaded:
            return

        if cls.dll_path:
            cls._load_dll()

        try:
          cls.bmc = importlib.import_module("bmc.bmc")
        except ImportError as e:
            message_err=f"DMBMC Error: could not import bmc module. Make sure the Boston Micromachines SDK is installed and the DLLs are accessible. Original error: {e}"
            raise ImportError(message_err)
        cls._dll_loaded = True
    
    def __init__(self,name=DEFAULT['name'], **kwargs):
        self.is_connected = False
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
        super().__init__(name=name, **kwargs)

        self.monitor = kwargs['monitor'] if 'monitor' in kwargs else DMBmc.DEFAULT['monitor']

        dll_path = r"C:\Program Files\Boston Micromachines\Bin64"
        if kwargs.get('dll_path'):
            dll_path=kwargs.get('dll_path')
        if not self._dll_loaded:
            self.set_dll_path(dll_path)
        self._ensure_loaded()


        self.matlab_calibration_file="C:\Program Files\Boston Micromachines\Calibration\Sample_Multi_OLC1_CAL.mat"
        if kwargs.get('calibration_file'):
            self.matlab_calibration_file=kwargs.get('calibration_file')

    def connect(self, serial_number='MultiUSB000', **kwargs):
        ''' connect to the instrument '''
        super().connect()
        self.dm = self.bmc.BmcDm()
        if "log_level" in kwargs and "log_file" in kwargs:
            self.dm.configure_log(kwargs["log_file"],kwargs["log_level"])
        self._set_error_code(self.dm.open_dm(serial_number))
        if self.err_code:
            raise Exception(self.dm.error_string(self.err_code))
        self.load_matlab_calibration(self.matlab_calibration_file)
        if self.err_code:
            raise Exception("Calibration file load error "+self.dm.error_string(self.err_code))
        self.active_aperture=0
        self.width = self.dm.num_actuators_width()
        self.surface=self.bmc.DoubleVector(self.width*self.width)
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
        actuator_commands = self.bmc.DoubleVector(self.dm.num_actuators())
        self._set_error_code(self.dm.get_data(actuator_commands))
        return np.array(actuator_commands)    

    def set_active_aperture(self, radius):
#        width = dm.num_actuators_width()
#        recommended_diameter = width - 3 # default (used if set to 0)
#        full_diameter = width - 1
        self.active_aperture = radius

    def _update_surface_from_image(self):
        self.sizeX, self.sizeY = self.image.shape
        self.surface=surface_from_image(self.image,self.bmc)

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
        self.surface=self.bmc.DoubleVector(self.width*self.width)
        # TODO: active_aperature is zero here since it will be handled in dispay_surface. Is this consistent?
        err_code, self.surface = self.dm.zernike_surface(rms_zernike_nm, 0, 0)
        self._set_error_code(err_code)
        self.image=image_from_surface(self.surface,self.width,self.width)


if __name__ == '__main__':
    import time
    dm=DMBmc()
    dm.connect(serial_number='17DW008#094')
    print('BMC report:', dm.report())
    err_code, minPiston, maxPiston = dm.dm.get_segment_range(0, dm.bmc.DM_Piston, 0, 0, 0, True);
    if err_code:
        raise Exception(dm.error_string(err_code))
    print('DM Piston range: %f to %f nm' % (minPiston, maxPiston))
    Zernike_coefficients = [ 0, 0, 0, 0, 0, 0 ]
    w = dm.dm.num_actuators_width()    # 12 for Multi, 13 for Multi-C
    # Add 200nm RMS of defocus
    Zernike_coefficients[4] = 200
    surface = dm.bmc.DoubleVector(w*w)
    err_code, surface = dm.dm.zernike_surface(Zernike_coefficients, 0, 0)
    #print(np.asarray(surface).reshape(w,w))
    if err_code:
        raise Exception(dm.dm.error_string(err_code))
    err_code = dm.dm.set_surface(surface, w, w)
    if err_code:
        raise Exception(dm.dm.error_string(err_code))
    time.sleep(2)

    dm.set_phase_map_from_zernike([0,0,0,0,100])
    dm.display_surface()
    print('BMC report after setting Zernike:', dm.report())
    print('Current error:', dm.report_last_error() )
    import matplotlib.pyplot as plt
    plt.imshow(dm.image)
    plt.show()
    print("Press any key")
    input()
    pass
