


import numpy as np
from viscope.instrument.base.baseSLM import BaseSLM
import scanImaging.algorithm.SimpleZernike as simzern

def image_from_surface(surface,size_x,size_y):
        ''' Convert a bmc.DoubleVector surface (C++ vector from SWIG) to a 2D numpy array,
            be careful with size_x and size_y and ordering of indices'''
        image=np.zeros((size_x,size_y))
        for i in range(size_x):
            for j in range(size_y):
                image[i,j] = surface[i*size_y + j]
        return image

def FakeDoubleVector(size):
    return np.zeros(size,np.float64)
    
def surface_from_image(image:np.ndarray):
        ''' Convert a 2D numpy array to a bmc.DoubleVector surface (C++ vector from SWIG)'''
        size_x,size_y=image.shape
        surface=FakeDoubleVector(size_x*size_y)
        for i in range(size_x):
            for j in range(size_y):
                surface[i*size_y + j] = image[i,j]
        return surface

class VirtualDMBmc(BaseSLM):
    ''' Light modulator class for Boston Micromachines deformable mirror'''
    
    DEFAULT = {'name':'VirtualDMBmc',
             'monitor': 1} 
    
    def __init__(self,name=DEFAULT['name'], **kwargs):
        super().__init__(name=name, **kwargs)

        self.monitor = kwargs['monitor'] if 'monitor' in kwargs else VirtualDMBmc.DEFAULT['monitor']
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
        self.serial_number = None
        self.actuators=None
        self.is_connected = False

    def connect(self, serial_number='MultiUSB000', **kwargs):
        ''' connect to the instrument '''
        super().connect()
        self.serial_number = serial_number
        self.active_aperture=0
        self.width = 12
        self.surface=FakeDoubleVector(self.width*self.width)
        self.sizeX, self.sizeY = self.width, self.width
        # set image
        self.setImage(np.zeros((self.sizeY,self.sizeX)))
        self.is_connected = True

    def disconnect(self):
        if not self.is_connected:
            return
        super().disconnect()
        self._set_error_code(None)
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
            return f"Error code of VirtualDMBmc {self.err_code}."
        else:
            return "No error."
        
    def report(self):
        return f"VirtualDMBmc connected: {self.is_connected}, serial number: {self.serial_number}, width: {self.width}, active aperture: {self.active_aperture}, error code: {self.err_code}"

    def load_matlab_calibration(self, filename):
        print(f'VirtualDMBmc: load_matlab_calibration from {filename} (not implemented)')


    def update_actuators(self):
         print('VirtualDMBmc: update_actuators (not implemented)')

    def set_actuators(self, actuator_array):
        self.actuators = actuator_array

    def get_actuator_commands(self):
        return self.actuators

    def set_active_aperture(self, radius):
#        width = dm.num_actuators_width()
#        recommended_diameter = width - 3 # default (used if set to 0)
#        full_diameter = width - 1
        self.active_aperture = radius

    def _update_surface_from_image(self):
        self.sizeX, self.sizeY = self.image.shape
        self.surface=surface_from_image(self.image)

    def get_downsampled_surface(self):
        return self.actuators,self.surface

    def display_surface(self):
        print('VirtualDMBmc: display_surface (not implemented)')


    def set_phase_map_nm(self,phase_map_nm:np.ndarray):
        self.image=phase_map_nm
        self._update_surface_from_image()

    def setImage(self,image):
        ''' Alias for consistency with SLM class in other projects, combines set_phase_map_nm and display_surface'''
        self.set_phase_map_nm(image)
        self.display_surface()

    def set_phase_map_from_zernike(self, rms_zernike_nm):
        self.surface=FakeDoubleVector(self.width*self.width)
        self.image=simzern.zernike_phase_map(rms_zernike_nm,self.width,self.width,self.active_aperture)
        self._update_surface_from_image()


if __name__ == '__main__':
    dm=VirtualDMBmc()
    dm.connect(serial_number='17DW008#094')
    print('BMC report:', dm.report())
    dm.set_phase_map_from_zernike([0,0,0,0,10])
    dm.display_surface()
    print('BMC report after setting Zernike:', dm.report())
    print('Current error:', dm.report_last_error() )
    import matplotlib.pyplot as plt
    plt.imshow(dm.image)
    plt.show()
    print(dm.image)
    pass
