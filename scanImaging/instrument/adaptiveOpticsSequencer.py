
from viscope.instrument.base.baseSequencer import BaseSequencer
from pathlib import Path
import numpy as np
import keyboard
import time


class ScannerImageProvider:
    ''' class for providing images from the scanner'''

    def __init__(self):
        self.scanner_device = None
        self.processor = None
        self.failed_acquisitions = 0
        self.timeout_seconds = 120
        self.continous_acquisition = False

    def getImage(self):
        if self.scanner_device is None or self.processor is None:
            raise Exception("Scanner device or processor not set in ScannerImageProvider.")
        self.processor.resetCounter()
        if not self.continous_acquisition:
            self.scanner_device.stopAcquisition()
            self.scanner_device.startAcquisition()
        # wait until the fullAccumulatedImageObtained is True or timeout
        start_time = time.time()
        while not self.processor.fullAccumulatedImageObtained():
            if self.timeout_seconds > 0 and time.time() - start_time > self.timeout_seconds:
                self.failed_acquisitions += 1
                raise TimeoutError("Timeout while waiting for full accumulated image.")
            time.sleep(0.2)
        if not self.continous_acquisition:
            self.scanner_device.stopAcquisition()
        image = self.scanner_device.getLatestImage()
        self.scanner_device.new_image_available = False
        return image

class AdaptiveOpticsSequencer(BaseSequencer):

    DEFAULT = {'name': 'AdaptiveOpticsSequencer'}

    def __init__(self, name=None, **kwargs):
        ''' initialisation '''

        if name== None: name= AdaptiveOpticsSequencer.DEFAULT['name']
        super().__init__(name=name, **kwargs)
        
        # devices
        self.deformable_mirror = None
        self.image_provider = None
        self.dm_display = None
        self.recorded_image_folder= "Data/AdaptiveOptics"
        self.optim_iterations=3
        self.optim_method='simple_interpolation'  # other methods can be implemented
        self.initial_zernike_indices=[4,11,2]  # defocus, astigmatism, spherical
        self.zernike_initial_coefficients_nm=[50,30,20]
        self.zernike_amplitude_scan_nm=[20,15,10]
        self.num_steps_per_mode=5
        self.imageSet=[]


    def connect(self,deformable_mirror=None,image_provider=None,dm_display=None):
        super().connect()
        if deformable_mirror is not None: self.setParameter('camera',deformable_mirror)
        if image_provider is not None: self.setParameter('image_provider',image_provider)
        if dm_display is not None: self.setParameter('dm_display',dm_display)

    def setParameter(self,name, value):
        ''' set parameter of the spectral camera'''
        super().setParameter(name,value)

        if name== 'deformable_mirror':
            self.deformable_mirror = value
        if name== 'image_provider':
            self.image_provider = value
        if name== 'dm_display':
            self.dm_display = value

    def getParameter(self,name):
        ''' get parameter of the camera '''
        _value = super().getParameter(name)
        if _value is not None: return _value        

        if name== 'deformable_mirror':
            return self.deformable_mirror
        if name== 'image_provider':
            return self.image_provider
        if name== 'dm_display':
            return self.dm_display
        
    def computeOptimalParametersSimpleInterpolation(self,image_stack,parameter_stack,optim_metric):
        if len(image_stack) != len(parameter_stack) or len(image_stack)<3:
            raise ValueError("Image stack and parameter stack must have the same non-zero length.")
        # take metic for each image in the stack
        metric_values = [optim_metric(img) for img in image_stack]
        # try a simple quadratic interpolation around the maximum
        max_index = np.argmax(metric_values)
        miss_max=0
        # handle edge cases (max at the boundaries)
        if max_index == 0:
            miss_max=-1
        if max_index == len(metric_values) - 1:
            miss_max=1
            max_index -= 1  # shift to allow interpolation

        # perform quadratic interpolation
        x0, x1, x2 = parameter_stack[max_index - 1], parameter_stack[max_index], parameter_stack[max_index + 1]
        y0, y1, y2 = metric_values[max_index - 1], metric_values[max_index], metric_values[max_index + 1]
        # general formula for the vertex of a parabola given three points (not equally spaced)
        yd1=(y1-y0)
        yd2=(y2-y1)
        xd1=(x1-x0)
        xd2=(x2-x1)
        denom = yd1*xd2-yd2*xd1
#           equally spaced: 
        #denom = (y0 - 2 * y1 + y2)
        if denom == 0:
            optimal_parameter = x1
        else:   
        # equally spaced    
            #   optimal_parameter = x1 + 0.5 * ((y0 - y2) / denom) * (x2 - x0) / 2
            optimal_parameter = x1 + 0.5 * (yd1*xd2*xd2 + yd2*xd1*xd1) / denom
        return optimal_parameter,miss_max

        
    def loop(self):

        # check if the folder exist, if not create it
        p = Path(self.recorded_image_folder)
        p.mkdir(parents=True, exist_ok=True)
        # translate zernike indices into full coefficient array
        initial_coefficients_full = np.zeros(np.max(self.initial_zernike_indices)+1)
        if len(self.initial_zernike_indices) != len(self.zernike_initial_coefficients_nm) or len(self.initial_zernike_indices) != len(self.zernike_amplitude_scan_nm):
            raise ValueError("Length of initial Zernike indices and coefficients must match.")
        for idx, zernike_index in enumerate(self.initial_zernike_indices):
            initial_coefficients_full[zernike_index] = self.zernike_initial_coefficients_nm[idx]
        imageSet = []
        parameter_set = []
        ''' finite loop of the sequence'''
        for ii in range(self.optim_iterations):
            # loop over each zernike mode
            for mode_idx, zernike_index in enumerate(self.initial_zernike_indices):
                # prepare scan parameters
                scan_amplitude = self.zernike_amplitude_scan_nm[mode_idx]
                # scan over the amplitude for this mode
                # symmetric scan around the initial coefficient
                scan_values = np.linspace(initial_coefficients_full[zernike_index] - scan_amplitude/2,
                                          initial_coefficients_full[zernike_index] + scan_amplitude/2,
                                          self.num_steps_per_mode)
                image_stack = []
                parameter_stack = []
                for val in scan_values:
                    # set current coefficient
                    current_coefficients = initial_coefficients_full.copy()
                    current_coefficients[zernike_index] = val
                    self.deformable_mirror.set_phase_map_from_zernike(current_coefficients)
                    self.deformable_mirror.display_surface()
                    # wait a bit for the system to stabilize
                    self.viscope.wait(0.1)
                    # acquire image
                    image = self.image_provider.getImage()
                    image_stack.append(image)
                    parameter_stack.append(val)
                    yield  # yield to allow interruption
                    if keyboard.is_pressed('ctrl+q'):
                        print("Loop aborted")
                        break
                # compute optimal parameter for this mode
                optimal_value = AdaptiveOpticsSequencer.computeOptimalParametersSimpleInterpolation(
                    image_stack, parameter_stack, optim_metric=lambda img: np.mean(img))
                # update initial coefficients
                initial_coefficients_full[zernike_index] = optimal_value
                print(f"Optimized Zernike index {zernike_index} to value {optimal_value} nm")
            
            image = self.image_provider.getImage()
            imageSet.append(image)
            parameter_set.append(initial_coefficients_full.copy())

        np.save(self.recorded_image_folder + '/' + 'imageSet',imageSet)
        np.save(self.recorded_image_folder + '/' + 'parameterSet',parameter_set)


if __name__ == "__main__":
    sequencer = AdaptiveOpticsSequencer()
    f=lambda x: -3.0*(x-1)**2 +2.0
    xv=[0.0,0.5,1.0,1.5,2.0]
    yv=[f(x) for x in xv]
    opt,miss=sequencer.computeOptimalParametersSimpleInterpolation(yv,xv,lambda x: x)
    print("Test completed, optimal value:", opt, " expected: 1.0", " miss:", miss)
    f=lambda x: -3.0*(x-10)**2 +2.0
    yv=[f(x) for x in xv]
    opt,miss=sequencer.computeOptimalParametersSimpleInterpolation(yv,xv,lambda x: x)
    print("Test completed, optimal value:", opt, " expected: 10.0", " miss:", miss)