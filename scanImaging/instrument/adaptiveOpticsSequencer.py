
from viscope.instrument.base.baseSequencer import BaseSequencer
from pathlib import Path
import numpy as np
import keyboard

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
        self.initial_zernike_indices=[5,12,3]
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
        
    def computeOptimalParametersSimpleInterpolation(image_stack,parameter_stack,optim_metric):
        if len(image_stack) != len(parameter_stack) or len(image_stack)<1:
            raise ValueError("Image stack and parameter stack must have the same non-zero length.")
        # take metic for each image in the stack
        metric_values = [optim_metric(img) for img in image_stack]
        # try a simple quadratic interpolation around the maximum
        max_index = np.argmax(metric_values)
        # handle edge cases (max at the boundaries)
        if max_index == 0 or max_index == len(metric_values) - 1:
            optimal_parameter = parameter_stack[max_index]
        else:
            # perform quadratic interpolation
            x0, x1, x2 = parameter_stack[max_index - 1], parameter_stack[max_index], parameter_stack[max_index + 1]
            y0, y1, y2 = metric_values[max_index - 1], metric_values[max_index], metric_values[max_index + 1]
            denom = (y0 - 2 * y1 + y2)
            if denom == 0:
                optimal_parameter = x1
            else:   
                optimal_parameter = x1 + 0.5 * ((y0 - y2) / denom) * (x2 - x0) / 2
        return optimal_parameter

        
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