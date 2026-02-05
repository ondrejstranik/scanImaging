from viscope.instrument.base.baseSequencer import BaseSequencer
from pathlib import Path
import numpy as np
import keyboard
import time


class ScannerImageProvider:
    ''' class for providing images from the scanner'''

    def __init__(self,scanner=None,processor=None):
        self.scanner_device = scanner
        self.processor = processor
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
        image = self.processor.getAccumulatedImage()
        self.scanner_device.new_image_available = False
        return image

class AdaptiveOpticsSequencer(BaseSequencer):

    DEFAULT = {'name': 'AdaptiveOpticsSequencer'}

    def __init__(self,viscope=None, name=None, **kwargs):
        ''' initialisation '''

        if name== None: name= AdaptiveOpticsSequencer.DEFAULT['name']
        super().__init__(name=name, **kwargs)
        self.viscope=viscope
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
        self.num_steps_per_mode=3
        self.imageSet=[]
        self._dependents = []

    def setInitialZernikeModes(self,indices,initial_coefficients_nm=None,amplitude_scan_nm=None):
        if indices is None or len(indices)==0:
            raise ValueError("Indices for initial Zernike modes must be provided.")
        self.initial_zernike_indices=indices
        if not (len(indices)==len(self.zernike_initial_coefficients_nm) and initial_coefficients_nm is None):
            if initial_coefficients_nm is not None and len(initial_coefficients_nm)==len(indices):
                self.zernike_initial_coefficients_nm=initial_coefficients_nm
            else:
                self.zernike_initial_coefficients_nm=[0]*len(indices)
        if not (len(indices)==len(self.zernike_amplitude_scan_nm) and amplitude_scan_nm is None):
            if amplitude_scan_nm is not None and len(amplitude_scan_nm)==len(indices):
                self.zernike_amplitude_scan_nm=amplitude_scan_nm
            else:
                self.zernike_amplitude_scan_nm=[10]*len(indices)

    def setAmplitudeScan(self,amplitude_scan_nm):
        if amplitude_scan_nm is None or len(amplitude_scan_nm)==0:
            raise ValueError("Amplitude scan values must be provided.")
        self.zernike_amplitude_scan_nm=list(amplitude_scan_nm)
        if len(amplitude_scan_nm)!=len(self.initial_zernike_indices):
            # pad initial zernike indices with default values
            self.initial_zernike_indices=list(range(4,4+len(amplitude_scan_nm)))
            self.zernike_initial_coefficients_nm=[0]*len(amplitude_scan_nm)


    def connect(self,deformable_mirror=None,image_provider=None):
        super().connect()
        if deformable_mirror is not None: self.setParameter('deformable_mirror',deformable_mirror)
        if image_provider is not None: self.setParameter('image_provider',image_provider)

    def setParameter(self,name, value):
        super().setParameter(name,value)
        if name== 'deformable_mirror':
            self.deformable_mirror = value
        if name== 'image_provider':
            self.image_provider = value


    def getParameter(self,name):
        ''' get parameter of the camera '''
        _value = super().getParameter(name)
        if _value is not None: return _value        

        if name== 'deformable_mirror':
            return self.deformable_mirror
        if name== 'image_provider':
            return self.image_provider
        
    # --- Observer / control helpers ---
    def register_dependent(self, dep):
        if not hasattr(self, '_dependents'):
            self._dependents = []
        if dep not in self._dependents:
            self._dependents.append(dep)

    def unregister_dependent(self, dep):
        if hasattr(self, '_dependents') and dep in self._dependents:
            self._dependents.remove(dep)

    def _notify_dependents(self, params=None):
        if not hasattr(self, '_dependents'):
            return
        print("AO Sequencer: notifying dependents...")
        for d in list(self._dependents):
            print(f" Notifying dependent type: {type(d)}, details: {d}")
            d.on_sequencer_update( params=params)
#            try:
#                if hasattr(d, 'on_sequencer_update'):
#                    d.on_sequencer_update(image=image, params=params)
#                elif hasattr(d, 'updateImage'):
#                    d.updateImage()
#                elif hasattr(d, 'updateFromDM'):
#                    d.updateFromDM()
#                elif callable(d):
#                    d(image=image, params=params)
#            except Exception as e:
#                print("Sequencer notify error:", e)
        
    def computeOptimalParametersSimpleInterpolation(self,image_stack,parameter_stack,optim_metric, plot=False):
        if len(image_stack) != len(parameter_stack) or len(image_stack)<3:
            raise ValueError("Image stack and parameter stack must have the same non-zero length.")
        # take metic for each image in the stack
        metric_values = [optim_metric(img) for img in image_stack]
        print("Metric values during scan:", metric_values)
        # try a simple quadratic interpolation around the maximum
        max_index = np.argmax(metric_values)
        miss_max=0
        # handle edge cases (max at the boundaries)
        if max_index == 0:
            miss_max=-1
        if max_index == len(metric_values) - 1:
            miss_max=1
            max_index -= 1  # shift to allow interpolation

        # for tests geenrate a plot of the metric values against parameters
        if plot:
            import matplotlib.pyplot as plt
            plt.figure()
            plt.plot(parameter_stack, metric_values, 'o-')
            plt.xlabel('Parameter value')
            plt.ylabel('Metric value')
            plt.title('Optimization Metric vs Parameter')
            plt.grid(True)
            plt.savefig('optimization_metric_plot.png') 
            plt.close()

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
        return optimal_parameter,miss_max,metric_values[max_index]



    def start(self):
        try:
            self.setParameter('threadingNow', True)
        except Exception:
            pass

    def stop(self):
        try:
            if getattr(self, 'worker', None) is not None:
                self.worker.quit()
        except Exception:
            pass

    def _updateImageAndDM(self,current_coefficients,zernike_index,val):
        self.deformable_mirror.set_phase_map_from_zernike(current_coefficients)
        # notify dependents about intermediate scan image/state
        try:
            self._notify_dependents( params={
                'current_coefficients': current_coefficients.copy(),
                'zernike_index': zernike_index,
                'scan_value': val
            })
        except Exception as e:
            print("Error notifying dependents during scan:", e)
            pass
        self.deformable_mirror.display_surface()
        # wait a bit for the system to stabilize
        time.sleep(0.5)
        image = self.image_provider.getImage()
        return image

    def loop(self):
        ''' main loop of the sequencer '''
        print("Starting Adaptive Optics Sequencer Loop")
        print(f"Adaptint the indices: {self.initial_zernike_indices} with initial coefficients (nm): {self.zernike_initial_coefficients_nm} and scan amplitudes (nm): {self.zernike_amplitude_scan_nm}")
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
        opt_v_set = []
        # set current coefficient
        current_coefficients = initial_coefficients_full.copy()
        ''' finite loop of the sequence'''
        for ii in range(self.optim_iterations):
            # loop over each zernike mode
            for mode_idx, zernike_index in enumerate(self.initial_zernike_indices):
                # prepare scan parameters
                scan_amplitude = self.zernike_amplitude_scan_nm[mode_idx]
                num_scan_points=self.num_steps_per_mode
                scan_values = np.linspace(initial_coefficients_full[zernike_index] - scan_amplitude/2,
                                              initial_coefficients_full[zernike_index] + scan_amplitude/2,
                                              num_scan_points)
                image_stack = []
                parameter_stack = []
                for val in scan_values:
                    current_coefficients[zernike_index] = val
                    # acquire image
                    image_stack.append(self._updateImageAndDM(current_coefficients=current_coefficients,zernike_index=zernike_index,val=val))
                    parameter_stack.append(val)
                    yield  # yield to allow interruption
                miss_max=1
                while num_scan_points<100 and abs(miss_max)!=0:
                    # compute optimal parameter for this mode
                    optimal_value,miss_max,opt_v = self.computeOptimalParametersSimpleInterpolation(
                        image_stack, parameter_stack, optim_metric=lambda img: np.var(img)-np.mean(img),plot=True) #np.sum(np.gradient(img)[0]**2 + np.gradient(img)[1]**2)
                    if miss_max != 0:
                        num_scan_points+=1
                        scan_amplitude *=1.2
                    else:
                        scan_amplitude *= 0.7
                    added_value=None
                    if miss_max<0:
                        added_value=parameter_stack[0]-scan_amplitude/2
                        if optimal_value<parameter_stack[0]:
                            added_value=optimal_value - scan_amplitude/2
                        #append at the beginning
                        parameter_stack=[added_value]+parameter_stack
                    if miss_max>0:
                        added_value=parameter_stack[-1]+scan_amplitude/2
                        if optimal_value>parameter_stack[-1]:
                            added_value=optimal_value + scan_amplitude/2
                        # append at the end
                        parameter_stack=parameter_stack+[added_value]
                    if added_value is not None:
                        current_coefficients[zernike_index] = added_value
                        # acquire image
                        image=self._updateImageAndDM(current_coefficients=current_coefficients,zernike_index=zernike_index,val=added_value)
                        image_stack=[image]+image_stack if miss_max<0 else image_stack+[image]
                        yield  # yield to allow interruption

                initial_coefficients_full[zernike_index] = optimal_value
                current_coefficients[zernike_index] = optimal_value
                self.deformable_mirror.set_phase_map_from_zernike(current_coefficients)
                self.deformable_mirror.display_surface()
                # notify dependents about updated coefficients after optimization of this mode
                try:
                    self._notify_dependents(image=None, params={'coefficients': initial_coefficients_full.copy()})
                except Exception:
                    pass
                print(f"Optimized Zernike index {zernike_index} to value {optimal_value} nm")
            
            image = self.image_provider.getImage()
            imageSet.append(image)
            parameter_set.append(initial_coefficients_full.copy())
            opt_v_set.append(opt_v)
            try:
                self._notify_dependents(params={'final_coefficients': initial_coefficients_full.copy()})
            except Exception as e:
                print("Error notifying dependents after iteration:", e)
                pass

        np.save(self.recorded_image_folder + '/' + 'imageSet',imageSet)
        np.save(self.recorded_image_folder + '/' + 'parameterSet',parameter_set)
        np.save(self.recorded_image_folder + '/' + 'optimalValuesSet',opt_v_set)


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
