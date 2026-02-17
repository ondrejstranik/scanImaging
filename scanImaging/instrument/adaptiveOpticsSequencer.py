from viscope.instrument.base.baseSequencer import BaseSequencer
from pathlib import Path
import numpy as np
import time


class ScannerImageProvider:
    ''' class for providing images from the scanner'''

    def __init__(self, scanner=None, processor=None,
                 timeout=30,              # Reduced from 120s
                 max_retries=3,           # New: retry on timeout
                 enable_health_check=False,  # New: configurable (OFF by default)
                 enable_auto_restart=True):  # New: auto-restart on failure
        self.scanner_device = scanner
        self.processor = processor
        self.timeout_seconds = timeout
        self.max_retries = max_retries
        self.enable_health_check = enable_health_check
        self.enable_auto_restart = enable_auto_restart
        self.failed_acquisitions = 0
        self.total_acquisitions = 0
        self.continous_acquisition = False

    def startContinuousMode(self):
        self.continous_acquisition = True
        self.scanner_device.startAcquisition()

    def stopContinuousMode(self):
        self.scanner_device.stopAcquisition()
        self.continous_acquisition = False
    
    def restartContinuousMode(self):
        self.stopContinuousMode()
        time.sleep(0.5)
        self.startContinuousMode()

    def health_check(self, timeout=5):
        """Optional health check before acquisition (may bleach sample if enabled)

        Args:
            timeout: Timeout in seconds for health check

        Returns:
            bool: True if scanner appears healthy, False otherwise
        """
        if not self.enable_health_check:
            return True  # Skip if disabled

        print("Performing scanner health check...")
        try:
            # Quick test acquisition with short timeout
            test_img = self.getImage(timeout=timeout, skip_health_check=True)
            if test_img is None or np.sum(test_img) < 1e-6:
                print("WARNING: Health check failed - no signal detected")
                return False
            print("Scanner health check passed")
            return True
        except TimeoutError:
            print("WARNING: Health check timeout - scanner may be stalled")
            return False
        except Exception as e:
            print(f"WARNING: Health check error: {e}")
            return False

    def restart_scanner(self):
        """Restart scanner hardware to recover from stalled state

        Returns:
            bool: True if restart successful, False otherwise
        """
        print("Restarting scanner...")
        try:
            # Stop acquisition
            self.scanner_device.stopAcquisition()
            time.sleep(0.5)

            # Reset processor state if method exists
            if hasattr(self.processor, 'reset_accumulation'):
                self.processor.reset_accumulation()
            else:
                # Fallback: use existing resetCounter method
                self.processor.resetCounter()
            time.sleep(0.5)

            # Restart acquisition
            self.scanner_device.startAcquisition()
            time.sleep(1.0)  # Allow scanner to stabilize

            print("Scanner restarted successfully")
            return True
        except Exception as e:
            print(f"ERROR: Scanner restart failed: {e}")
            return False

    def get_statistics(self):
        """Return acquisition statistics for monitoring

        Returns:
            dict: Statistics including total, failed, and success rate
        """
        success_rate = 1.0
        if self.total_acquisitions > 0:
            failed_count = self.failed_acquisitions
            success_rate = 1.0 - (failed_count / self.total_acquisitions)
        return {
            'total': self.total_acquisitions,
            'failed': self.failed_acquisitions,
            'success_rate': success_rate
        }

    def getImage(self, timeout=None, skip_health_check=False):
        """Get accumulated image with retry logic

        Args:
            timeout: Override default timeout (seconds)
            skip_health_check: Skip health check even if enabled (for health check itself)

        Returns:
            numpy.ndarray: Accumulated image

        Raises:
            TimeoutError: If all retry attempts fail
        """
        if self.scanner_device is None or self.processor is None:
            raise Exception("Scanner device or processor not set in ScannerImageProvider.")

        if timeout is None:
            timeout = self.timeout_seconds

        self.total_acquisitions += 1
        last_exception = None

        # Retry loop
        for attempt in range(self.max_retries):
            try:
                # Reset processor counter
                self.processor.resetCounter()

                # Start acquisition if not in continuous mode
                if not self.continous_acquisition:
                    self.scanner_device.stopAcquisition()
                    self.scanner_device.startAcquisition()

                # Wait until full accumulated image obtained or timeout
                start_time = time.time()
                while not self.processor.fullAccumulatedImageObtained():
                    if timeout > 0 and time.time() - start_time > timeout:
                        raise TimeoutError(
                            f"Timeout waiting for image (attempt {attempt+1}/{self.max_retries}, "
                            f"{timeout}s timeout)")
                    time.sleep(0.2)

                # Stop acquisition if not in continuous mode
                if not self.continous_acquisition:
                    self.scanner_device.stopAcquisition()

                # Get the image
                image = self.processor.getAccumulatedImage()
                self.scanner_device.new_image_available = False

                # Success - reset failure counter if we had previous failures
                if self.failed_acquisitions > 0:
                    print(f"Acquisition recovered after {self.failed_acquisitions} consecutive failures")
                    self.failed_acquisitions = 0

                return image

            except TimeoutError as e:
                last_exception = e
                self.failed_acquisitions += 1
                print(f"Acquisition timeout: {e}")

                # Check if we should retry
                if attempt < self.max_retries - 1:
                    print(f"Retrying... ({attempt+2}/{self.max_retries})")

                    # Try automatic restart if enabled
                    if self.enable_auto_restart and not skip_health_check:
                        if self.restart_scanner():
                            # Wait a bit before retry
                            time.sleep(0.5)
                            continue
                        else:
                            print("Scanner restart failed, trying again anyway...")

                    # Reset processor and retry even if restart failed
                    if hasattr(self.processor, 'reset_accumulation'):
                        self.processor.reset_accumulation()
                    time.sleep(0.5)
                else:
                    # All retries exhausted
                    print(f"ERROR: Image acquisition failed after {self.max_retries} attempts")
                    raise

            except Exception as e:
                # Unexpected error - don't retry, just fail
                self.failed_acquisitions += 1
                print(f"ERROR: Unexpected error during image acquisition: {e}")
                raise

        # Should not reach here, but just in case
        if last_exception:
            raise last_exception
        return None

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
        self.initial_zernike_indices=[4,5,6]  # defocus, astigmatism, spherical
        self.zernike_initial_coefficients_nm=[0,0,0]
        self.zernike_amplitude_scan_nm=[20,15,10]
        self.num_steps_per_mode=3
        self.imageSet=[]
        self._dependents = []
        self.continuous_scan=False
        self.image_log=False
        self.print_plot=False
        self.parameter_stack=None
        self.metric_values=None
        self.use_current_dm_state=True  # Initialize from current DM coefficients (default: preserve existing state)
        self.final_coefficients=None  # Will store final optimized coefficients after AO loop completes

        # Logging control
        self.verbose=True  # Set to False to reduce logging output

        # Scanner robustness settings
        self.scanner_timeout = 30  # seconds (reduced from 120s for faster feedback)
        self.scanner_max_retries = 5  # number of retry attempts on timeout
        self.scanner_enable_health_check = False  # OFF by default to avoid bleaching
        self.scanner_enable_auto_restart = True  # automatically restart scanner on failure

        # Optimization metrics
        self.selected_metric='laplacian_variance'
        self._setup_metrics()

    def _setup_metrics(self):
        """Initialize dictionary of available optimization metrics"""
        self.metric_functions = {
            'laplacian_variance': self._metric_laplacian_variance,
            'brenner': self._metric_brenner,
            'normalized_variance': self._metric_normalized_variance,
            'tenengrad': self._metric_tenengrad,
            'gradient_squared': self._metric_gradient_squared,
        }

    def _metric_laplacian_variance(self, img):
        """Laplacian variance - sensitive to focus, but noise-sensitive"""
        from scipy.ndimage import laplace
        return np.mean(laplace(img)**2) / (np.mean(img) + 1e-12)

    def _metric_brenner(self, img):
        """Brenner gradient - more robust to Poisson noise"""
        # Vertical and horizontal gradients with 2-pixel spacing
        grad_v = np.sum((img[2:, :] - img[:-2, :])**2)
        grad_h = np.sum((img[:, 2:] - img[:, :-2])**2)
        return (grad_v + grad_h) / (np.sum(img) + 1e-12)

    def _metric_normalized_variance(self, img):
        """Normalized variance - simple, less sensitive to focus but robust"""
        return np.var(img) / (np.mean(img) + 1e-12)

    def _metric_tenengrad(self, img):
        """Tenengrad (Sobel-based) - standard autofocus metric"""
        from scipy.ndimage import sobel
        grad_x = sobel(img, axis=1)
        grad_y = sobel(img, axis=0)
        return np.mean(grad_x**2 + grad_y**2)

    def _metric_gradient_squared(self, img):
        """Sum of squared gradients - simpler than Tenengrad"""
        grad_v = np.sum(np.diff(img, axis=0)**2)
        grad_h = np.sum(np.diff(img, axis=1)**2)
        return (grad_v + grad_h) / (np.sum(img) + 1e-12)

    def get_metric_function(self):
        """Get the currently selected metric function"""
        return self.metric_functions.get(self.selected_metric, self._metric_laplacian_variance)

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
            # Apply scanner robustness configuration to image provider
            if self.image_provider is not None and isinstance(self.image_provider, ScannerImageProvider):
                self.image_provider.timeout_seconds = self.scanner_timeout
                self.image_provider.max_retries = self.scanner_max_retries
                self.image_provider.enable_health_check = self.scanner_enable_health_check
                self.image_provider.enable_auto_restart = self.scanner_enable_auto_restart


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
        if self.verbose:
            print(f"AO Sequencer: notifying {len(self._dependents)} dependent(s)")
        for d in list(self._dependents):
            d.on_sequencer_update( params=params)
            if hasattr(d, 'updateAOMetrics') and self.parameter_stack is not None and self.metric_values is not None:
                d.updateAOMetrics(self.metric_values,self.parameter_stack)

    # --- Save/Load Coefficients ---
    def save_coefficients(self, filepath):
        """Save optimized Zernike coefficients to file

        Args:
            filepath: Path to save coefficients (will add .npz if not present)
        """
        from datetime import datetime

        # Priority order for getting coefficients:
        # 1. Final optimized coefficients from last AO run
        # 2. Current coefficients from DM (if method exists)
        # 3. Initial coefficients as fallback
        if hasattr(self, 'final_coefficients') and self.final_coefficients is not None:
            coefficients = self.final_coefficients.copy()
            print(f"Saving final optimized coefficients from AO loop")
        elif self.deformable_mirror is not None and hasattr(self.deformable_mirror, 'get_current_coefficients'):
            coefficients = self.deformable_mirror.get_current_coefficients()
            print(f"Saving current coefficients from DM")
        else:
            # Fallback to stored initial coefficients (warning: these are NOT optimized!)
            coefficients = np.zeros(max(self.initial_zernike_indices) + 1)
            for idx, zernike_index in enumerate(self.initial_zernike_indices):
                coefficients[zernike_index] = self.zernike_initial_coefficients_nm[idx]
            print(f"Warning: Saving initial (not optimized) coefficients - run AO optimization first!")

        # Ensure .npz extension
        if not filepath.endswith('.npz'):
            filepath += '.npz'

        # Save with metadata
        np.savez(filepath,
                 coefficients=coefficients,
                 zernike_indices=self.initial_zernike_indices,
                 timestamp=datetime.now().isoformat(),
                 metric_used=self.selected_metric,
                 iterations=self.optim_iterations)

        print(f"Saved AO coefficients to: {filepath}")
        print(f"  Coefficients (first 12): {coefficients[:min(12, len(coefficients))]}")
        return filepath

    def load_coefficients(self, filepath):
        """Load Zernike coefficients from file and optionally apply to DM

        Args:
            filepath: Path to load coefficients from

        Returns:
            dict: Loaded data including coefficients and metadata
        """
        # Load data
        data = np.load(filepath, allow_pickle=True)
        coefficients = data['coefficients']

        # Update initial coefficients to match loaded values
        if 'zernike_indices' in data:
            loaded_indices = data['zernike_indices']
            # Update coefficients for loaded indices
            for idx, zernike_index in enumerate(loaded_indices):
                if idx < len(self.initial_zernike_indices):
                    self.zernike_initial_coefficients_nm[idx] = coefficients[zernike_index]

        # Print metadata if available
        if 'timestamp' in data:
            print(f"Loaded coefficients from: {data['timestamp']}")
        if 'metric_used' in data:
            print(f"  Metric used: {data['metric_used']}")
        if 'iterations' in data:
            print(f"  Iterations: {data['iterations']}")

        print(f"Loaded AO coefficients from: {filepath}")
        print(f"Coefficients shape: {coefficients.shape}")

        return {
            'coefficients': coefficients,
            'metadata': {k: data[k] for k in data.files if k != 'coefficients'}
        }

    def apply_coefficients_to_dm(self, coefficients):
        """Apply given coefficients to the deformable mirror

        Args:
            coefficients: Array of Zernike coefficients to apply
        """
        if self.deformable_mirror is None:
            print("Warning: No deformable mirror connected")
            return

        self.deformable_mirror.set_phase_map_from_zernike(coefficients)
        self.deformable_mirror.display_surface()
        print(f"Applied coefficients to DM: {coefficients[:min(12, len(coefficients))]}")

    def computeOptimalParametersSimpleInterpolation(self,image_stack,parameter_stack,optim_metric, plot=False):
        if len(image_stack) != len(parameter_stack) or len(image_stack)<3:
            raise ValueError("Image stack and parameter stack must have the same non-zero length.")
        # take metic for each image in the stack
        metric_values = [optim_metric(img) for img in image_stack]
        if self.verbose:
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
        self.parameter_stack=parameter_stack
        self.metric_values=metric_values
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

    def _initialize_coefficients(self):
        """Initialize Zernike coefficients from current DM state or specified initial values

        Returns:
            np.ndarray: Initial coefficient array for optimization
        """
        # Validate input arrays
        if len(self.initial_zernike_indices) != len(self.zernike_initial_coefficients_nm) or \
           len(self.initial_zernike_indices) != len(self.zernike_amplitude_scan_nm):
            raise ValueError("Length of initial Zernike indices and coefficients must match.")

        # Initialize from current DM state if requested
        if self.use_current_dm_state and self.deformable_mirror is not None:
            try:
                if hasattr(self.deformable_mirror, 'get_current_coefficients'):
                    current_dm_coeffs = self.deformable_mirror.get_current_coefficients()

                    # Check if DM returned valid coefficients
                    if current_dm_coeffs is None or len(current_dm_coeffs) == 0:
                        if self.verbose:
                            print("Warning: DM returned empty coefficients, falling back to initial values")
                        return self._create_initial_coefficient_array()

                    # Create array large enough for both DM coeffs and optimization indices
                    required_size = max(np.max(self.initial_zernike_indices)+1, len(current_dm_coeffs))
                    initial_coefficients_full = np.zeros(required_size)
                    initial_coefficients_full[:len(current_dm_coeffs)] = current_dm_coeffs

                    if self.verbose:
                        print(f"Initialized from current DM state: {current_dm_coeffs[:min(12, len(current_dm_coeffs))]}")
                    return initial_coefficients_full

                else:
                    # DM doesn't support get_current_coefficients
                    if self.verbose:
                        print("Warning: DM doesn't support get_current_coefficients(), starting from specified values")
                    return self._create_initial_coefficient_array()

            except Exception as e:
                if self.verbose:
                    print(f"Error reading DM state: {e}, falling back to initial values")
                return self._create_initial_coefficient_array()
        else:
            # Use specified initial coefficients
            if self.verbose:
                print(f"Initialized from specified values: {self.zernike_initial_coefficients_nm}")
            return self._create_initial_coefficient_array()

    def _create_initial_coefficient_array(self):
        """Create coefficient array from specified initial values

        Returns:
            np.ndarray: Coefficient array with specified initial values
        """
        initial_coefficients_full = np.zeros(np.max(self.initial_zernike_indices)+1)
        for idx, zernike_index in enumerate(self.initial_zernike_indices):
            initial_coefficients_full[zernike_index] = self.zernike_initial_coefficients_nm[idx]
        return initial_coefficients_full

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
        time.sleep(0.1)
        image = self.image_provider.getImage()
        return image

    def loop(self):
        ''' main loop of the sequencer '''
        if self.verbose:
            print("Starting Adaptive Optics Sequencer Loop")
            print(f"Optimizing indices: {self.initial_zernike_indices} with initial coefficients (nm): {self.zernike_initial_coefficients_nm} and scan amplitudes (nm): {self.zernike_amplitude_scan_nm}")
            print(f"Using metric: {self.selected_metric}")

        # check if the folder exist, if not create it
        if self.image_log:
            p = Path(self.recorded_image_folder)
            p.mkdir(parents=True, exist_ok=True)

        # Initialize coefficients using refactored method
        initial_coefficients_full = self._initialize_coefficients()
        imageSet = []
        parameter_set = []
        opt_v_set = []
        # set current coefficient
        if self.continuous_scan:
            self.image_provider.startContinuousMode()
        current_coefficients = initial_coefficients_full.copy()
        try:
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
                        metric_fn = self.get_metric_function()
                        optimal_value,miss_max,opt_v = self.computeOptimalParametersSimpleInterpolation(
                            image_stack, parameter_stack, optim_metric=metric_fn, plot=self.print_plot)
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
                    if self.verbose:
                        print(f"Optimized Zernike index {zernike_index} to value {optimal_value} nm")

                image = self.image_provider.getImage()
                imageSet.append(image)
                parameter_set.append(initial_coefficients_full.copy())
                opt_v_set.append(opt_v)
                self._updateImageAndDM(current_coefficients=initial_coefficients_full,zernike_index=0,val=0)
                try:
                    self._notify_dependents(params={'final_coefficients': initial_coefficients_full.copy()})
                except Exception as e:
                    if self.verbose:
                        print("Error notifying dependents after iteration:", e)
                    pass

            # Save final optimized coefficients to instance variable
            self.final_coefficients = initial_coefficients_full.copy()
            if self.verbose:
                print(f"Final optimized coefficients: {self.final_coefficients[:min(12, len(self.final_coefficients))]}")

            # Save results after successful completion
            if self.image_log:
                np.save(self.recorded_image_folder + '/' + 'imageSet',imageSet)
                np.save(self.recorded_image_folder + '/' + 'parameterSet',parameter_set)
                np.save(self.recorded_image_folder + '/' + 'optimalValuesSet',opt_v_set)
        finally:
            # Always stop continuous mode if it was started, even if loop is interrupted
            if self.continuous_scan:
                if self.verbose:
                    print("Stopping continuous acquisition mode...")
                self.image_provider.stopContinuousMode()


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
