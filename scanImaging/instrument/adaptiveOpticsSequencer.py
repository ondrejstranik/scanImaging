from viscope.instrument.base.baseSequencer import BaseSequencer
from pathlib import Path
import numpy as np
import time

# Import optimizer module
from .ao_optimizers import SequentialOptimizer


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
        if not self.continous_acquisition and self.scanner_device is not None:
            self.continous_acquisition = True
            self.scanner_device.startAcquisition()

    def stopContinuousMode(self):
        if self.continous_acquisition and self.scanner_device is not None:
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

class AdaptiveOpticsController(BaseSequencer):
    """
    Adaptive Optics controller - coordinates hardware, processing, and optimization.

    UPDATE FLOW ARCHITECTURE - TWO MODES OF OPERATION:
    --------------------------------------------------

    MODE 1: MANUAL CONTROL (GUI owns DM):
    - User adjusts Zernike coefficients via GUI sliders/controls
    - GUI updates hardware DM directly (set_phase_map_from_zernike)
    - GUI updates its own visualizations
    - Sequencer is NOT running
    - Purpose: Manual exploration, testing, setup

    MODE 2: AO OPTIMIZATION (Sequencer owns DM):
    - Sequencer runs optimization algorithm
    - At start: Reads current DM state (if use_current_dm_state=True)
    - During optimization:
      a) Sequencer updates hardware DM
      b) Sequencer notifies GUI (one-way: Sequencer → GUI)
      c) GUI updates visualizations only (but may redundantly call display_surface)
    - GUI does NOT send updates back to hardware during optimization
    - Purpose: Automated aberration correction

    INITIALIZATION (use_current_dm_state parameter):
    - True (default): Start optimization from current DM coefficients
      • Preserves manual adjustments made in GUI
      • Allows incremental optimization
      • GUI → Sequencer handoff is seamless
    - False: Start from zero or predefined values
      • Clean slate optimization
      • Useful for testing/comparison

    NOTIFICATION FLOW (during AO optimization):
    1. Sequencer updates hardware DM
    2. Sequencer calls _notify_dependents(params={'current_coefficients': ...})
    3. GUI.on_sequencer_update() receives notification
    4. GUI updates visualizations:
       - DM surface plot
       - Zernike coefficient display
       - Convergence plot (via separate updateAOMetrics callback)
    5. GUI may call display_surface() again (redundant but harmless)

    WHY "REDUNDANT" UPDATES EXIST:
    - GUI needs hardware update capability for MODE 1 (manual control)
    - Sequencer needs hardware update capability for MODE 2 (optimization)
    - Both may call display_surface() - this overlap is intentional
    - During optimization, GUI's hardware calls are redundant but harmless
    - Alternative would be complex mode locking - current design is simpler

    KEY PRINCIPLE:
    - Manual mode: GUI controls hardware
    - Optimization mode: Sequencer controls hardware, GUI follows
    - Transition: use_current_dm_state enables smooth handoff between modes
    """

    DEFAULT = {'name': 'AdaptiveOpticsController'}

    def __init__(self,viscope=None, name=None, **kwargs):
        ''' initialisation '''

        if name== None: name= AdaptiveOpticsController.DEFAULT['name']
        super().__init__(name=name, **kwargs)
        self.viscope=viscope
        # devices
        self.deformable_mirror = None
        self.image_provider = None
        self.dm_display = None

        # NEW: Optional image processing and z-scanning support
        self.image_processor = kwargs.get('image_processor', None)  # ISM/FLIM processing pipeline
        self.z_stage = kwargs.get('z_stage', None)                  # z-stage for volume imaging
        self.current_z = None                                        # current z-position
        self.recorded_image_folder= "Data/AdaptiveOptics"
        self.optim_iterations=3
        self.optim_iterations_per_mode=[]  # Per-mode iterations (empty = use optim_iterations for all)
        self.optim_method='simple_interpolation'  # other methods can be implemented
        self.initial_zernike_indices=[4,3,5,11,6,7]  # Hierarchical order: defocus, astigmatism, spherical, coma
        self.zernike_initial_coefficients_nm=[0,0,0,0,0,0]  # Start at zero (or use Gibson-Lanni for Z11)
        self.zernike_amplitude_scan_nm=[80,60,60,200,50,50]  # Conservative step sizes (10-200 nm range)
        self.num_steps_per_mode=3
        self.imageSet=[]
        self._dependents = []
        self.continuous_scan=False
        self.image_log=False
        self.print_plot=True
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
        self.scanner_enable_auto_restart = False  # automatically restart scanner on failure

        # Loop control and cleanup
        self._stop_requested = False  # Flag to signal loop to exit cleanly

        # Convergence detection
        self.enable_convergence_detection = False  # Enable auto-stop when converged
        self.convergence_threshold = 0.01  # Relative improvement threshold (1%)
        self.convergence_window = 2  # Number of iterations to check for convergence

        # SPGD parameters
        self.spgd_gain = 0.05  # Step size for coefficient updates (tune this!)
        self.spgd_delta = 10.0  # Perturbation size in nm (typically 10-50 nm)
        self.spgd_iterations = 100  # Number of SPGD iterations

        # Random search parameters
        self.random_search_iterations = 100  # Number of random samples to try
        self.random_search_range = 200.0  # Search range in nm (±range around initial)

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
            d.on_sequencer_update(params=params)
            # Call updateAOMetrics if available
            if hasattr(d, 'updateAOMetrics'):
                # Check if we have iteration-based data (SPGD, random_search) or scan-based data
                if params and 'spgd_metric_history' in params:
                    # SPGD iteration-based updates
                    metric_history = params['spgd_metric_history']
                    iteration = params.get('iteration', len(metric_history)-1)
                    d.updateAOMetrics(metric_history, mode='iteration', iteration=iteration)
                elif params and 'random_search_metric_history' in params:
                    # Random search iteration-based updates
                    metric_history = params['random_search_metric_history']
                    iteration = params.get('iteration', len(metric_history)-1)
                    d.updateAOMetrics(metric_history, mode='iteration', iteration=iteration)
                elif self.parameter_stack is not None and self.metric_values is not None:
                    # Scan-based methods (simple_interpolation, weighted_fit)
                    d.updateAOMetrics(self.metric_values, self.parameter_stack, mode='scan')

    def _update_dm_and_notify(self, coefficients, **extra_params):
        """
        Apply coefficients to DM hardware and notify GUI (standard pattern).

        This is the STANDARD PATTERN for all DM updates in optimization algorithms.
        Every DM update triggers a GUI notification to keep visualizations synchronized.

        Parameters:
        -----------
        coefficients : array-like
            Zernike coefficients to apply to DM
        **extra_params : dict
            Additional parameters to include in notification (e.g., iteration, metric)

        Usage Examples:
        ---------------
        # Simple update:
        self._update_dm_and_notify(coeffs)

        # With iteration info (for SPGD, random_search):
        self._update_dm_and_notify(coeffs, iteration=42, metric=123.4,
                                    spgd_metric_history=[...])

        # With scan info (for scan-based methods):
        self._update_dm_and_notify(coeffs, zernike_index=4, scan_value=50.0)
        """
        # Update DM hardware
        self.deformable_mirror.set_phase_map_from_zernike(coefficients)
        self.deformable_mirror.display_surface()

        # Build notification parameters
        params = {'current_coefficients': coefficients.copy()}
        params.update(extra_params)

        # Notify GUI and other dependents
        try:
            self._notify_dependents(params=params)
        except Exception as e:
            if self.verbose:
                print(f"Error notifying dependents: {e}")

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

    def computeOptimalParametersWeightedFit(self, image_stack, parameter_stack, optim_metric, plot=False):
        """
        Compute optimal parameter using ALL scan points with Poisson weighting.

        More robust to noise than 3-point interpolation by using all measured data.
        Poisson weighting: w ∝ √metric (since variance ∝ mean for Poisson noise).

        Parameters:
        -----------
        image_stack : list of images
            All acquired images during parameter scan
        parameter_stack : array-like
            Corresponding parameter values
        optim_metric : callable
            Function to compute image quality metric
        plot : bool
            If True, save diagnostic plot

        Returns:
        --------
        optimal_parameter : float
            Estimated optimal parameter value
        miss_max : int
            -1 if optimum below scan range, +1 if above, 0 if within
        optimal_metric : float
            Predicted metric value at optimum (or best measured if parabola invalid)
        """
        if len(image_stack) == 0 or len(parameter_stack) == 0:
            raise ValueError("Image stack and parameter stack must have non-zero length.")
        if len(image_stack) != len(parameter_stack):
            raise ValueError("Image stack and parameter stack must have the same length.")

        # Compute metric for each image
        metric_values = np.array([optim_metric(img) for img in image_stack])
        if self.verbose:
            print(f"Metric values (weighted fit): {metric_values}")

        # Poisson weighting: weight ∝ √metric (variance ∝ mean)
        # Add small epsilon to avoid division by zero
        weights = np.sqrt(np.maximum(metric_values, 1e-12))

        # Fit quadratic to ALL points with weights
        try:
            coeffs = np.polyfit(parameter_stack, metric_values, deg=2, w=weights)
            a, b, c = coeffs
        except Exception as e:
            if self.verbose:
                print(f"Polynomial fit failed: {e}, falling back to best measured point")
            max_idx = np.argmax(metric_values)
            return parameter_stack[max_idx], 0, metric_values[max_idx]

        # Check if parabola opens downward (concave) - required for maximum
        if a >= 0:
            # Not a maximum - parabola opens upward or is flat
            if self.verbose:
                print(f"Parabola opens upward (a={a:.2e}), using best measured point")
            max_idx = np.argmax(metric_values)
            return parameter_stack[max_idx], 0, metric_values[max_idx]

        # Compute vertex of parabola: x = -b / (2a)
        optimal_parameter = -b / (2 * a)

        # Check if optimum is within scan range
        miss_max = 0
        min_param = np.min(parameter_stack)
        max_param = np.max(parameter_stack)

        if optimal_parameter < min_param:
            miss_max = -1
            if self.verbose:
                print(f"Optimum below scan range: {optimal_parameter:.2f} < {min_param:.2f}")
        elif optimal_parameter > max_param:
            miss_max = 1
            if self.verbose:
                print(f"Optimum above scan range: {optimal_parameter:.2f} > {max_param:.2f}")

        # Predicted optimal metric value at vertex
        optimal_metric = a * optimal_parameter**2 + b * optimal_parameter + c

        # Diagnostic plot if requested
        if plot:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 6))

            # Measured points
            plt.scatter(parameter_stack, metric_values, c='blue', s=100,
                       alpha=0.6, label='Measured')

            # Fitted parabola
            param_fine = np.linspace(min_param, max_param, 200)
            metric_fine = a * param_fine**2 + b * param_fine + c
            plt.plot(param_fine, metric_fine, 'r--', linewidth=2,
                    label='Weighted fit')

            # Optimal point
            plt.scatter([optimal_parameter], [optimal_metric], c='red', s=200,
                       marker='*', edgecolors='black', linewidth=2,
                       label=f'Optimum: {optimal_parameter:.2f}')

            plt.xlabel('Parameter value', fontsize=12)
            plt.ylabel('Metric value', fontsize=12)
            plt.title('All-Points Weighted Fit (Poisson weighting)', fontsize=14)
            plt.legend(fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.savefig('optimization_weighted_fit.png', dpi=150, bbox_inches='tight')
            plt.close()

        # Store for diagnostics
        self.parameter_stack = parameter_stack
        self.metric_values = metric_values

        return optimal_parameter, miss_max, optimal_metric

    def loop_SPGD(self):
        """
        SPGD (Stochastic Parallel Gradient Descent) optimization loop.

        Optimizes ALL Zernike modes simultaneously using stochastic perturbations.
        Very robust to Poisson noise and mode coupling.

        Algorithm:
        1. Apply random perturbation +δ to all coefficients
        2. Measure metric J+
        3. Apply perturbation -δ
        4. Measure metric J-
        5. Estimate gradient: ∇J ≈ (J+ - J-) / (2δ) × δ
        6. Update coefficients: c += gain × ∇J
        """
        from scipy.ndimage import laplace
        import time
        import numpy as np
        from pathlib import Path

        if self.verbose:
            print("\n" + "="*60)
            print("Starting SPGD optimization")
            print("="*60)
            print(f"Gain: {self.spgd_gain}")
            print(f"Delta: {self.spgd_delta} nm")
            print(f"Iterations: {self.spgd_iterations}")
            print(f"Modes: {self.initial_zernike_indices}")
            print(f"Using metric: {self.selected_metric}")

        # Initialize coefficients
        initial_coefficients_full = self._initialize_coefficients()
        coeffs = initial_coefficients_full.copy()

        # Metric function
        metric_fn = self.get_metric_function()

        # History tracking
        metric_history = []
        coeff_history = []

        try:
            if self.continuous_scan:
                self.image_provider.startContinuousMode()

            for iteration in range(self.spgd_iterations):
                if self._stop_requested:
                    if self.verbose:
                        print("Stop requested - exiting SPGD loop")
                    return

                # Generate random perturbation for optimized modes only
                perturbation = np.zeros_like(coeffs)
                for zernike_index in self.initial_zernike_indices:
                    perturbation[zernike_index] = np.random.randn() * self.spgd_delta

                # Measure metric with positive perturbation
                coeffs_plus = coeffs + perturbation
                self.deformable_mirror.set_phase_map_from_zernike(coeffs_plus)
                self.deformable_mirror.display_surface()
                time.sleep(0.05)  # Shorter settling for SPGD
                img_plus = self.acquire_and_process_image()
                metric_plus = metric_fn(img_plus)

                yield  # Allow interruption

                if self._stop_requested:
                    if self.verbose:
                        print("Stop requested - exiting SPGD loop")
                    return

                # Measure metric with negative perturbation
                coeffs_minus = coeffs - perturbation
                self.deformable_mirror.set_phase_map_from_zernike(coeffs_minus)
                self.deformable_mirror.display_surface()
                time.sleep(0.05)
                img_minus = self.acquire_and_process_image()
                metric_minus = metric_fn(img_minus)

                yield  # Allow interruption

                # Estimate gradient and update coefficients
                gradient_estimate = (metric_plus - metric_minus) / (2 * self.spgd_delta)
                coeffs += self.spgd_gain * gradient_estimate * perturbation

                # Track metrics
                avg_metric = (metric_plus + metric_minus) / 2
                metric_history.append(avg_metric)
                coeff_history.append(coeffs.copy())

                # Apply updated coefficients and notify (standard pattern)
                self._update_dm_and_notify(coeffs,
                                            iteration=iteration,
                                            metric=avg_metric,
                                            spgd_metric_history=metric_history.copy())

                # Print progress every 5 iterations to avoid console spam
                if self.verbose and (iteration % 5 == 0 or iteration == self.spgd_iterations - 1):
                    print(f"SPGD iter {iteration+1}/{self.spgd_iterations}: metric={avg_metric:.4f}")
                    print(f"  Coeffs: {coeffs[self.initial_zernike_indices]}")

                # Convergence check
                if self.enable_convergence_detection and len(metric_history) >= self.convergence_window + 1:
                    recent_metrics = metric_history[-self.convergence_window-1:]
                    relative_improvement = (recent_metrics[-1] - recent_metrics[0]) / (abs(recent_metrics[0]) + 1e-12)

                    if relative_improvement < self.convergence_threshold:
                        if self.verbose:
                            print(f"SPGD converged after {iteration+1} iterations!")
                            print(f"Improvement {relative_improvement:.4f} < threshold {self.convergence_threshold:.4f}")
                        break

            # Save final coefficients
            self.final_coefficients = coeffs.copy()
            if self.verbose:
                print(f"\nSPGD optimization complete!")
                print(f"Final coefficients: {coeffs[:min(12, len(coeffs))]}")
                print(f"Final metric: {metric_history[-1]:.4f}")

            # Final notification (with complete metric history for final plot)
            try:
                self._notify_dependents(params={
                    'current_coefficients': coeffs.copy(),  # For DM surface update
                    'final_coefficients': coeffs.copy(),  # For backwards compatibility
                    'spgd_metric_history': metric_history.copy(),
                    'iteration': len(metric_history) - 1
                })
            except Exception:
                pass

        finally:
            if self.continuous_scan:
                if self.verbose:
                    print("Stopping continuous acquisition mode...")
                self.image_provider.stopContinuousMode()

    def loop_random_search(self):
        """
        Random search baseline optimization.

        Simple baseline algorithm that randomly samples the coefficient space.
        Useful for comparison with more sophisticated methods.

        Algorithm:
        1. Generate random coefficients within search range
        2. Measure metric
        3. Keep best coefficients found
        4. Repeat for specified iterations

        Very robust (no local minima issues) but inefficient compared to
        gradient-based methods. Serves as a baseline to validate that other
        algorithms actually improve performance.
        """
        import time
        import numpy as np

        if self.verbose:
            print("\n" + "="*60)
            print("Starting Random Search optimization")
            print("="*60)
            print(f"Iterations: {self.random_search_iterations}")
            print(f"Search range: ±{self.random_search_range} nm")
            print(f"Modes: {self.initial_zernike_indices}")
            print(f"Using metric: {self.selected_metric}")

        # Initialize coefficients
        initial_coefficients_full = self._initialize_coefficients()
        best_coeffs = initial_coefficients_full.copy()

        # Metric function
        metric_fn = self.get_metric_function()

        # Measure initial metric
        self.deformable_mirror.set_phase_map_from_zernike(best_coeffs)
        self.deformable_mirror.display_surface()
        time.sleep(0.05)
        img_initial = self.acquire_and_process_image()
        best_metric = metric_fn(img_initial)

        if self.verbose:
            print(f"Initial metric: {best_metric:.4f}")
            print(f"Initial coefficients: {best_coeffs[self.initial_zernike_indices]}")

        # History tracking
        metric_history = [best_metric]

        try:
            if self.continuous_scan:
                self.image_provider.startContinuousMode()

            for iteration in range(self.random_search_iterations):
                if self._stop_requested:
                    if self.verbose:
                        print("Stop requested - exiting random search")
                    return

                # Generate random coefficients within search range
                random_coeffs = best_coeffs.copy()
                for zernike_index in self.initial_zernike_indices:
                    # Random offset from initial value
                    offset = np.random.uniform(-self.random_search_range,
                                               self.random_search_range)
                    random_coeffs[zernike_index] = initial_coefficients_full[zernike_index] + offset

                # Measure metric at random point
                self.deformable_mirror.set_phase_map_from_zernike(random_coeffs)
                self.deformable_mirror.display_surface()
                time.sleep(0.05)
                img = self.acquire_and_process_image()
                metric_value = metric_fn(img)

                # Update best if improved
                if metric_value > best_metric:
                    best_metric = metric_value
                    best_coeffs = random_coeffs.copy()
                    if self.verbose:
                        print(f"Iteration {iteration+1}/{self.random_search_iterations}: "
                              f"New best metric = {best_metric:.4f}")
                        print(f"  Best coefficients: {best_coeffs[self.initial_zernike_indices]}")

                metric_history.append(best_metric)

                # Apply best coefficients to DM and notify (standard pattern)
                # This ensures DM state matches what we're reporting to GUI
                self._update_dm_and_notify(best_coeffs,
                                            iteration=iteration,
                                            metric=best_metric,
                                            random_search_metric_history=metric_history.copy())

                yield  # Allow interruption

            # Save final coefficients
            self.final_coefficients = best_coeffs.copy()

            if self.verbose:
                print(f"\nRandom search optimization complete!")
                print(f"Best metric: {best_metric:.4f}")
                print(f"Best coefficients: {best_coeffs[:min(12, len(best_coeffs))]}")
                print(f"Improvement: {((best_metric - metric_history[0]) / metric_history[0] * 100):.1f}%")

            # Final notification with complete history
            self._update_dm_and_notify(best_coeffs,
                                        final_coefficients=best_coeffs.copy(),
                                        random_search_metric_history=metric_history.copy(),
                                        iteration=len(metric_history) - 1)

        finally:
            if self.continuous_scan:
                if self.verbose:
                    print("Stopping continuous acquisition mode...")
                self.image_provider.stopContinuousMode()


    def start(self):
        """Start the AO sequencer loop"""
        try:
            # Reset stop flag before starting
            self._stop_requested = False
            self.setParameter('threadingNow', True)
        except Exception:
            pass

    def stop(self):
        """Stop the AO sequencer and ensure cleanup happens immediately"""
        if self.verbose:
            print("Stop requested - cleaning up...")

        try:
            # Set flag to signal loop to exit (checked at each yield)
            self._stop_requested = True
            # Immediately stop continuous scan mode (don't wait for finally block)
            if self.continuous_scan and self.image_provider is not None:
                try:
                    
                    if self.verbose:
                        print("Stopping continuous acquisition mode...")
                    self.image_provider.stopContinuousMode()
                except Exception as e:
                    if self.verbose:
                        print(f"Error stopping continuous mode: {e}")

            # Quit the worker thread
            if getattr(self, 'worker', None) is not None:
                self.worker.quit()

        except Exception as e:
            if self.verbose:
                print(f"Error during stop: {e}")

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

    def acquire_and_process_image(self):
        """
        Acquire image with optional processing pipeline.

        If image_processor is provided (ISM, FLIM, etc.), applies processing
        and returns processed image. Otherwise returns raw image.

        Returns:
            numpy.ndarray: Image for AO metric computation
        """
        # Acquire raw data
        raw_data = self.image_provider.getImage()

        # Apply processing if available
        if self.image_processor is not None:
            try:
                processed = self.image_processor.process(raw_data)

                # Update display with additional data (ISM, FLIM layers)
                if hasattr(self, '_update_multimodal_display'):
                    self._update_multimodal_display(processed)

                # Return processed image for metric computation
                return processed.get('image', raw_data)

            except Exception as e:
                if self.verbose:
                    print(f"Warning: Image processing failed: {e}")
                    print("Falling back to raw image")
                return raw_data
        else:
            # No processing: return raw image
            return raw_data

    def _update_multimodal_display(self, processed_data):
        """
        Update display with multi-modal data (ISM, FLIM).

        Override this method or implement in GUI integration for full functionality.

        Args:
            processed_data: Dict with keys like 'image', 'ism_enhanced', 'lifetime', etc.
        """
        # Placeholder - implement when ISM/FLIM modules added
        pass

    def _updateImageAndDM(self, current_coefficients, zernike_index, val):
        """
        Update DM, notify, wait, and acquire image.

        Used by scan-based methods (simple_interpolation, weighted_fit).
        Combines the most common sequence: DM update -> notify -> wait -> acquire.
        """
        # Use helper for DM update + notification
        self._update_dm_and_notify(current_coefficients,
                                    zernike_index=zernike_index,
                                    scan_value=val)

        # Wait for stabilization and acquire image with optional processing
        time.sleep(0.1)
        return self.acquire_and_process_image()

    def _build_iterations_dict(self):
        """Build dictionary mapping Zernike mode → number of iterations

        Uses optim_iterations_per_mode vector if provided, otherwise falls back
        to scalar optim_iterations for all modes.

        Returns:
            dict: {mode: iterations} for each mode in initial_zernike_indices
        """
        modes = self.initial_zernike_indices
        iterations_vector = self.optim_iterations_per_mode

        # If no per-mode iterations specified, use global default for all
        if not iterations_vector or len(iterations_vector) == 0:
            return {mode: self.optim_iterations for mode in modes}

        # Warn if vector is longer than needed
        if len(iterations_vector) > len(modes):
            if self.verbose:
                print(f"Warning: optim_iterations_per_mode has {len(iterations_vector)} values but only {len(modes)} modes. Extra values will be ignored.")

        # Extend vector with default if too short
        if len(iterations_vector) < len(modes):
            if self.verbose:
                print(f"Note: optim_iterations_per_mode has {len(iterations_vector)} values for {len(modes)} modes. Extending with default value {self.optim_iterations}.")
            iterations_vector = list(iterations_vector) + [self.optim_iterations] * (len(modes) - len(iterations_vector))

        # Build dictionary
        return {mode: int(iters) for mode, iters in zip(modes, iterations_vector)}

    def loop_scan_based_modular(self):
        """
        Sequential scan-based optimization using modular optimizer architecture.

        This is the new implementation that delegates to SequentialOptimizer.
        """
        if self.verbose:
            print("Starting Adaptive Optics Sequencer Loop (Modular Optimizer)")
            print(f"Optimizing indices: {self.initial_zernike_indices} with initial coefficients (nm): {self.zernike_initial_coefficients_nm} and scan amplitudes (nm): {self.zernike_amplitude_scan_nm}")
            print(f"Using metric: {self.selected_metric}")

        # Initialize coefficients
        initial_coefficients_full = self._initialize_coefficients()

        # Create and initialize optimizer
        optimizer = SequentialOptimizer(self)
        optimizer.initialize(
            initial_coefficients=initial_coefficients_full,
            zernike_indices=self.initial_zernike_indices,
            scan_amplitudes=self.zernike_amplitude_scan_nm
        )

        # Run optimization (yields for GUI updates)
        yield from optimizer.run()

        # Store final results
        results = optimizer.get_results()
        self.final_coefficients = results['final_coefficients']

        if self.verbose:
            print(f"Optimization complete. Final coefficients: {self.final_coefficients[:min(12, len(self.final_coefficients))]}")

    def loop_scan_based(self):
        """
        Sequential scan-based optimization (simple_interpolation or weighted_fit).

        Delegates to modular optimizer implementation (SequentialOptimizer).
        The legacy implementation is preserved in loop_scan_based_legacy() for reference.
        """
        # Use modular implementation
        yield from self.loop_scan_based_modular()
        return

        # LEGACY IMPLEMENTATION BELOW (unreachable, kept for reference)
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

        # Build per-mode iteration counts (hierarchical optimization)
        iterations_dict = self._build_iterations_dict()
        if self.verbose and self.optim_iterations_per_mode:
            print(f"Using per-mode iterations: {iterations_dict}")

        try:
            ''' finite loop of the sequence - hierarchical optimization'''
            # Loop over each zernike mode (outer loop)
            for mode_idx, zernike_index in enumerate(self.initial_zernike_indices):
                mode_iterations = iterations_dict.get(zernike_index, self.optim_iterations)
                if self.verbose:
                    print(f"\n=== Optimizing mode {zernike_index} with {mode_iterations} iterations ===")

                # Initialize convergence tracking for this mode
                metric_history = []

                # Initialize accumulated data for weighted_fit algorithm
                accumulated_images = []
                accumulated_params = []

                # Optimize this mode for the specified number of iterations (inner loop)
                for ii in range(mode_iterations):
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
                        if self._stop_requested:
                            if self.verbose:
                                print("Stop requested - exiting AO loop")
                            return  # Exit early, finally block will still execute
                    # Accumulate data for weighted_fit (across all iterations for this mode)
                    if self.optim_method == 'weighted_fit':
                        accumulated_images.extend(image_stack)
                        accumulated_params.extend(parameter_stack)

                    miss_max=1
                    while num_scan_points<100 and abs(miss_max)!=0:
                        # compute optimal parameter for this mode using selected algorithm
                        metric_fn = self.get_metric_function()

                        # Select optimization algorithm
                        if self.optim_method == 'weighted_fit':
                            # Use accumulated data across all iterations for this mode
                            optimal_value, miss_max, opt_v = self.computeOptimalParametersWeightedFit(
                                accumulated_images, accumulated_params, optim_metric=metric_fn, plot=self.print_plot)
                            # For weighted_fit, use accumulated_params for range checking
                            range_check_params = accumulated_params
                        else:  # Default: 'simple_interpolation'
                            # Use only current scan data
                            optimal_value, miss_max, opt_v = self.computeOptimalParametersSimpleInterpolation(
                                image_stack, parameter_stack, optim_metric=metric_fn, plot=self.print_plot)
                            # For simple_interpolation, use current scan data
                            range_check_params = parameter_stack

                        if miss_max != 0:
                            num_scan_points+=1
                            scan_amplitude *=1.2
                        else:
                            scan_amplitude *= 0.7
                        added_value=None
                        if miss_max<0:
                            added_value=range_check_params[0]-scan_amplitude/2
                            if optimal_value<range_check_params[0]:
                                added_value=optimal_value - scan_amplitude/2
                            #append at the beginning
                            parameter_stack=[added_value]+parameter_stack
                        if miss_max>0:
                            added_value=range_check_params[-1]+scan_amplitude/2
                            if optimal_value>range_check_params[-1]:
                                added_value=optimal_value + scan_amplitude/2
                            # append at the end
                            parameter_stack=parameter_stack+[added_value]
                        if added_value is not None:
                            current_coefficients[zernike_index] = added_value
                            # acquire image
                            image=self._updateImageAndDM(current_coefficients=current_coefficients,zernike_index=zernike_index,val=added_value)
                            image_stack=[image]+image_stack if miss_max<0 else image_stack+[image]
                            # Also accumulate this additional point for weighted_fit
                            if self.optim_method == 'weighted_fit':
                                if miss_max < 0:
                                    accumulated_images.insert(0, image)
                                    accumulated_params.insert(0, added_value)
                                else:
                                    accumulated_images.append(image)
                                    accumulated_params.append(added_value)
                            yield  # yield to allow interruption
                            if self._stop_requested:
                                if self.verbose:
                                    print("Stop requested - exiting AO loop")
                                return  # Exit early, finally block will still execute

                    initial_coefficients_full[zernike_index] = optimal_value
                    current_coefficients[zernike_index] = optimal_value
                    self.deformable_mirror.set_phase_map_from_zernike(current_coefficients)
                    self.deformable_mirror.display_surface()
                    # notify dependents about updated coefficients AND convergence plot
                    try:
                        # This triggers both DM surface update and convergence plot update
                        self._notify_dependents(params={'coefficients': initial_coefficients_full.copy()})
                    except Exception:
                        pass
                    if self.verbose:
                        print(f"Optimized Zernike index {zernike_index} to value {optimal_value} nm (iteration {ii+1}/{mode_iterations}), metric={opt_v:.4f}")

                    # Convergence detection
                    metric_history.append(opt_v)
                    if self.enable_convergence_detection and len(metric_history) >= self.convergence_window + 1:
                        # Check relative improvement over convergence_window
                        recent_metrics = metric_history[-self.convergence_window-1:]
                        relative_improvement = (recent_metrics[-1] - recent_metrics[0]) / (abs(recent_metrics[0]) + 1e-12)

                        if relative_improvement < self.convergence_threshold:
                            if self.verbose:
                                print(f"Converged! Improvement {relative_improvement:.4f} < threshold {self.convergence_threshold:.4f}")
                                print(f"Stopping after {ii+1}/{mode_iterations} iterations for mode {zernike_index}")
                            break  # Exit inner loop (iterations for this mode)

                # After completing all iterations for this mode, get final image
                image = self.acquire_and_process_image()
                imageSet.append(image)
                parameter_set.append(initial_coefficients_full.copy())
                opt_v_set.append(opt_v)
                self._updateImageAndDM(current_coefficients=initial_coefficients_full,zernike_index=0,val=0)
                try:
                    self._notify_dependents(params={'final_coefficients': initial_coefficients_full.copy()})
                except Exception as e:
                    if self.verbose:
                        print("Error notifying dependents after mode optimization:", e)
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

    def loop(self):
        """
        Main optimization loop - routes to selected algorithm.

        Delegates to specialized methods based on self.optim_method:
        - 'spgd': Stochastic Parallel Gradient Descent (simultaneous optimization)
        - 'random_search': Random sampling baseline
        - 'simple_interpolation' or 'weighted_fit': Sequential scan-based optimization
        """
        if self.optim_method == 'spgd':
            if self.verbose:
                print("Using SPGD optimization method")
            yield from self.loop_SPGD()
        elif self.optim_method == 'random_search':
            if self.verbose:
                print("Using Random Search optimization method")
            yield from self.loop_random_search()
        elif self.optim_method in ['simple_interpolation', 'weighted_fit']:
            if self.verbose:
                print(f"Using {self.optim_method} optimization method")
            yield from self.loop_scan_based()
        else:
            raise ValueError(f"Unknown optimization method: {self.optim_method}")


# Backward compatibility alias
AdaptiveOpticsSequencer = AdaptiveOpticsController


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
