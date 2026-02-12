# VirtualISM class for simulating an image scanning microscope instrument

import numpy as np
from scipy.signal import fftconvolve
import time

def generatePSF(aperture_mask, aperture_phase, wavelength):
    aperture_mask = np.asarray(aperture_mask, dtype=np.float32)
    aperture_phase = np.asarray(aperture_phase, dtype=np.float32)
    phase_factor = np.exp(1j * (2.0 * np.pi / float(wavelength)) * aperture_phase).astype(np.complex64)
    ft_psf = aperture_mask.astype(np.complex64) * phase_factor
    psf_c = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(ft_psf))).astype(np.complex64)
    psf = (np.abs(psf_c).astype(np.float32))**2
    return psf

import numpy as np

def add_microscopy_noise(image, mean_photons=500, read_noise_e=3, background=0.05):
    """
    Add realistic microscopy noise to a clean ground truth image.
    
    Parameters:
    -----------
    image : ndarray
        Clean ground truth image (normalized 0-1)
    mean_photons : float
        Mean photon count at maximum intensity (higher = less noisy)
    read_noise_e : float
        Camera read noise in electrons RMS
    background : float
        Constant background fluorescence level (fraction of max)
    
    Returns:
    --------
    noisy_image : ndarray
        Image with Poisson + Gaussian noise added
    """
    # Add background
    if background>0:
        image += background
        print(f"added {background}")
    
    # Poisson noise (photon shot noise): This is added in the scanner
    if mean_photons>=1:
        image = np.random.poisson(image * mean_photons).astype(float) / mean_photons
    
    # Gaussian noise (read noise)
    if read_noise_e>0:
        # Combine and clip to avoid negative values, might be not correct for some detectors.
        image=np.clip(image + np.random.normal(0, read_noise_e / mean_photons, image.shape), 0, None) 


# Example usage:
# noisy = add_microscopy_noise(ground_truth, mean_photons=1000, read_noise_e=2, background=0.05)

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
        channelgridx=4
        channelgridy=4
        self.base_image = kwargs.get('base_image', np.zeros((256, 256)))
        self.virtualScanner = kwargs.get('virtualScanner', None)
        self.virtualAdaptiveOptics = kwargs.get('virtualAdaptiveOptics', None)
        self.parameters['lambda'] = kwargs.get('wavelength', 520)  # in nm
        self.parameters['baseImage']=kwargs.get('baseImage','image1')
        self.parameters['emissionWavelength'] = kwargs.get('emissionWavelength', 520)  # in nm
        self.parameters['numberOfChannelX'] = kwargs.get('numberOfChannelX', channelgridx) 
        self.parameters['numberOfChannelY'] = kwargs.get('numberOfChannelY', channelgridy) 
        self.parameters['NA'] = kwargs.get('NA', 1.4)
        self.parameters['pixelSize'] = kwargs.get('pixelSize', 5)  #
        self.parameters['microscopeType'] = kwargs.get('microscopeType', 'widefield')  # 'confocal', 'widefield', or ISM
        self.parameters['pinholeSize'] = kwargs.get('pinholeSize', 1.0)  # in Airy units
        self.parameters['ismPixelShift'] = kwargs.get('ismPixelShift', 10)  # in pixels
        self.parameters['systemAberrations'] = kwargs.get('systemAberrations', np.zeros(11)) # Zernike modes
        self.backGroundLevel=0.0
        self.poissonBackground=0.0
        self.gaussBackground=0.0
        from pathlib import Path
        base_path = Path(__file__).resolve().parent
        imagepath1 = rf"{base_path}/images/Gemini_Generated_Image_saa5ihsaa5ihsaa5.png"
        imagepath2 = rf"{base_path}/images/Gemini_Generated_Image_18skaw18skaw18sk.png"
        imagepath3 = rf"{base_path}/images/radial-sine-144.png"
        self.imagedict={"image1":imagepath1,"image2":imagepath2,"image3":imagepath3}
        self.currentimage="empty"

    def makeIsmOffsetGrid(self, channelgridx, channelgridy, pixelshift=10):
        ''' create an ISM detector offset grid '''
        offsetgrid = np.array([[i*pixelshift,j*pixelshift] for i in range(channelgridx) for j in range(channelgridy)])
        return offsetgrid
    
    def setProbeParameter(self, param):
        val=param.get("photons_per_pixel",0)
        if val>=1 and (self.virtualScanner is not None):
            self.virtualScanner.setMaxPhotonPerPixel(val)
        val=param.get("background_level",-1)
        if val>=0.0:
            self.backGroundLevel=val
        val=param.get("dark_noise",-1)
        if val>=0.0:
            self.poissonBackground=val
        val=param.get("read_noise",-1)
        if val>=0:
            self.gaussBackground=val

    def setImage(self,image:str):
        imagepath=self.imagedict.get(image,None)
        if image!=self.currentimage and imagepath is not None:
            print(f"loading image {imagepath}")
            from PIL import Image
            img = Image.open(imagepath).convert("L")
            rows, cols = self.virtualScanner.imageSize  # numpy array or tuple
            img = img.resize((int(cols), int(rows)), resample=Image.BILINEAR)
            # to numpy float32 and normalize to [0,1]
            arr = np.asarray(img, dtype=np.float32)
            if arr.max() != 0:
                arr /= arr.max()
            self.virtualScanner.setVirtualProbe(arr)  
            self.base_image = self.virtualScanner.virtualProbe.reshape(self.virtualScanner.imageSize)
            self.currentimage=image


    def connect(self, virtualScanner=None,virtualAdaptiveOptics=None):
        ''' simulate connecting to the virtual ISM instrument '''
        if virtualScanner is not None:
            self.virtualScanner = virtualScanner
            self.setImage("image1")
            if hasattr(self.virtualScanner, 'virtualProbe'):
                self.base_image = self.virtualScanner.virtualProbe.reshape(self.virtualScanner.imageSize)
        if virtualAdaptiveOptics is not None:
            self.virtualAdaptiveOptics = virtualAdaptiveOptics
            virtualAdaptiveOptics.register_dependent(self)
        self.is_connected = True
        print(f"{self.name} connected.")

    def updateFromDM(self):
        '''Update the ISM image based on the current DM surface'''
        self.updateImage()

    def updateImage(self):
        '''Calculate the ISM image based on the current base image and virtual scanner settings'''
        if (not self.is_connected) or self.virtualScanner is None or self.virtualAdaptiveOptics is None:
            raise Exception("VirtualISM is not connected.")
        # For simplicity, we just return the base image modified by a factor
        # In a real implementation, this would involve complex calculations
        start_time=time.perf_counter()
        self.setImage(self.getParameter('baseImage'))
        # ensure single-precision for the heavy ops
        def _to_float32(arr):
            arr = np.asarray(arr)
            if arr.dtype == np.float32:
                return arr
            try:
                return arr.astype(np.float32, copy=False)
            except Exception:
                return arr.astype(np.float32, copy=True)
        print("Updating ISM image...")
        wavelength = self.getParameter('lambda')
        channelgridx = self.getParameter('numberOfChannelX')
        channelgridy = self.getParameter('numberOfChannelY')
        channels = channelgridx * channelgridy
        na=self.getParameter('NA')
        pixelSize=self.getParameter('pixelSize')
        aperture_radius = na * wavelength / (2 * pixelSize)
        apterure_mask = np.zeros_like(self.base_image)
        rr, cc = np.ogrid[:apterure_mask.shape[0], :apterure_mask.shape[1]]
        center = (apterure_mask.shape[0] // 2, apterure_mask.shape[1] // 2)
        mask = (rr - center[0])**2 + (cc - center[1])**2 <= aperture_radius**2
        apterure_mask[mask] = 1.0
        phase_mask=apterure_mask.copy()
        systemAberrations = self.getParameter('systemAberrations')
        if systemAberrations is not None and len(systemAberrations) > 0:
            # apply system aberrations as Zernike modes
            from scanImaging.algorithm.SimpleZernike import zernike_phase_map
            system_phase_on_aperture = zernike_phase_map(systemAberrations, int(aperture_radius*2), int(aperture_radius*2))
            startx = (apterure_mask.shape[1]-system_phase_on_aperture.shape[1]) // 2
            starty = (apterure_mask.shape[0]-system_phase_on_aperture.shape[0]) // 2
            system_phase = np.zeros((apterure_mask.shape[0], apterure_mask.shape[1]))
            system_phase[starty:starty + system_phase_on_aperture.shape[0], startx:startx + system_phase_on_aperture.shape[1]] = system_phase_on_aperture
            phase_mask[mask] *= system_phase[mask]
        if self.virtualAdaptiveOptics is not None:
            dmsurface = self.virtualAdaptiveOptics.image
            # get the active aperture size of the dm
            dmaperture_size = self.virtualAdaptiveOptics.active_aperture
            if dmaperture_size == 0:
                dmaperture_size = dmsurface.shape[0]
            # map DM surface with DM apterure radius to ISM apterture with radius given by NA and wavelength
            # this means the size of the DM surfact has to be scaled to the ISM apterture size
            # If the ISM apterture is larger (in terms of pixel size), we need to interpolate the values.
            # If it is larger, we need to remove some values.
            # First we need an interpolation function for the complete dm surface
            from scipy.ndimage import zoom
            zoomfactor = (2 * aperture_radius) / dmaperture_size
            if zoomfactor != 1.0:
                dmsurface_resized = zoom(dmsurface, zoomfactor, order=1)
            else:
                dmsurface_resized = dmsurface
            # plot the resized dm surface for debugging
            # Now we need to crop or pad the resized dm surface to match the complete apterure mask size

            if dmsurface_resized.shape[0] > apterure_mask.shape[0]:
                startx = (dmsurface_resized.shape[1] - apterure_mask.shape[1]) // 2
                starty = (dmsurface_resized.shape[0] - apterure_mask.shape[0]) // 2
                dmsurface_cropped = dmsurface_resized[starty:starty + apterure_mask.shape[0], startx:startx + apterure_mask.shape[1]]
            else:
                startx = (apterure_mask.shape[1]-dmsurface_resized.shape[1]) // 2
                starty = (apterure_mask.shape[0]-dmsurface_resized.shape[0]) // 2
                dmsurface_cropped = np.zeros((apterure_mask.shape[0], apterure_mask.shape[1]))
                dmsurface_cropped[starty:starty + dmsurface_resized.shape[0], startx:startx + dmsurface_resized.shape[1]] = dmsurface_resized
            # Now it can be directly applied to the apterture mask
            phase_mask += apterure_mask * dmsurface_cropped
        # Generate the PSF
        print("Generating PSF...")


        base = _to_float32(self.base_image)
        apterure_mask = _to_float32(apterure_mask)
        phase_mask = _to_float32(phase_mask)

        # PSF timing
        t0 = time.perf_counter()
        psf = generatePSF(apterure_mask, phase_mask, wavelength)
        t1 = time.perf_counter()
        final_image = None
        # if confocal or ISM, modify the PSF accordingly
        if self.getParameter('microscopeType').lower() in ['confocal', 'ism']:
            # apply pinhole
            emission_wavelength = self.getParameter('emissionWavelength')
            psf_emission = generatePSF(apterure_mask, phase_mask, emission_wavelength)
            pinhole_size = self.getParameter('pinholeSize')  # in Airy units
            airy_radius = 1.22 * wavelength / (2 * na)
            pinhole_radius_pixels = int((pinhole_size * airy_radius) / pixelSize)
            rr, cc = np.ogrid[:psf.shape[0], :psf.shape[1]]
            center = (psf.shape[0] // 2, psf.shape[1] // 2)
            pinhole_mask = ((rr - center[0])**2 + (cc - center[1])**2 <= pinhole_radius_pixels**2).astype(np.float32)
            # convolve emission psf with pinhole
            psf_emission = fftconvolve(psf_emission, pinhole_mask, mode='same')
            if self.getParameter('microscopeType').lower() == 'ism':
                # create ISM offset grid
                print("ISM mode: convolving with shifted PSFs...")
                channelgridx = self.getParameter('numberOfChannelX')
                channelgridy = self.getParameter('numberOfChannelY')
                ism_pixelshift = self.getParameter('ismPixelShift')
                offsetgrid = self.makeIsmOffsetGrid(channelgridx, channelgridy, ism_pixelshift)
                # shift the emission PSF according to the offset grid, mutlitply with excitation PSF, convolve with the image, and write them to each channel
                final_image = np.zeros((base.shape[0], base.shape[1], channelgridx*channelgridy), dtype=np.float32)
                for ch in range(channelgridx*channelgridy):
                    shifted_psf = np.roll(np.roll(psf_emission, offsetgrid[ch,0], axis=0), offsetgrid[ch,1], axis=1)
                    combined_psf = psf * shifted_psf
                    final_image[:,:,ch] = fftconvolve(base, combined_psf.astype(np.float32, copy=False), mode='same')
            else:
                print("Confocal mode: convolving with modified PSF...")
                psf*=psf_emission
                final_image = fftconvolve(base, psf.astype(np.float32, copy=False), mode='same')
        else:
            t_conv_start = time.perf_counter()
            final_image = fftconvolve(base, psf.astype(np.float32, copy=False), mode='same')
            t_conv_end = time.perf_counter()
            print(f"PSF gen: {t1-t0:.3f}s, convolution (fft): {t_conv_end-t_conv_start:.3f}s")
        # Normalize the convolved image
        final_image = final_image / np.max(final_image) if np.max(final_image) != 0 else final_image
        # default: widefield or confocal: same image in all channels
        if not self.getParameter('microscopeType').lower() == 'ism':
            final_image = np.repeat(final_image[:, :, np.newaxis], channels, axis=2)
        # set the ism image to the virtual scanner
        add_microscopy_noise(final_image,0,self.gaussBackground,self.backGroundLevel)
        self.virtualScanner.setVirtualProbe(final_image)
        print(f"ISM image updated in {time.perf_counter()-start_time:.2f} seconds.")

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