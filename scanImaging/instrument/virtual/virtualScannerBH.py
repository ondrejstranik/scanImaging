#%%
''' module for a virtual beckel&Hickel scanner having internal stack for the data'''

import numpy as np
import time
from viscope.instrument.base.baseADetector import BaseADetector
from viscope.virtualSystem.component.sample import Sample


class VirtualBHScanner(BaseADetector):
    ''' class to emulate B&H scanner
    data are stored in a stack
    data are in the form (2 bytes): 
    bits 31-28 flags
    b31 ... =1 no photon
    b30 ... =1 macrotime overflow
    b29 ... =1 stack overflow. Data loss. Dismiss previous data
    b28 ... =1 trigger for new line
    b16-27 ... =microtime.  12 bits resolution. photon delay = 2**12 - microtime
    b12-15 ... = channel. 4 bits resolution. total 16 channels
    b0-11 ... =macrotime. 12 bit resolution
    
    data are scanned horizontally, single lined (this is given by a separate scanner)

    stack is in the form:
    columns of : tMacroFlag,newLineFlag,tMacroSaw,_virtualPhoton

    


    '''

    DEFAULT = {'name':'virtualBHScanner',
               'pixelTime': 1e-5, # in [s], time per Pixel
               'maxPhotonPerPixel': 10, # maximal number of event per pixel
               'macroTimeIncrement': 1e-7, # in [s] time to increase the macrotime counter per one
               'imageSize' : np.array([512,512]),
               'timeSize'  : 2**12, 
               'numberOfChannel': 16,
               'timeRange': np.array([0, 20]), # range of the time axis
               'scanOffSet' : np.array([5,10]) # add (horizontal,vertical) lines
                # simulate returning path of the scanner 
               } 

    def __init__(self, name=DEFAULT['name'], **kwargs):
        ''' detector initialization'''
        super().__init__(name=name, **kwargs)

        # probe, which it will scan over. 
        self.virtualProbe = None
        self.maxPhotonPerPixel=self.DEFAULT['maxPhotonPerPixel']
        # variable to generate virtual data
        self.pixelTime = self.DEFAULT['pixelTime'] # dwell time on one pixel
        self.signalTime = self.pixelTime/self.maxPhotonPerPixel # smallest time between photons
        self.macroTimeIncrement = self.DEFAULT['macroTimeIncrement']
        self.numberOfChannel = self.DEFAULT['numberOfChannel']
        self.timeSize = self.DEFAULT['timeSize']


        self.timeRange = self.DEFAULT['timeRange'] # range of the time axis
        self.macroTimeTotal = 0 # start counting with start of the measurement, in [self.signalTime units]
        self.scanPosition = 0 # current position of the scanner in linear dimension
        self.overflowFlag = False

        self.acquiring = False
        if self.acquiring is False:
            self.acquisitionStopTime = 0
            self.lastStackTime = 0

        # setting the virtual probe
        self.imageSize = self.DEFAULT['imageSize']
        # scan size take into acount the return of scanner and double it
        # the doubling assure the roll over the new image 
        #self.scanSize = (self.imageSize + self.DEFAULT['imageSize'])*([2,1])
        self.scanSize = (self.imageSize + self.DEFAULT['scanOffSet'])

        # maximal position of the scan (for one sample) in subpixels 
        self.maxScanPosition = int(np.prod(self.scanSize)*(self.maxPhotonPerPixel))

        #Probes
        # 1. constant with a thick line
        self.virtualProbe = np.zeros(self.imageSize) +0
        self.virtualProbe[:, int(self.imageSize[0]*0.4):int(self.imageSize[0]*0.6)] = 1

#        _Sample = Sample()
#        _Sample.setAstronaut(sampleSize=self.imageSize,photonRateMax=100)
#        self.virtualProbe = _Sample.get()

        #  add scan simulation return of the scan 
        self.virtualProbeExtra = np.zeros(self.scanSize)
        self.virtualProbeExtra[0:self.imageSize[0],0:self.imageSize[1]]= self.virtualProbe

        # double the virtual probe. it overcome the indexing issue with scan rolling over
        self.virtualProbeExtra = np.vstack((self.virtualProbeExtra,self.virtualProbeExtra))

        # linearize the probe
        self.virtualProbeExtra = self.virtualProbeExtra.reshape(-1)

        # --- FLIM lifetime simulation ---
        # When flimEnabled, photon microtimes are drawn from a per-pixel bi-exponential
        # decay (+ Gaussian IRF) instead of a uniform random value. Maps are linearized
        # the same way as the probe so they can be indexed by the scan pixel index.
        # Emitted microtimes follow the B&H reverse start-stop convention
        # (raw microtime = timeSize - photon_delay_in_bins).
        self.flimEnabled = False
        self.tau1Extra = None   # linearized fast/long component lifetime map [ns]
        self.tau2Extra = None   # linearized second component lifetime map [ns]
        self.fracExtra = None   # linearized amplitude fraction of component 1 (0..1)
        self.irfSigma = 0.0     # Gaussian IRF width [ns]
        self.irfOffset = 0.0    # Gaussian IRF centre offset [ns]

    def _linearize2D(self, map2d):
        ''' linearize a 2D (imageSize) map into the doubled/extended scan layout,
        matching the transform used for the virtual probe so it can be indexed by
        the scan pixel index '''
        arr = np.zeros(self.scanSize)
        arr[0:self.imageSize[0], 0:self.imageSize[1]] = map2d
        arr = np.vstack((arr, arr))
        return arr.reshape(-1)

    def setLifetimeMap(self, tau1Map, tau2Map, fracMap):
        ''' set per-pixel bi-exponential lifetime maps (each of shape imageSize, in ns).
        tau1Map, tau2Map ... lifetimes of the two decay components
        fracMap ... amplitude fraction assigned to component 1 (0..1) '''
        self.tau1Extra = self._linearize2D(np.asarray(tau1Map, dtype=float))
        self.tau2Extra = self._linearize2D(np.asarray(tau2Map, dtype=float))
        self.fracExtra = self._linearize2D(np.asarray(fracMap, dtype=float))
        self.flimEnabled = True

    def setLifetimeUniform(self, tau1, tau2, frac):
        ''' convenience: set a spatially uniform bi-exponential lifetime '''
        ones = np.ones(self.imageSize)
        self.setLifetimeMap(tau1*ones, tau2*ones, frac*ones)

    def setIRF(self, sigma, offset=0.0):
        ''' set Gaussian instrument response function (sigma and centre offset, in ns) '''
        self.irfSigma = float(sigma)
        self.irfOffset = float(offset)

    def disableFlim(self):
        ''' revert to uniform random microtimes '''
        self.flimEnabled = False

    def generatePhotonMicroTimes(self, pixelIdx):
        ''' generate raw B&H microtimes for photons located at the given scan pixel
        indices, drawn from the per-pixel bi-exponential decay convolved with a
        Gaussian IRF. Returns a uint-compatible int array in the reverse start-stop
        convention (raw = timeSize - delay_in_bins).

        This is the core lifetime model; it is also called directly by the
        verification harness. '''
        pixelIdx = np.asarray(pixelIdx, dtype=int)
        n = pixelIdx.size
        if n == 0:
            return np.zeros(0, dtype=int)

        if self.flimEnabled and self.tau1Extra is not None:
            tau1 = self.tau1Extra[pixelIdx]
            tau2 = self.tau2Extra[pixelIdx]
            frac = self.fracExtra[pixelIdx]
        else:
            # fall back to a uniform default lifetime if FLIM is on but no map was set
            tau1 = np.full(n, 2.5)
            tau2 = np.full(n, 0.6)
            frac = np.full(n, 0.7)

        # pick component per photon, then sample exponential delay [ns]
        comp1 = np.random.rand(n) < frac
        tau = np.where(comp1, tau1, tau2)
        tau = np.where(tau <= 0, 1e-6, tau)   # guard against zero/negative lifetimes
        delay_ns = np.random.exponential(tau)

        # Gaussian instrument response
        if self.irfSigma > 0 or self.irfOffset != 0:
            delay_ns = delay_ns + np.random.normal(self.irfOffset, self.irfSigma, n)

        # convert delay [ns] -> time bins, then to reverse start-stop raw microtime
        span = float(self.timeRange[1] - self.timeRange[0])
        delay_bins = np.round(delay_ns / span * self.timeSize).astype(int)
        np.clip(delay_bins, 0, self.timeSize - 1, out=delay_bins)
        return (self.timeSize - 1 - delay_bins).astype(int)

    def setMaxPhotonPerPixel(self,val):
        if val<1:
            return
        self.maxPhotonPerPixel=int(val)
        self.signalTime = self.pixelTime/self.maxPhotonPerPixel
        self.maxScanPosition = int(np.prod(self.scanSize)*(self.maxPhotonPerPixel))


    def setVirtualProbe(self, virtualProbe):
        ''' set the virtual probe to be scanned '''
        self.virtualProbe = virtualProbe
        self._updateProbeExtra()
        
    def _updateProbeExtra(self):
        ''' update the extra probe when the main probe is changed
          Include also possible probe with several channels'''
        if self.virtualProbe is None:
            raise Exception("Virtual probe is not set")
        # if the proble contains images for each channel
        if len(self.virtualProbe.shape)==3:
            if self.virtualProbe.shape[2]!=self.numberOfChannel:
                raise Exception("Virtual probe channel number mismatch")
            # loop over all channels and keep the probes each in a separate array
            for ch in range(self.numberOfChannel):
                currentvirtualProbeExtra = np.zeros(self.scanSize)
                currentvirtualProbeExtra[0:self.imageSize[0],0:self.imageSize[1]]= self.virtualProbe[:,:,ch]
                # double the probe in vertical direction
                currentvirtualProbeExtra = np.vstack((currentvirtualProbeExtra,currentvirtualProbeExtra))
                # linearize the probe
                currentvirtualProbeExtra = currentvirtualProbeExtra.reshape(-1)
                if ch==0:
                    self.virtualProbeExtra = np.zeros((self.scanSize[0]*2*self.scanSize[1], self.numberOfChannel))
                self.virtualProbeExtra[:, ch] = currentvirtualProbeExtra
            return
        self.virtualProbeExtra = np.zeros(self.scanSize)
        self.virtualProbeExtra[0:self.imageSize[0],0:self.imageSize[1]]= self.virtualProbe

        # double the virtual probe. it overcome the indexing issue with scan rolling over 
        self.virtualProbeExtra = np.vstack((self.virtualProbeExtra,self.virtualProbeExtra))

        # linearize the probe
        self.virtualProbeExtra = self.virtualProbeExtra.reshape(-1)
        
    def startAcquisition(self):
        '''start detector collecting data in a stack '''
        super().startAcquisition()
        self.macroTimeTotal = 0 # start recording the total time
        self.lastStackTime = time.time_ns() #self.acquisitionStartTime*1
        self.acquisitionStopTime = None        
        self.scanPosition = 0
        # set arbitrary start position of scanning
        # TODO: switched off for debugging
        #self.scanPosition = np.random.randint(int(self.maxScanPosition)) 

    def stopAcquisition(self):
        '''stop detector collecting data in a stack '''
        super().stopAcquisition()
        self.acquisitionStopTime = time.time_ns()

    def _calculateStack(self):
        ''' calculate the virtual stack '''

        currentTime = time.time_ns()
        
        if self.acquisitionStopTime is not None:
            currentTime = self.acquisitionStopTime
        hasvirtualChannels=(len(self.virtualProbe.shape)==3)

        if self.acquisitionStopTime == self.lastStackTime:
            virtualStack = None
        else:

            # number of sub pixels to generate
            nSubPixel = int((currentTime - self.lastStackTime)*1e-9/self.signalTime)
            newScanPosition = self.scanPosition + nSubPixel

            # photon events generation
            # generate the photons ... 1= photon was generated                
            scanRange = np.arange(self.scanPosition, newScanPosition)
            pixelIdx = (scanRange//int(self.pixelTime/self.signalTime)).astype(int)
            threshold = 0.5
            
            # no new data
            if pixelIdx.size == 0:
                return None

            # detect overflow 
            if pixelIdx[-1]>2*(np.prod(self.scanSize)-1):
                self.overflowFlag = True
                np.clip(pixelIdx,0,2*(np.prod(self.scanSize)-1),out=pixelIdx)
                print(f'scanner overfLow!')

            # change this later with different probability depending on the channel (scaled by sum intensity per channel)
            _channel_events = np.random.randint(0,self.numberOfChannel,len(pixelIdx))

            # this probability calculation already contain the the poisson noise
            if hasvirtualChannels:
                _virtualPhoton = (self.virtualProbeExtra[pixelIdx,_channel_events]*np.random.rand(len(pixelIdx))>threshold)*1
            else:
                _virtualPhoton = (self.virtualProbeExtra[pixelIdx]*np.random.rand(len(pixelIdx))>threshold)*1

            # macro time + macroTimeFlag generation
            #print(f'time : {(currentTime - self.lastStackTime)}')
            #print(f'nSubPixel {nSubPixel}')
            tMacro = self.macroTimeTotal + np.arange(nSubPixel+1)*self.signalTime/self.macroTimeIncrement
            tMacroSaw = (tMacro%2**12)
            tMacroFlag = (tMacroSaw[1:] - tMacroSaw[0:-1])<0

            # update macroTimeTotal
            self.macroTimeTotal = tMacro[-1]
            #print(f'len(tMacroSaw): {len(tMacroSaw)}')
            
            # new line flag generation
            newLineFlag = scanRange%(self.scanSize[1]*self.maxPhotonPerPixel)==0
            # remove new line flag from rows below sample
            # this rows imitates return of the beam
            outerRows = np.logical_and(scanRange>np.prod(self.scanSize-self.DEFAULT['scanOffSet']*[1,0])*self.maxPhotonPerPixel-1,
                                        scanRange< self.maxScanPosition)
            #outerRows = scanRange< self.maxScanPosition
            #print(f'outerRows: {outerRows}')
            newLineFlag[outerRows]= False

            # update the position, correct for roll over
            if newScanPosition > self.maxScanPosition:
                self.scanPosition = newScanPosition%self.maxScanPosition
            else:              
                self.scanPosition = newScanPosition

            # generate the signal
            # remove the events without photons (but keep the flag event)
            validSignal = ((_virtualPhoton ==1) | tMacroFlag | newLineFlag)
            #print(f'size of stack {np.sum(validSignal*1)}')

            _numberOfSignal = np.sum(validSignal)
            # generate microtimes. With FLIM enabled they follow the per-pixel
            # bi-exponential decay (+ IRF); otherwise a uniform random value.
            # (Non-photon flag events also get a microtime but are discarded by the
            #  processor, so sampling them from the pixel's decay is harmless.)
            if self.flimEnabled:
                _microTime = self.generatePhotonMicroTimes(pixelIdx[validSignal])
            else:
                _microTime = np.random.randint(0,self.timeSize,_numberOfSignal)
            if hasvirtualChannels:
                _channel = _channel_events[validSignal]
            else:
                _channel = np.random.randint(0,self.numberOfChannel,_numberOfSignal)


            virtualStack = np.vstack([tMacroFlag[validSignal],
                                      newLineFlag[validSignal],
                                      tMacroSaw[1:][validSignal],
                                      _channel,
                                      _microTime]).T


        self.lastStackTime = currentTime
        return virtualStack
        
    def updateStack(self):
        ''' get data from the stack'''        
        res = self._calculateStack()
        
        if res is None:
            return self.stack
        
        if self.isEmptyStack():
            self.stack = res
        else:
            self.stack = np.vstack([self.stack,res])

        return self.stack

if __name__ == '__main__':

    pass





# %%