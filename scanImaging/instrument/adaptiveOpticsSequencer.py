
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
        
    def loop(self):

        # for synchronisation reasons it stop the camera acquisition
        self.camera.stopAcquisition()

        # check if the folder exist, if not create it
        p = Path(self.recorded_image_folder)
        p.mkdir(parents=True, exist_ok=True)

        ''' finite loop of the sequence'''
        for ii in range(self.num_scans):
            print(f'recording {ii} image')

        np.save(self.dataFolder + '/' + 'imageSet',self.imageSet)