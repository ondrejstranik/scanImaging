''' test spectral viewer '''

import pytest
import napari

@pytest.mark.GUI
def test_FlimViewer():
    ''' check if gui works'''
    from scanImaging.algorithm.flimData import FlimData
    from scanImaging.gui.flimViewer.flimViewer import FlimViewer
    from qtpy.QtWidgets import QApplication, QMainWindow
    import numpy as np

    flimData = FlimData(np.random.randint(0,10,size=(10,3,50,50)),[0,30])
    flimData.processData()

    app = QApplication([])
    window =QMainWindow()
    
    fV = FlimViewer(flimData=flimData,show=False)

    window.setCentralWidget(fV.viewer.window._qt_window)
    window.show()
    app.exec()


