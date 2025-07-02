''' tests '''

import pytest

def test_bHScanner():
    ''' check the data are generated'''
    from scanImaging.instrument.bHscanner import BHScanner
    import time

    bhScanner = BHScanner()
    bhScanner.connect()
    bhScanner.startAcquisition()
    i = 0
    while i<30: 
        time.sleep(0.01)
        myStack = bhScanner.getStack()
        print(myStack)
        i += 1
    bhScanner.stopAcquisition()
    bhScanner.disconnect()



