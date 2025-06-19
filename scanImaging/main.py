'''
class for live viewing spectral images
'''
#%%


if __name__ == "__main__":

    from scanImaging.instrument.virtual.virtualScanner import VirtualScanner
    import time
    import numpy as np

    scanner = VirtualScanner()
    scanner.connect()
    scanner.setParameter('threadingNow',True)
    scanner.startAcquisition()

    cTime = time.time()
    while time.time()-cTime < 3:
        if scanner.flagLoop.is_set():
            data = scanner.stack
            print(f'stack \n {data}')
            scanner.flagLoop.clear()

        time.sleep(0.001)
    scanner.stopAcquisition()
    scanner.disconnect()    


