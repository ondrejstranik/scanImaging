''' tests '''

import pytest


def test_BHData_validation():
    from scanImaging.algorithm.dataCoding import BHData
    import numpy as np
    
    data1 = BHData()
    data1._setArbitraryData(5)

    stream = data1.dataToStream()
    data2 = BHData()
    data2.streamToData(stream)

    print(f'stream \n {stream}')
    print(f'microtime original \n {data1.microTime}')
    print(f'microtime recovered \n {data2.microTime}')

    assert np.all(data1.microTime==data2.microTime) 


    