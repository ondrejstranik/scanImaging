#!/usr/bin/env python
#%%
# Before running this, copy the "bmc" module from Python3\site-packages\bmc
#  to your Python installation's site-packages directory.

import sys
# Add the current directory to sys.path
sys.path.append(r'C:\Users\localxueweikai\Documents\GitHub\scanImaging\scanImaging\instrument\dmc')

import os
import sys

if sys.version_info >= (3, 8):
    os.add_dll_directory(r"C:\Program Files\Boston Micromachines\Bin64")



import _bmc
#%%

import bmc



dm = bmc.BmcDm()
err_code = dm.open_dm('17DW008#094')
if err_code:
    raise Exception(dm.error_string(err_code))

mapping = list(dm.default_mapping())

#%%
data = bmc.DoubleVector()
data.assign(dm.num_actuators(), 0.0)
dm.send_data(data)
#%%
monotonic_map = range(0, dm.num_actuators())
dm.send_data_custom_mapping(data, monotonic_map)

print('BMC error status:', dm.error_string(dm.get_status()))

dm.close_dm()

