# scanImaging
package to control scanning imaging systems


## Windows package installation instruction (conda)

0. install [Viscope package](https://github.com/ondrejstranik/viscope)
1. start conda, activate your environment `conda activate xxx` (xxx .. name of your environment)
2. move to the package folder `cd yyy` (yyy ... name of the folder)
3. type `python -m pip install -e.`

The virtual mode does not require additional hardware and can be used to test features and GUI.

### Hardware installations

- To install the bmc package add `pip install -e extra_libs/bmc` and place the appropriate pyd file in the `bmc/bmc` directory. The DLL path might also change.  Alternatively install the default bmc package.
- The Boston Micromachines deformable mirror has some default dll and calibration file locations hardcoded in the file DMBmc.py. Please check. Due to compatibility issues with the python version, the SDK of Boston Micromachines had to be recompiled. 
- The old scanner uses a Windows 2000 computer. It is controlled via serial port and emulated mouse commands. 


## Testing

- The most relevant tests run with `python -m pytest tests/ -m "not GUI"`
- Best way to test GUI: use virtual devices. 

## New developments

- The development folder contains some new hardware developments. It is used to test new hardware before integration into the main code.