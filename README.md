# UV-VIS-Aquisition-Software
Software to interface the Ocean Optics USB-4000 spectrometer with MBED-driven A2D inputs via RPC


![Temperature Programmed Sublimation - UV/VIS](https://github.com/jklobas/UV-VIS-Aquisition-Software/blob/master/TPDfiglabeled.png)

This python 2.7 program is designed to be used in an interactive python environment such as ipython. Custom-built A2D circuitry, run on a MBED device, is called via remote procedure call via python. Because of this, the software must be modified heavily to operate with the devices of other users. 

This software will:  

(1) initialize the spectrometer via ocean optics seabreeze drivers.
(2) initialize the MBED adcs
(3) acquire dark, blank, and sample spectra
(4) visualize in real time, the raw spectrum, the absorbance spectrum, and system temperatures

![(4) visualize in real time, the raw spectrum, the absorbance spectrum, and system temperatures](https://github.com/jklobas/UV-VIS-Aquisition-Software/blob/master/Acquisitionscreen.png?raw=true)
(5) provide interactive views of various absorbances and raw spectra of saved samples
![(5) provide interactive views of various absorbances and raw spectra of saved samples](https://github.com/jklobas/UV-VIS-Aquisition-Software/blob/master/postanneal1050-1150.png)
(6) acquire data via 'monitor mode' or via scripting supported function call
(7) save / load data via pickle

Additionally the analysis software will:
(1) import and slice/interpolate sample and reference standard data
(2) visualize acquired data as 3D plots and 2D plots
![experiment as acquired](https://github.com/jklobas/UV-VIS-Aquisition-Software/blob/master/experimental.png)
(3) perform multicomponent optimization via selected scipy.optimize method

(4) visualize fits as 3D plots and 2D plots, as well as system diagnostic information
![2D fit of chlorine](https://github.com/jklobas/UV-VIS-Aquisition-Software/blob/master/Cl2lin.png)
![Diagnostic information](https://github.com/jklobas/UV-VIS-Aquisition-Software/blob/master/ramp.png)
![simulated spectra from fit](https://github.com/jklobas/UV-VIS-Aquisition-Software/blob/master/simulated.png?raw=true)


