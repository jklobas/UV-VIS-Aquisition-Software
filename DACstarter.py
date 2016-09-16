from mbedRPC import *
from time import sleep
import numpy as np



serdev = '/dev/ttyACM'
mbed = SerialRPC(serdev, 9600)


green = DigitalOut(mbed, "Green")
red = DigitalOut(mbed, "Red")
blue = DigitalOut(mbed, "Blue")


gnd23= RPCFunction(mbed, "ADSREAD23GND")
gnd01= RPCFunction(mbed, "ADSREAD01GND")


vdd01= RPCFunction(mbed, "ADSREAD01VDD")
vdd23= RPCFunction(mbed, "ADSREAD23VDD")


scl23= RPCFunction(mbed, "ADSREAD23SCL")
scl01= RPCFunction(mbed, "ADSREAD01SCL")


sda01= RPCFunction(mbed, "ADSREAD01SDA")
sda23= RPCFunction(mbed, "ADSREAD23SDA")


gnd23var = RPCVariable(mbed,"readingGND23")
gnd01var = RPCVariable(mbed,"readingGND01")


vdd01var = RPCVariable(mbed,"readingVDD01")
vdd23var = RPCVariable(mbed,"readingVDD23")


scl23var = RPCVariable(mbed,"readingSCL23")
scl01var = RPCVariable(mbed,"readingSCL01")
 

sda01var = RPCVariable(mbed,"readingSDA01")
sda23var = RPCVariable(mbed,"readingSDA23")


