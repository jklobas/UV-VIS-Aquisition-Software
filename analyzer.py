from __future__ import division
import autograd.numpy as np
import math
import pandas as pd
import pickle
#from scipy.optimize import minimize, fmin
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from autograd import grad
from autograd.convenience_wrappers import hessian_vector_product as hvp
from lmfit import minimize, Parameters, Parameter, report_fit
from decimal import Decimal as dec

#this function suite performs the following
#(1) Loader(filename)
#(2) variableinit()
#(3) standardsimport()
#(4) find_nearest(array, value)
#(5) wavetruncator(min, max)
#(6) cl2maker(wave, temp)







def loader(filename): #recover previously saved files. may need to extract variables to use functions above
    global lister, sample, waves, datetimer, secondser, intimer, avgnumer, absorbances, temps, tempsinterperavg, abssecs
    lister = pd.read_pickle(str(filename) + ".pkl")
    sample = lister[1]
    waves = lister[0]
    datetimer = lister[2]
    secondser = lister[3]
    intimer = lister[4]
    avgnumer = lister[5]
    absorbances = lister[6]
    temps = lister[7]
    tempsinterperavg = lister[8]
    abssecs = [absorbances[l][4] for l in range(0, len(absorbances))]
    


def variableinit(): #initiates variables
    global wavers, absorbancers, tempers, params, concs, absplotparamlistflag, rawplotparamlistflag, res, outer, absnum, flags, fittemps, fitsecs, fitval
    
    wavers = []
    absorbancers = []
    tempers = []
    params = []
    concs =[]
    tempidx = [absorbances[l][1] for l in range(0,len(absorbances))]
    [tempers.append([tempsinterperavg[l][1], tempsinterperavg[l][2], tempsinterperavg[l][3], tempsinterperavg[l][4]]) for l in tempidx]
    absplotparamlistflag = 0
    rawplotparamlistflag = 0
    res = []
    outer = []
    absnum = []
    flags = []
    fittemps = []
    fitsecs = []
    fitval = []



def standardsimport(): #imports standards
    global O3, Cl2, ClOOCl, OClO, Cl2O3, ClOrt, ClOlist, ClClO2, Cl2O6
    path = '/home/klobas/Documents/workstuff/Anderson/ClOOCl/crosssections/'
    O3 = np.genfromtxt (path + 'O3295smooth.csv' , delimiter=",")
    OClO = np.genfromtxt (path + 'OClOsmooth.csv', delimiter=",")
    ClOOCl = np.genfromtxt (path + 'cloocl_jpl.txt')
    Cl2O3 = np.genfromtxt (path + 'Cl2O3extrap.csv', delimiter = ",")
    ClOlist = np.genfromtxt (path + 'tdept/clospectra.txt')
    ClOrt = [ [ClOlist[l][0], ClOlist[l][11]] for l in range (0,len(ClOlist))] 
    Cl2 = np.genfromtxt(path + 'homemade/Cl2home.txt')
    ClClO2 = np.genfromtxt(path+'ClClO2_MullerWillner(1992)_298K_220-390nm.txt')
    Cl2O6 = np.genfromtxt(path+ 'Cl2O6_Green(2004)_298K_200-450nm.txt')
    
def find_nearest(array,value): #Gives index of value closest to input in array
    idx = (np.abs(array-value)).argmin()
    return [idx]

def wavetruncator(min = 245, max = 400 ): #truncates the spectrum to the specified wavelength
    global wavers, absorbancers, temps, Cl2, O3, OClO, ClOOCl, Cl2O3, ClO, ClClO2, Cl2O6, bounds, varsp, xx
    wavers = (waves[find_nearest(waves, int(min))[0]:find_nearest(waves, int(max))[0]])
    absorbancers = np.nan_to_num([absorbances[l][0][find_nearest(waves, int(min))[0]:find_nearest(waves,int(max))[0]] for l in range(0,len(absorbances))])
    O3intens = [l[1] for l in O3]
    O3 = O3intens[find_nearest(waves, int(min))[0]:find_nearest(waves,int(max))[0]]
    O3 = [float(i) for i in O3]
    OClOintens = [l[1] for l in OClO]
    OClO = OClOintens[find_nearest(waves, int(min))[0]:find_nearest(waves,int(max))[0]]
    OClO = [float(i) for i in OClO]
    Cl2 = [[ cl2maker(x,l[2]) for x in wavers] for l in tempers]
    Cl2 = Cl2[0]
#    Cl2intens = [l[1] for l in Cl2]
#    Cl2 = Cl2intens[find_nearest(waves,int(min))[0]:find_nearest(waves,int(max))[0]]
#    Cl2 = [float(i) for i in Cl2]
    ClOOCwave = [l[0] for l in ClOOCl]
    ClOOCintens = [l[1] for l in ClOOCl]
    ClOOCl = np.interp(wavers, ClOOCwave, ClOOCintens)
    ClOOCl = [float(i) for i in ClOOCl]
    Cl2O3wave = [l[0] for l in Cl2O3]
    Cl2O3intens = [l[1] for l in Cl2O3]
    Cl2O3 = np.interp(wavers, Cl2O3wave, Cl2O3intens)
    Cl2O3 = [float(i) for i in Cl2O3]
    ClOwave = [l[0] for l in ClOrt]
    ClOwave = ClOwave[::-1]
    ClOintens = [l[1] for l in ClOrt]
    ClOintens = ClOintens[::-1]
    ClO = np.interp(wavers, ClOwave, ClOintens)
    ClO = [float(i) for i in ClO]
    ClClO2wave = [l[0] for l in ClClO2]
    ClClO2intens = [l[1] for l in ClClO2]
    ClClO2 = np.interp(wavers, ClClO2wave, ClClO2intens)
    ClClO2 = [float(i) for i in ClClO2]
    varsp = [1e12, 1e15, 1e13, 1e13, 1e13, 1e13, 1e13]
    xx = [O3,Cl2, ClO, OClO, ClOOCl, Cl2O3, ClClO2] 
    Cl2O6wave = [l[0] for l in Cl2O6]
    Cl2O6intens = [l[1] for l in Cl2O6]
    Cl2O6 = np.interp(wavers, Cl2O6wave, Cl2O6intens)
    Cl2O6 = [float(i) for i in Cl2O6]
    bounds = [(0 , 1e20), (0 , 1e20), (0 , 1e20),(0 , 1e20),(0 , 1e20),(0 , 1e20)] 
#    vars = [1e16, 1e16, 1e16]
#    xx = [Cl2[0], ClO, OClO]
#     

def cl2maker(wave, temp): #synthetic Cl2 spectrum
    return  2.73e-19*math.sqrt(math.tanh(402.67852903632944/temp))*math.exp(-99.0*math.tanh(402.6785290363294/temp)*math.log(329.5/wave)**2) + 9.32e-21*math.sqrt(math.tanh(402.6785290363294/temp))*math.exp(-91.5*math.tanh(402.6785290363294/temp)*math.log(402.7/ wave )**2)
    
    
def residualizer(varsp, x, data, flag): #builds error function
    o3 = varsp[0]
    cl2 = varsp[1]
    clO = varsp[2]
    oClO = varsp[3]
    clOOCl = varsp[4]
    cl2O3 = varsp[5]
    clClO2 = varsp[6]
    
    if int(flag) == 1:
        model = np.dot(oClO,x[3])
    if int(flag) == 2:
        model = np.dot(oClO,x[3]) + np.dot(cl2,x[1])
    if int(flag) == 3:
        model =  np.dot(cl2,x[1]) + np.dot(oClO,x[3]) + np.dot(clOOCl, x[4])
    if int(flag) == 4:
        model =  np.dot(cl2,x[1]) + np.dot(cl2O3,x[5]) + np.dot(oClO,x[3]) + np.dot(clOOCl, x[4])
    if int(flag) == 5:
        model = np.dot(cl2,x[1]) + np.dot(clO,x[2]) + np.dot(oClO,x[3]) + np.dot(clOOCl, x[4]) + np.dot(cl2O3, x[5])
    if int(flag) == 6:
        model = np.dot(o3, x[0]) + np.dot(cl2,x[1]) + np.dot(clO,x[2]) + np.dot(oClO,x[3]) + np.dot(clOOCl, x[4]) + np.dot(cl2O3, x[5])
    if int(flag) == 7:
        model = np.dot(oClO, x[3]) + np.dot(clClO2, x[6])
    if int(flag) == 8:
        model = np.dot(o3, x[0]) + np.dot(cl2,x[1]) + np.dot(oClO,x[3]) + np.dot(clOOCl, x[4])


    return np.sum((np.subtract(data,model))**2)

def fitter(data, flag): #fitting algorithm
    global concs, params, res, outer, absnum,flags, fittemps, fitsecs
    out = fmin(residualizer, varsp, args = (xx , absorbancers[int(data)], int(flag)),maxiter = 10000, maxfun = 10000, full_output = 1)
    outer.append(out)
    params.append([out[0][0], out[0][1], out[0][2], out[0][3],out[0][4],out[0][5], out[0][6]])
    concs.append([out[0][0]/91.44,out[0][1]/91.44,out[0][2]/91.44,out[0][3]/91.44,out[0][4]/91.44,out[0][5]/91.44, out[0][6]/91.44])
    res.append(out[1])
    absnum.append(int(data))
    flags.append(int(flag))
    fittemps.append(tempers[data][2])
    fitsecs.append(abssecs[data])




#def residualizer( x): #builds error function
#  
#    model = np.dot(xx[1],x[1]) + np.dot(xx[2],x[2]) + np.dot(xx[3],x[3]) + np.dot(xx[4], x[4]) + np.dot(xx[5], x[5])

#    return np.sum((np.subtract(datar,model))**2)


#def fitter(data): #fitting algorithm
#    global concs, params, res, outer, absnum,flags, fittemps, fitsecs, datar
# 
#    
##    if int(flag) == 1:
##        model = np.dot(oClO,x[3])
##    if int(flag) == 2:
##        model = np.dot(oClO,x[3]) + np.dot(cl2,x[1])
##    if int(flag) == 3:
##        model =  np.dot(cl2,x[1]) + np.dot(oClO,x[3]) + np.dot(clOOCl, x[4])
##    if int(flag) == 4:
##        model =  np.dot(cl2,x[1]) + np.dot(cl2O3,x[5]) + np.dot(oClO,x[3]) + np.dot(clOOCl, x[4]) + x[6]
##    if int(flag) == 5:
##        model = np.dot(cl2,x[1]) + np.dot(clO,x[2]) + np.dot(oClO,x[3]) + np.dot(clOOCl, x[4]), + np.dot(cl2O3, x[5])
##    if int(flag) == 6:
##        model = np.dot(o3, x[0]) + np.dot(cl2,x[1]) + np.dot(clO,x[2]) + np.dot(oClO,x[3]) + np.dot(clOOCl, x[4]), + np.dot(cl2O3, x[5])
#    
##    out = fmin(residualizer, varsp, args = (xx , absorbancers[int(data)], int(flag)),maxiter = 10000, maxfun = 10000, full_output = 1)

#    datar = absorbancers[int(data)]
#    out = minimize(residualizer, varsp,method = 'Nelder-Mead',jac = grad(residualizer), hessp = hvp(residualizer), bounds = bounds, options={'disp': True, 'eps': 0.001, 'maxiter': 1000, 'ftol': 1e-04})
#    outer.append(out)
#    params.append([out[0][0], out[0][1], out[0][2], out[0][3],out[0][4],out[0][5]])
#    concs.append([out[0][0]/91.44,out[0][1]/91.44,out[0][2]/91.44,out[0][3]/91.44,out[0][4]/91.44,out[0][5]/91.44])
#    res.append(out[1])
#    absnum.append(int(data))
#    flags.append(int(flag))
#    fittemps.append(tempers[data][2])
#    fitsecs.append(abssecs[data])
    
def fitdel(): #deletes result of fitter(data,flag)
    global concs, params, res, outer, absnum, flags, fittemps, fitsecs
    concs = []
    params = []
    res = []
    outer = []
    absnum = []
    flags = []
    fittemps = []
    fitsecs = []
    

def plotinit(): #builds the figure window
    global abslineids, absfig, absax1, absxlims, absylims, absax2, abyslimsresid
    if absplotparamlistflag == 1:
        absylimmin = absplotparamlist[2]
        absylimmax = absplotparamlist[3]
        absxlimmin = absplotparamlist[0]
        absxlimmax = absplotparamlist[1]
    else:
        absylimmin = 0
        absylimmax = 1
        absxlimmin = 245
        absxlimmax = 400
    abslineids = []
    absfig, (absax1, absax2) = plt.subplots(nrows=2, ncols = 1, figsize = (10,10))
    absfig.patch.set_facecolor('white')
    absylims = [int(absylimmin), int(absylimmax)]
    absxlims = [int(absxlimmin), int(absxlimmax)]
    absylimsresid = [-0.1, 0.1]
    absax1.set_ylim(absylims)
    absax1.set_xlim(absxlims)
    absax1.grid(which = 'both')
    absax2.set_ylim(absylimsresid)
    absax2.set_xlim(absxlims)
    absax2.grid(which='both')
    
def plotparams(xmin,xmax,ymin,ymax, flag = 0): #sets plot parameters
    global rawplotparamlist, absplotparamlist, rawplotparamlistflag, absplotparamlistflag
    if flag == 0:
        rawplotparamlist = [int(xmin),int(xmax),int(ymin),int(ymax)]
        rawplotparamlistflag = 1
    if flag ==1: 
        absplotparamlist = [int(xmin), int(xmax), int(ymin), int(ymax)]
        absplotparamlistflag = 1
        
def absresplot(samplenum, flag = 0): #displays absorbance and residual
    global abslineids
    absfig
    plt.ion()
    data = absnum[int(samplenum)]
    abslineids.append(int(samplenum))
    li, = absax1.plot(wavers,absorbancers[data])
    if flags[int(samplenum)] == 1:
        li2, = absax2.plot(wavers, (absorbancers[int(data)] - np.dot(concs[int(samplenum)][3]*91.44,OClO)))
    if flags[int(samplenum)] == 2:
        li2, = absax2.plot(wavers, (absorbancers[int(data)]  -   np.dot(concs[int(samplenum)][1]*91.44,Cl2) - np.dot(concs[int(samplenum)][3]*91.44,OClO)))
    if flags[int(samplenum)] == 3:
        li2, = absax2.plot(wavers, (absorbancers[int(data)]  -   np.dot(concs[int(samplenum)][1]*91.44,Cl2) - np.dot(concs[int(samplenum)][3]*91.44,OClO) - np.dot(concs[int(samplenum)][4]*91.44,ClOOCl)))
    if flags[int(samplenum)] == 4:
        li2, = absax2.plot(wavers, (absorbancers[int(data)]  - np.dot(concs[int(samplenum)][1]*91.44,Cl2)-  np.dot(concs[int(samplenum)][5]*91.44,Cl2O3) - np.dot(concs[int(samplenum)][3]*91.44,OClO) - np.dot(concs[int(samplenum)][4]*91.44,ClOOCl)))
    if flags[int(samplenum)] == 5:
        li2, = absax2.plot(wavers, (absorbancers[int(data)]  - np.dot(concs[int(samplenum)][1]*91.44,Cl2)-  np.dot(concs[int(samplenum)][2]*91.44,ClO) - np.dot(concs[int(samplenum)][3]*91.44,OClO) - np.dot(concs[int(samplenum)][4]*91.44,ClOOCl) - np.dot(concs[int(samplenum)][5]*91.44, Cl2O3)))
    if flags[int(samplenum)] == 6:
        li2, = absax2.plot(wavers, (absorbancers[int(data)]  - np.dot(concs[int(samplenum)][0]*91.44, O3) - np.dot(concs[int(samplenum)][1]*91.44,Cl2)-  np.dot(concs[int(samplenum)][2]*91.44,ClO) - np.dot(concs[int(samplenum)][3]*91.44,OClO) - np.dot(concs[int(samplenum)][4]*91.44,ClOOCl) - np.dot(concs[int(samplenum)][5]*91.44, Cl2O3)))
    if flags[int(samplenum)] == 7:
        li2, = absax2.plot(wavers, (absorbancers[int(data)] - np.dot(concs[int(samplenum)][3]*91.44,OClO) - np.dot(concs[int(samplenum)][6]*91.44,ClClO2)))
    if flags[int(samplenum)] == 8:
        li2, = absax2.plot(wavers, (absorbancers[int(data)]  - np.dot(concs[int(samplenum)][0]*91.44, O3) - np.dot(concs[int(samplenum)][1]*91.44,Cl2)- np.dot(concs[int(samplenum)][3]*91.44,OClO) - np.dot(concs[int(samplenum)][4]*91.44,ClOOCl)))

    if flag ==0:
        absax1.set_xlabel('wavelength (nm)', fontsize = 16, color = 'black')
        absax1.set_ylabel('absorbance, base e', fontsize = 16, color = 'black')
        absax1.set_title('OClO Cross Section Experiment', fontsize = 20, color = 'black')
        
        absax2.xaxis.tick_top()
        absax2.set_ylabel('absorbance, base e', fontsize = 16, color = 'black')
        absax2.set_xlabel('residual from fit', fontsize = 20, color = 'black')
    plt.show(block=False)
    
def tracedel(number, flag = 0): #deletes a trace from the (raw/abs)fig graphic - use (raw/abs)lineids to track which is which. Flag = 1 for absorbance, zero, or blank for raw spectra
    if flag !=0:
        absfig
        absax1.lines.remove(absax1.lines[int(number)])
        absax2.lines.remove(absax2.lines[int(number)])
        absfig.canvas.draw()
        abslineids.pop(int(number))
    else:
        absfig    
        absax1.lines.remove(absax1.lines[int(number)])
        absfig.canvas.draw()
        abslineids.pop(int(number))
        
def absplot(samplenum, flag = 0): #converts a value in sample array to absorbance - requires two scans - sample and reference
    global abslineids
    absfig
    plt.ion()
    abslineids.append(int(samplenum))
    li, = absax1.plot(wavers,absorbancers[int(samplenum)])
    if flag ==0:
        absax1.set_xlabel('wavelength (nm)', fontsize = 16, color = 'black')
        absax1.set_ylabel('absorbance, base e', fontsize = 16, color = 'black')
        absax1.set_title('OClO Cross Section Experiment', fontsize = 20, color = 'black')


def tdplotme(x,y, cstrid = 1000, ystrid = 10, cmapp = 'Set1'):
    global fig3d, ax3d, SeZ
    temptemp = np.ndarray.flatten(np.array([float(tempers[l][2]) for l in range(int(x),int(y))]))
    SeX , SeY = np.meshgrid(temptemp, wavers)
    SeZ = np.array([absorbancers[x] for x in range(int(x), int(y))])
    SeZ = SeZ.transpose()
    fig3d = plt.figure()
    ax3d = fig3d.add_subplot(111, projection = '3d')
    ax3d.set_xlabel('temperature (K)', fontsize = 16, labelpad = 12)
    ax3d.set_ylabel('wavelength (nm)', fontsize = 16, labelpad = 12)
    ax3d.set_zlabel('absorbance, base e', fontsize = 16)
    ax3d.plot_surface(SeX,SeY,SeZ, cstride = int(cstrid), rstride = int(ystrid), cmap = cmapp, shade = False, lw = .5)
    plt.show(block = False)
    
def tdadsplotme(x,y, cstrid = 1000, ystrid = 10, cmapp = 'Set1'):
    global fig3d, ax3d
    temptemp = np.ndarray.flatten(np.array([float(tempers[l][3]) for l in range(int(x),int(y))]))
    SeX , SeY = np.meshgrid(temptemp, wavers)
    SeZ = np.array([absorbancers[x] for x in range(int(x), int(y))])
    SeZ = SeZ.transpose()
    fig3d = plt.figure()
    ax3d = fig3d.add_subplot(111, projection = '3d')
    ax3d.set_xlabel('temperature (K)', fontsize = 16, labelpad = 12)
    ax3d.set_ylabel('wavelength (nm)', fontsize = 16, labelpad = 12)
    ax3d.set_zlabel('absorbance, base e', fontsize = 16)
    ax3d.plot_surface(SeX,SeY,SeZ, cstride = int(cstrid), rstride = int(ystrid), cmap = cmapp, shade = False, lw = .5)
    plt.show(block = False)
    
    
def tpdplot(x , y, flag):
    global temptemp, clooclconc, ocloconc, cloconc, cl2conc, Cl2O3conc, data, figlab, xdata,xlab
    temptemp =  [float(fittemps[l]) for l in range(int(x), int(y))]
    clooclconc = [float(concs[l][4]) for l in range(int(x),int(y))]
    ocloconc = [float(concs[l][3]) for l in range(int(x),int(y))]
    cloconc = [float(concs[l][2]) for l in range(int(x),int(y))]
    cl2conc = [float(concs[l][1]) for l in range(int(x),int(y))]
    Cl2O3conc = [float(concs[l][5]) for l in range(int(x), int(y))]
    ClClO2conc = [float(concs[l][6]) for l in range(int(x),int(y))]
    o3conc = [float(concs[l][0]) for l in range(int(x),int(y))]
    xdata = temptemp
    xlab = 'temperature (K)'
    if int(flag) == 1:
        data = cl2conc
        figlab = 'Chlorine'
    if int(flag) == 2:
        data = cloconc
        figlab = '$ClO$'
    if int(flag) == 3:
        data = ocloconc
        figlab = 'OClO'
    if int(flag) == 4:
        data = clooclconc
        figlab = 'ClOOCl'
    if int(flag) == 5:
        data = Cl2O3conc
        figlab = '$Cl_2O_3$'
    if int(flag) == 6:
        data = ClClO2conc
        figlab = '$Cl_2O_2$'
    if int(flag) == 7:
        data = o3conc
        figlab = '$O_3$'
        
    tpdplotme(int(x), int(y))
    
def timepdplot(x,y,flag):
    global secsec, clooclconc, ocloconc, cloconc, cl2conc, data, figlab, xdata,xlab
    secsec = np.ndarray.flatten(np.array([float(abssecs[l]) - float(abssecs[x]) for l in range(int(x),int(y))]))
    clooclconc = [float(concs[l][4]) for l in range(int(x),int(y))]
    ocloconc = [float(concs[l][3]) for l in range(int(x),int(y))]
    cloconc = [float(concs[l][2]) for l in range(int(x),int(y))]
    cl2conc = [float(concs[l][1]) for l in range(int(x),int(y))]
    Cl2O3conc = [float(concs[l][5]) for l in range(int(x), int(y))]
    o3conc = [float(concs[l][0]) for l in range(int(x), int(y))]
    ClClO2conc =[float(concs[l][6]) for l in range(int(x), int(y))]
    xdata = secsec
    xlab = 'time (sec)'
    if int(flag) == 1:
        data = cl2conc
        figlab = 'Chlorine'
    if int(flag) == 2:
        data = cloconc
        figlab = '$ClO$'
    if int(flag) == 3:
        data = ocloconc
        figlab = 'OClO'
    if int(flag) == 4:
        data = clooclconc
        figlab = 'ClOOCl'
    if int(flag) == 5:
        data = Cl2O3conc
        figlab = '$Cl_2O_3$'
    if int(flag) == 6:
        data = ClClO2conc
        figlab = '$Cl_2O_2$'
    if int(flag) == 7:
        data = o3conc
        figlab = '$O_3$'
    tpdplotme(int(x), int(y))
    
def tpdplotme(x , y):
    global fig2d, ax2d
    fig2d = plt.figure()
    ax2d = fig2d.add_subplot(111)
    ax2d.plot(xdata, data)
    fig2d.patch.set_facecolor('white')
    ax2d.set_xlabel(xlab, fontsize = 16)
    ax2d.set_ylabel('number density - ' + figlab + ' (#/$cm^3$)' , fontsize = 16)
    ax2d.set_title('tpd trace - ' + figlab, fontsize = 20)
    plt.show(block = False)
    
    

def timeplotme(x,y, cstrid = 1000, ystrid = 10, cmapp = 'Set1'):
    global figtime3d, axtime3d
    secsec = np.ndarray.flatten(np.array([float(abssecs[l]) - float(abssecs[x]) for l in range(int(x),int(y))]))
    SeX , SeY = np.meshgrid(secsec, wavers)
    Secon = np.meshgrid(np.ones(len(secsec)),np.ones(len(wavers)))
    SeZ = np.array([absorbancers[x] for x in range(int(x), int(y))])
    SeZ = SeZ.transpose()
    figtime3d = plt.figure()
    axtime3d = figtime3d.add_subplot(111, projection = '3d')
    axtime3d.set_xlabel('time (sec)', fontsize = 16, labelpad = 12)
    axtime3d.set_ylabel('wavelength (nm)', fontsize = 16, labelpad = 12)
    axtime3d.set_zlabel('absorbance, base e', fontsize = 16)
    axtime3d.plot_surface(SeX,SeY,SeZ, cstride = int(cstrid), rstride = int(ystrid), cmap = cmapp, shade = False, lw = .5)
    plt.show(block = False)
    
def ramp(x,y):
    global rampfig, rampax
    secsec = np.ndarray.flatten(np.array([float(secondser[l]) - float(secondser[x]) for l in range(int(x),int(y))]))
    temptemp = np.ndarray.flatten(np.array([float(tempers[l][2]) for l in range(int(x),int(y))]))
    rampfig, rampax = plt.subplots()
    rampfig.patch.set_facecolor('white')
    rampax.plot(secsec, temptemp, 'ko')
    rampax.set_ylabel('temperature (K)', fontsize = 16)
    rampax.set_xlabel('time (sec)', fontsize = 16)
    rampax.set_title('Temperature ramp', fontsize = 20)
    plt.show(block = False)
    

def adsramp(x,y):
    global rampfig, rampax
    secsec = np.ndarray.flatten(np.array([float(secondser[l]) - float(secondser[x]) for l in range(int(x),int(y))]))
    temptemp = np.ndarray.flatten(np.array([float(tempers[l][3]) for l in range(int(x),int(y))]))
    rampfig, rampax = plt.subplots()
    rampfig.patch.set_facecolor('white')
    rampax.plot(secsec, temptemp, 'ko')
    rampax.set_ylabel('temperature (K)', fontsize = 16)
    rampax.set_xlabel('time (sec)', fontsize = 16)
    rampax.set_title('Temperature ramp', fontsize = 20)
    plt.show(block = False)

def fitconstruct(x,y, cstrid = 1000, ystrid = 10, cmapp = 'Set1'):
    global fig3d, ax3d, SeZZ
    temptemp = np.ndarray.flatten(np.array([float(fittemps[l]) for l in range(int(x),int(y))]))
    SeX , SeY = np.meshgrid(temptemp, wavers)
    SeZZ = []
    for z in range(x,y):
        if flags[int(z)] == 1:
            SeZZ.append(np.array(np.dot(concs[int(z)][3]*91.44,OClO)))
        if flags[int(z)] == 2:
            SeZZ.append(np.array(np.dot(concs[int(z)][1]*91.44,Cl2) + np.dot(concs[int(z)][3]*91.44,OClO)))
        if flags[int(z)] == 3:
            SeZZ.append(np.array(np.dot(concs[int(z)][1]*91.44,Cl2) + np.dot(concs[int(z)][3]*91.44,OClO) + np.dot(concs[int(z)][4]*91.44,ClOOCl)))
        if flags[int(z)] == 4:
            SeZZ.append(np.array(np.dot(concs[int(z)][1]*91.44,Cl2)+  np.dot(concs[int(z)][5]*91.44,Cl2O3) + np.dot(concs[int(z)][3]*91.44,OClO) + np.dot(concs[int(z)][4]*91.44,ClOOCl)))
        if flags[int(z)] == 5:
            SeZZ.append(np.array(np.dot(concs[int(z)][1]*91.44,Cl2)+  np.dot(concs[int(z)][2]*91.44,ClO) + np.dot(concs[int(z)][3]*91.44,OClO) + np.dot(concs[int(z)][4]*91.44,ClOOCl) + np.dot(concs[int(z)][5]*91.44, Cl2O3)))
        if flags[int(z)] == 6:
            SeZZ.append(np.array(np.dot(concs[int(z)][0]*91.44, O3) + np.dot(concs[int(z)][1]*91.44,Cl2)+  np.dot(concs[int(z)][2]*91.44,ClO) + np.dot(concs[int(z)][3]*91.44,OClO) + np.dot(concs[int(z)][4]*91.44,ClOOCl) + np.dot(concs[int(z)][5]*91.44, Cl2O3)))
        if flags[int(z)] == 8:
            SeZZ.append(np.array(np.dot(concs[int(z)][0]*91.44, O3) + np.dot(concs[int(z)][1]*91.44,Cl2) + np.dot(concs[int(z)][3]*91.44,OClO) + np.dot(concs[int(z)][4]*91.44,ClOOCl)))
    SeZZ = np.array(SeZZ)
    SeZZ = np.transpose(SeZZ)
    fig3d = plt.figure()
    ax3d = fig3d.add_subplot(111, projection = '3d')
    ax3d.set_xlabel('temperature (K)', fontsize = 16, labelpad = 12)
    ax3d.set_ylabel('wavelength (nm)', fontsize = 16, labelpad = 12)
    ax3d.set_zlabel('absorbance, base e', fontsize = 16)
    ax3d.set_title('simulated TPD', fontsize = 30)
    ax3d.plot_surface(SeX,SeY,SeZZ, cstride = int(cstrid), rstride = int(ystrid), cmap = cmapp, shade = False, lw = .5)
    plt.show(block = False)
    
def fiterrorplot(x,y, cstrid = 1000, ystrid = 10, cmapp = 'Set1'):
    global fig3d, ax3d
    temptemp = np.ndarray.flatten(np.array([float(fittemps[l]) for l in range(int(x),int(y))]))
    SeX , SeY = np.meshgrid(temptemp, wavers)
    SeZZZ = SeZ - SeZZ
    fig3d = plt.figure()
    ax3d = fig3d.add_subplot(111, projection = '3d')
    ax3d.set_xlabel('temperature (K)', fontsize = 16, labelpad = 12)
    ax3d.set_ylabel('wavelength (nm)', fontsize = 16, labelpad = 12)
    ax3d.set_zlabel('absorbance, base e', fontsize = 16)
    ax3d.set_title('residual from TPD fit', fontsize = 30)
    ax3d.plot_surface(SeX,SeY,SeZZZ, cstride = int(cstrid), rstride = int(ystrid), cmap = cmapp, shade = False, lw = .5)
    plt.show(block = False)
    
def paramgen():
    global params
    params = Parameters()
    params.add('o3',value = varsp[0], min = 0, max = 1e20)
    params.add('cl2', value = varsp[1], min = 0, max = 1e20)
    params.add('clo', value = varsp[2], min = 0, max = 1e18)
    params.add('oclo', value = varsp[3], min = 0, max = 1e18)
    params.add('cloocl', value = varsp[4], min = 0, max = 1e18)
    params.add('cl2o3', value = varsp[5], min = 0, max = 1e17)
    params.add('clclo2',value  = varsp[6], min = 0, max = 1e17)

    
def limfitresid(params, data, flag): #builds error function
    v = params.valuesdict()
    
    if int(flag) == 1:
        model = np.dot(v['oclo'],OClO)
    if int(flag) == 2:
        model = np.dot(v['oclo'],OClO) + np.dot(v['cl2'],Cl2)
    if int(flag) == 3:
        model =  np.dot(v['cl2'],Cl2) + np.dot(v['oclo'],OClO) + np.dot(v['cloocl'],ClOOCl)
    if int(flag) == 4:
        model =  np.dot(v['cl2'],Cl2) + np.dot(v['cl2o3'],Cl2O3) + np.dot(v['oclo'],OClO) + np.dot(v['cloocl'],ClOOCl)
    if int(flag) == 5:
        model = np.dot(v['cl2'],Cl2) + np.dot(v['clo'],ClO) + np.dot(v['oclo'],OClO) + np.dot(v['cloocl'],ClOOCl) + np.dot(v['cl2o3'],Cl2O3)
    if int(flag) == 6:
        model = np.dot(v['o3'],O3) + np.dot(v['cl2'],Cl2) + np.dot(v['clo'],ClO) + np.dot(v['oclo'],OClO) + np.dot(v['cloocl'],ClOOCl) + np.dot(v['cl2o3'],Cl2O3)
    if int(flag) == 7:
        model = np.dot(v['oclo'], OClO) + np.dot(v['clclo2'],ClClO2)
    if int(flag) == 8:
        model = np.dot(v['o3'],O3) + np.dot(v['cl2'],Cl2) + np.dot(v['oclo'],OClO) + np.dot(v['cloocl'],ClOOCl)
    return np.sum(np.subtract(data,model)**2)

def limfitter(data, flag): #fitting algorithm
    global concs, params, res, outer, absnum,flags, fittemps, fitsecs, fitval

    

        
    out = minimize(limfitresid, params, args = ( absorbancers[int(data)].transpose(), int(flag)), method = 'differential_evolution')
    outer.append(out)
    fitval.append([out.params['o3'].value, out.params['cl2'].value, out.params['clo'].value, out.params['oclo'].value,out.params['cloocl'].value,out.params['cl2o3'].value, out.params['clclo2'].value])
    concs.append([float(out.params['o3'].value) / 91.44, float(out.params['cl2'].value) / 91.44, float(out.params['clo'].value) / 91.44, float(out.params['oclo'].value) / 91.44,float(out.params['cloocl'].value) / 91.44,float(out.params['cl2o3'].value) / 91.44, float(out.params['clclo2'].value) / 91.44])
    res.append(out.residual)
    absnum.append(int(data))
    flags.append(int(flag))
    fittemps.append(tempers[data][2])
    fitsecs.append(abssecs[data])


