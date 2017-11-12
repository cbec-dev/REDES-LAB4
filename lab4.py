import scipy.io.wavfile as wav
import matplotlib.pyplot as matplot
import numpy as np
import math
from scipy import fftpack, integrate
import warnings


def generateCarrier(d, f):
    n = np.size(d)
    T = n/f     #Tiempo
    fc = 4000000    #Frecuencia de la portadora (4 veces la frecuencia de la señal)
    tc = np.arange(0, T, 1.0/(fc))
    carrier = np.cos(2*f*np.pi*tc)
    matplot.plot(carrier)
    matplot.show()
    return (carrier, tc, fc)

def AMmod(signal, f, M):
    M = M/100   #Porcentaje de modulación
    Amax = np.max(abs(signal))  #Se guarda amplitud máxima
    n = np.size(signal)
    T = n/f     #Tiempo
    carrier, tc, fc = generateCarrier(signal, f)    #Se genera la portadora
    carrier = carrier * M
    told = np.arange(0, T, 1.0/f)
    tnew = np.arange(0, T, 1.0/(fc))
    signalInterp = np.interp(tnew, told, signal)   #Se interpola la señal original
    datamod = (1 + signalInterp)*carrier
    signalInterp = signalInterp + Amax  #Shift Up para mantener señales sobre el eje y
    datamod = datamod + Amax
    return (datamod, tnew, signalInterp, carrier)

def FMmod(signal, f, M):
    M = M/100
    n = np.size(signal)
    T = n/f
    fc = 4000000
    tc = np.arange(0, T, 1.0/(fc))
    carrier = np.cos(2*f*np.pi*tc)
    told = np.arange(0, T, 1.0/f)
    tnew = np.arange(0, T, 1.0/(fc))
    t = np.linspace(0, n, np.size(tc))
    print(np.size(t))
    print(np.size(tc))
    signalInterp = np.interp(tnew, told, signal)
    integral = integrate.cumtrapz(signalInterp, tc, initial=0)
    k = 1.0
    fmMod = np.cos(2*np.pi*f*tc + k*integral)
    return (fmMod, tnew, signalInterp, carrier)



if __name__ == '__main__':

    
	# Se leen los archivos de audio, se guarda la frecuencia en f y las muestras en data
    f, signal = wav.read('handel.wav')
    signal = signal[5000:5050]
    

    fm, tnew, signalInterp, carrier = AMmod(signal, f, 100)
    

    fp, axarr = matplot.subplots(3)
    axarr[0].plot(signalInterp)
    axarr[0].set_title('Señal')
    axarr[1].plot(carrier)
    axarr[1].set_title('carrier')
    axarr[2].plot(fm)
    axarr[2].set_title('fm')

    matplot.show()


 







