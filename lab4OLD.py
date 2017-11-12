import scipy.io.wavfile as wav
import matplotlib.pyplot as matplot
import numpy as np
import math
from scipy import fftpack
import warnings

def plotDataTime(d, f):     # Función para graficar en el tiempo
    n = len(d)
    T = n/f   # duración del audio
    t = np.arange(0,T, 1.0/f)
    matplot.plot(t, d)
    matplot.xlabel('Tiempo (s)')
    matplot.ylabel('Amplitud')
    matplot.show()

def generateCarrier(d, f):
    n = np.size(d)
    T = n/f     #Tiempo
    fc = 4*f    #Frecuencia de la portadora (4 veces la frecuencia de la señal)
    tc = np.arange(0, T, 1.0/(fc))
    carrier = np.cos(2*f*np.pi*tc)
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
    signalInterp = signalInterp + Amax  #Shif Up para mantener señales sobre el eje y
    datamod = datamod + Amax
    return (datamod, tnew, signalInterp)



if __name__ == '__main__':

    
	# Se leen los archivos de audio, se guarda la frecuencia en f y las muestras en data
    f, signal = wav.read('handel.wav')
    signal = signal[4000:4050]
    
    #carrier, tc, fc = generateCarrier(signal, f)
    #signal_mod, td, signal_interp = AMmod(signal, f, 150)

    #FM MOD
    modulator_frequency = 4.0
    fc = 40000
    modulation_index = 1.0

    n = len(signal)
    T = n/f   # duración del audio
    t = np.arange(0,T, 1.0/f)

    modulator = signal
    carrier = np.sin(2.0*np.pi*fc*t)
    product = np.zeros_like(modulator)

    for i,t in enumerate(t):
        product[i] = np.sin(2.0*np.pi*(fc*t+modulator[i]))

#    product = fftpack.fft(product)
    fp, axarr = matplot.subplots(3)
    axarr[0].plot(modulator)
    axarr[0].set_title('Señal')
    axarr[1].plot(carrier)
    axarr[1].set_title('Carrier')
    axarr[2].plot(product)
#    axarr[1].plot(time, modulator, color='r')
    axarr[2].set_title('Señal modulada')

    matplot.show()


 








#    plotDataTime(data, f)

#    dmod = carrier*data

#    plotDataTime(dmod, f)