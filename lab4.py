import scipy.io.wavfile as wav
import matplotlib.pyplot as matplot
import matplotlib.patches as mpatches
import numpy as np
import math
from scipy import fftpack, integrate
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')


def generateCarrier(d, f):
    n = np.size(d)
    T = n/f     #Tiempo
    fc = 400000    #Frecuencia de la portadora
    tc = np.arange(0, T, 1.0/(fc))
    carrier = np.cos(2*f*np.pi*tc)
    return (carrier, tc, fc)

#Función que calcula la modulación AM de la señal signal, con frecuencia f e indice de modulacion M
def AMmod(signal, f, M):
    M = M/100.0   #Índice de modulación
    if(M>=1): M = M-1
    if(M<1): M = M+1
    Amax = np.amax(abs(signal))  #Se guarda amplitud máxima
    Amin = np.amin(abs(signal))  #Se guarda amplitud mínima
    Ac = (Amax-Amin)/M      #Amplitud portadora
    n = np.size(signal)
    T = n/f     #Tiempo
    carrier, tc, fc = generateCarrier(signal, f)    #Se genera la portadora
    told = np.arange(0, T, 1.0/f)
    tnew = np.arange(0, T, 1.0/(fc))
    signalInterp = np.interp(tnew, told, signal)   #Se interpola la señal original
    datamod = (Ac + M*signalInterp)*carrier + Amax  #Se calcula señal modulada
    return (datamod, tnew, signalInterp, carrier)

#Función que calcula la modulación FM de la señal signal, con frecuencia f e indice de modulacion M
def FMmod(signal, f, M):
    k = M/100   #Índice de modulación
    n = np.size(signal)
    T = n/f
    fc = 4000000
    tc = np.arange(0, T, 1.0/(fc))
    carrier = np.cos(2*f*np.pi*tc)
    told = np.arange(0, T, 1.0/f)
    tnew = np.arange(0, T, 1.0/(fc))
    t = np.linspace(0, n, np.size(tc))
    signalInterp = np.interp(tnew, told, signal)    #Se interpola la señal original
    integral = integrate.cumtrapz(signalInterp, tc, initial=0)  
    fmMod = np.cos(2*np.pi*f*tc + k*integral)   #Se calcula señal modulada
    return (fmMod, tnew, signalInterp, carrier)

#Función que realiza la demodulación de la señal s (calculando la envolvente)
def AMdemod(s):
    envolvente = np.zeros(s.shape)

    u_x = [0,]
    u_y = [s[0],]

    l_x = [0,]
    l_y = [s[0],]

    for k in range(1,len(s)-1):
        if (np.sign(s[k]-s[k-1])==1) and (np.sign(s[k]-s[k+1])==1):
            u_x.append(k)
            u_y.append(s[k])

        if (np.sign(s[k]-s[k-1])==-1) and ((np.sign(s[k]-s[k+1]))==-1):
            l_x.append(k)
            l_y.append(s[k])

    u_x.append(len(s)-1)
    u_y.append(s[-1])

    l_x.append(len(s)-1)
    l_y.append(s[-1])

    u_p = interp1d(u_x,u_y, kind = 'linear',bounds_error = False, fill_value=0.0)
    l_p = interp1d(l_x,l_y,kind = 'linear',bounds_error = False, fill_value=0.0)

    for k in range(0,len(s)):
        envolvente[k] = u_p(k)

    for k in range(0,len(s)):
        envolvente[k] = u_p(k) 
        
    return envolvente


if __name__ == '__main__':

    
	# Se leen los archivos de audio, se guarda la frecuencia en f y las muestras en data
    f, signal = wav.read('handel.wav')
    signal = signal[5000:5100]


    Amax = np.amax(abs(signal))  #Se guarda amplitud máxima
    

    #Se obtienen señales moduladas y demoduladas en amplitud
    mod15, t, signalInterp, carrier = AMmod(signal, f, 15)
    demod15 = AMdemod(mod15) - 2*Amax
    mod15 = mod15 - Amax

    mod100, t, signalInterp, carrier = AMmod(signal, f, 100)
    demod100 = AMdemod(mod100) - 2*Amax
    mod100 = mod100 - Amax

    mod125, t, signalInterp, carrier = AMmod(signal, f, 125)
    demod125 = AMdemod(mod125) - 2*Amax
    mod125 = mod125 - Amax


    
    
    #Se grafican las señales moduladas por amplitud
    matplot.plot(t, mod15)
    matplot.xlabel('Tiempo (s)')
    matplot.ylabel('Amplitud')
    matplot.title('Señal modulada por amplitud con porcentaje de modulación de 15%')
    matplot.tight_layout()
    matplot.show()
    matplot.plot(t, mod100)
    matplot.xlabel('Tiempo (s)')
    matplot.ylabel('Amplitud')
    matplot.title('Señal modulada por amplitud con porcentaje de modulación de 100%')
    matplot.tight_layout()
    matplot.show()
    matplot.plot(t, mod125)
    matplot.xlabel('Tiempo (s)')
    matplot.ylabel('Amplitud')
    matplot.title('Señal modulada por amplitud con porcentaje de modulación de 125%')
    matplot.tight_layout()
    matplot.show()

    #Se grafican las señales demoduladas por amplitud
    matplot.rcParams["figure.figsize"] = [16,9]
    matplot.plot(t, signalInterp, color='b')
    matplot.plot(t, demod15, color='g')
    matplot.plot(t, demod100, color='r')
    blue_patch = mpatches.Patch(color='b', label='Señal original')
    green_patch = mpatches.Patch(color='g', label='Señal recuperada de modulación del 15%')
    red_patch = mpatches.Patch(color='r', label='Señal recuperada de modulación del 100%')
    matplot.legend(handles=[blue_patch, green_patch, red_patch])
    matplot.xlabel('Tiempo (s)')
    matplot.ylabel('Amplitud')
    matplot.title('Señal original v/s demoduladas en 15% y 100%')
    matplot.tight_layout()
    matplot.show()

    matplot.rcParams["figure.figsize"] = [16,9]
    matplot.plot(t, signalInterp, color='b')
    matplot.plot(t, demod125, color='purple')
    blue_patch = mpatches.Patch(color='b', label='Señal original')
    purple_patch = mpatches.Patch(color='purple', label='Señal recuperada de modulación del 125%')
    matplot.legend(handles=[blue_patch, purple_patch])
    matplot.xlabel('Tiempo (s)')
    matplot.ylabel('Amplitud')
    matplot.title('Señal original v/s demodulada en 125%')
    matplot.tight_layout()
    matplot.show()


    #Se obtienen señales moduladas por frecuencia
    fm_mod15, t, signalInterp, carrier = FMmod(signal, f, 15)

    fm_mod100, t, signalInterp, carrier = FMmod(signal, f, 100)

    fm_mod125, t, signalInterp, carrier = FMmod(signal, f, 125)


    #Se grafican señales moduladas por frecuencia
    matplot.rcParams["figure.figsize"] = [16,9]
    matplot.plot(t, fm_mod15)
    matplot.xlabel('Tiempo (s)')
    matplot.ylabel('Amplitud')
    matplot.title('Señal modulada por frecuencia con porcentaje de modulación de 15%')
    matplot.show()
    matplot.plot(t, fm_mod100)
    matplot.xlabel('Tiempo (s)')
    matplot.ylabel('Amplitud')
    matplot.title('Señal modulada por frecuencia con porcentaje de modulación de 100%')
    matplot.show()
    matplot.plot(t, fm_mod125)
    matplot.xlabel('Tiempo (s)')
    matplot.ylabel('Amplitud')
    matplot.title('Señal modulada por frecuencia con porcentaje de modulación de 125%')
    matplot.show()



    #Se obtiene el espectro de frecuencias de cada modulación
    mod15_fft = fftpack.fft(mod15)
    mod100_fft = fftpack.fft(mod100)
    mod125_fft = fftpack.fft(mod125)

    fm_mod15_fft = fftpack.fft(fm_mod15)
    fm_mod100_fft = fftpack.fft(fm_mod100)
    fm_mod125_fft = fftpack.fft(fm_mod125)

    #Gráficos de espectros AM
    f, axarr = matplot.subplots(2)
    axarr[0].set_title('Espectro de señal modulada AM al 15%')
    axarr[0].plot(mod15_fft)
    axarr[0].axvline(x=20, color='r')
    axarr[0].axvline(x=180, color='r')
    axarr[0].set_xlabel('Frecuencia (Hz)')
    axarr[0].set_ylabel('Amplitud')
    axarr[1].axvline(x=85, color='g', linestyle='dashed')
    axarr[1].axvline(x=115, color='g', linestyle='dashed')
    axarr[1].plot(mod15_fft[0:250])
    axarr[1].set_xlabel('Frecuencia (Hz)')
    axarr[1].set_ylabel('Amplitud')
    matplot.tight_layout()
    matplot.show()

    f, axarr = matplot.subplots(2)
    axarr[0].set_title('Espectro de señal modulada AM al 100%')
    axarr[0].plot(mod100_fft)
    axarr[0].axvline(x=20, color='r')
    axarr[0].axvline(x=180, color='r')
    axarr[0].set_xlabel('Frecuencia (Hz)')
    axarr[0].set_ylabel('Amplitud')
    axarr[1].axvline(x=85, color='g', linestyle='dashed')
    axarr[1].axvline(x=115, color='g', linestyle='dashed')
    axarr[1].plot(mod100_fft[0:250])
    axarr[1].set_xlabel('Frecuencia (Hz)')
    axarr[1].set_ylabel('Amplitud')
    matplot.tight_layout()
    matplot.show()

    f, axarr = matplot.subplots(2)
    axarr[0].set_title('Espectro de señal modulada AM al 125%')
    axarr[0].plot(mod125_fft)
    axarr[0].axvline(x=20, color='r')
    axarr[0].axvline(x=180, color='r')
    axarr[0].set_xlabel('Frecuencia (Hz)')
    axarr[0].set_ylabel('Amplitud')
    axarr[1].axvline(x=85, color='g', linestyle='dashed')
    axarr[1].axvline(x=115, color='g', linestyle='dashed')
    axarr[1].plot(mod125_fft[0:250])
    axarr[1].set_xlabel('Frecuencia (Hz)')
    axarr[1].set_ylabel('Amplitud')
    matplot.tight_layout()
    matplot.show()

    #Gráficos de espectros FM
    f, axarr = matplot.subplots(2)
    axarr[0].set_title('Espectro de señal modulada FM al 15%')
    axarr[0].plot(fm_mod15_fft)
    axarr[0].axvline(x=-600, color='r')
    axarr[0].axvline(x=600, color='r')
    axarr[0].set_xlabel('Frecuencia (Hz)')
    axarr[0].set_ylabel('Amplitud')
    axarr[1].axvline(x=85, color='g', linestyle='dashed')
    axarr[1].axvline(x=115, color='g', linestyle='dashed')
    axarr[1].plot(fm_mod15_fft[0:250])
    axarr[1].set_xlabel('Frecuencia (Hz)')
    axarr[1].set_ylabel('Amplitud')
    matplot.tight_layout()
    matplot.show()

    f, axarr = matplot.subplots(2)
    axarr[0].set_title('Espectro de señal modulada FM al 100%')
    axarr[0].plot(fm_mod100_fft)
    axarr[0].axvline(x=-600, color='r')
    axarr[0].axvline(x=900, color='r')
    axarr[0].set_xlabel('Frecuencia (Hz)')
    axarr[0].set_ylabel('Amplitud')
    axarr[1].axvline(x=40, color='g', linestyle='dashed')
    axarr[1].axvline(x=160, color='g', linestyle='dashed')
    axarr[1].plot(fm_mod100_fft[0:250])
    axarr[1].set_xlabel('Frecuencia (Hz)')
    axarr[1].set_ylabel('Amplitud')
    matplot.tight_layout()
    matplot.show()

    f, axarr = matplot.subplots(2)
    axarr[0].set_title('Espectro de señal modulada FM al 125%')
    axarr[0].plot(fm_mod125_fft)
    axarr[0].axvline(x=-600, color='r')
    axarr[0].axvline(x=600, color='r')
    axarr[0].set_xlabel('Frecuencia (Hz)')
    axarr[0].set_ylabel('Amplitud')
    axarr[1].axvline(x=40, color='g', linestyle='dashed')
    axarr[1].axvline(x=160, color='g', linestyle='dashed')
    axarr[1].plot(fm_mod125_fft[0:250])
    axarr[1].set_xlabel('Frecuencia (Hz)')
    axarr[1].set_ylabel('Amplitud')
    matplot.tight_layout()
    matplot.show()




 







