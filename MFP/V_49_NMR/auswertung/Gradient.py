import numpy as np
import matplotlib.pyplot as plt
#Laden der Daten aus der Datei "echo_gradient.csv"
#Die erste Spalte enthält die Zeiten in Sekunden, die zweite Spalte
#den Realteil und die dritte Spalte den Imaginärteil
#data = np.loadtxt("../Data/scope_7.csv", delimiter=",", skiprows=3, unpack= True)
data = np.genfromtxt("../Data/scope_7.csv", delimiter=",")


plt.plot(data[:,0],data[:,1],"rx")
plt.plot(data[:,0],data[:,2],"bx")
plt.show()
plt.close()


times = data[:,0]
real = data[:,1]
imag = data[:,2]
#Suchen des Echo-Maximums und alle Daten davor abschneiden
start = np.argmax(real)
times = times[start:] 

real = real[start:]
imag = imag[start:]
#Phasenkorrektur - der Imaginärteil bei t=0 muss = 0 sein
phase = np.arctan2(imag[0], real[0])
#Daten in komplexes Array mit Phasenkorrektur speichern
compsignal = (real*np.cos(phase)+imag*np.sin(phase))+ \
(-real*np.sin(phase)+imag*np.cos(phase))*1j
#Offsetkorrektur, ziehe den Mittelwert der letzten 512 Punkte von allen Punkten ab
compsignal = compsignal - compsignal[-512:-1].mean()
#Der erste Punkt einer FFT muss halbiert werden
compsignal[0] = compsignal[0]/2.0
#Anwenden einer Fensterfunktion (siehe z. Bsp.
#https://de.wikipedia.org/wiki/Fensterfunktion )
#Hier wird eine Gaußfunktion mit sigma = 100 Hz verwendet
apodisation = 100.0*2*np.pi
compsignal = compsignal*np.exp(-1.0/2.0*((times-times[0])*apodisation)**2)
#Durchführen der Fourier-Transformation
fftdata = np.fft.fftshift(np.fft.fft(compsignal))
#Generieren der Frequenzachse
freqs = np.fft.fftshift(np.fft.fftfreq(len(compsignal), times[1]-times[0]))
#Speichern des Ergebnisses als txt
np.savetxt("echo_gradient_fft.txt", np.array([freqs, np.real(fftdata), \
np.imag(fftdata)]).transpose())
#Erstellen eines Plots
plt.plot(freqs, np.real(fftdata))
plt.savefig("echo_gradient.pdf") 