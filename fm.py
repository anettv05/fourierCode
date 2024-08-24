
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
from scipy.io import wavfile
import sounddevice as sd
from scipy.io import wavfile

# Load the audio file
audio_file = '/Users/anettvarghese/Downloads/Test rec.wav' #add the path of the audio file it should be in .wav format
sample_rate, data = wavfile.read(audio_file)
kp=10*np.pi
kf=2*np.pi*10**5
#fm=5000  # message signal frequency
t = np.linspace(0,len(data)/sample_rate,len(data))
#mt = np.sign(np.sin(2*fm*np.pi*t)) # message signal
mt = data
wc = 2* np.pi * 10**9
amp = np.sin(wc*t+kp*mt)       # phase transform
imt = np.cumsum(mt)            # integrated m(t)
amp2 = np.sin(wc*t+kf*imt)     # frequency transform
wi1 = np.gradient(wc*t+kp*mt, t)  # derivative of phase of PM
wi2 = np.gradient(wc*t+kf*imt, t) # derivative of phase of FM
fourier_transform = np.fft.fft(data)
frequencies = np.fft.fftfreq(len(data), 1/sample_rate)
demo = data
plt.figure(figsize=(10,6))
plt.subplot(4,1,1)
plt.plot(t,amp)
plt.title("Phase modulation")
plt.xlabel("time")
plt.ylabel("amp")

plt.subplot(4,1,3)
plt.plot(t,amp2)
plt.title("Freq modulation")
plt.xlabel("time")
plt.ylabel("amp")

plt.subplot(4,1,2)
plt.plot(t,wi1)
plt.title("Phase modulation inst freq")
plt.xlabel("time")
plt.ylabel("freq")

plt.subplot(4,1,4)
plt.plot(t,wi2)
plt.title("Freq modulation inst freq")
plt.xlabel("time")
plt.ylabel("freq")
plt.tight_layout()
plt.show()


plt.subplot(4,1,1)
plt.plot(frequencies,fourier_transform)
plt.title("Freq demodulation")
plt.xlabel("time")
plt.ylabel("amp")

plt.subplot(4,1,2)
plt.plot(t,data)
plt.title("Freq demodulation")
plt.xlabel("time")
plt.ylabel("amp")

plt.subplot(4,1,3)
plt.plot(t,demo)
plt.title("Freq demodulation")
plt.xlabel("time")
plt.ylabel("amp")
fourier_transform = np.fft.fft(demo)

plt.subplot(4,1,4)
plt.plot(frequencies,fourier_transform)
plt.title("Freq demodulation")
plt.xlabel("time")
plt.ylabel("amp")
plt.tight_layout()
plt.show()
#sd.play(demo,sample_rate)
sd.wait()