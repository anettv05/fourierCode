import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
from scipy.io import wavfile


# Load the audio file
audio_file = '/Users/anettvarghese/Downloads/Test rec.wav' #add the path of the audio file it should be in .wav format
sample_rate, data = wavfile.read(audio_file)
fc = 900* 10**3
wc = 2* np.pi * fc
A = 1
# Convert stereo to mono if the audio is stereo
'''
if len(data.shape) > 1:
    data = data.mean(axis=1)
'''
t = np.linspace(0,len(data)/sample_rate,len(data))

# Compute the Fourier transform
data = data/max(data)

# Compute the frequencies corresponding to the FFT
frequencies = np.fft.fftfreq(len(data), 1/sample_rate)
ampmod = (data+A) * np.cos(wc * t)

'''
# Plot the Fourier transform magnitude
plt.figure(figsize=(10, 4))
plt.title('Fourier Transform of Audio Signal')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.plot(frequencies[:len(frequencies)//2], np.abs(fourier_transform)[:len(frequencies)//2])
plt.grid(True)
'''
#fourier of audio
fourier_transform = np.fft.fft(data)
plt.subplot(2,1,1)
plt.title(' Audio Signal')
plt.xlabel('Time')
plt.ylabel('Magnitude')
plt.plot(t, data)
plt.grid(True)

plt.subplot(2,1,2)
plt.title('Fourier Transform of Audio Signal')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.plot(frequencies[:len(frequencies)//2], np.abs(fourier_transform)[:len(frequencies)//2])
plt.grid(True)
plt.tight_layout()
plt.show()

#plt.figure(figsize=(10, 4))
plt.subplot(2,1,1)
plt.title('Amp modulation of Audio Signal')
plt.xlabel('Time')
plt.ylabel('Magnitude')
plt.plot(t, ampmod)
plt.grid(True)
fourier_transform = np.fft.fft(ampmod)
# Plot the Fourier AM
plt.subplot(2,1,2)
plt.title('Fourier Transform of AM Signal')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.plot(frequencies[:len(frequencies)//2], np.abs(fourier_transform)[:len(frequencies)//2])
plt.grid(True)
plt.tight_layout()
plt.show()

demSig = ampmod* np.cos(wc*t)
fourier_transform = np.fft.fft(demSig)
plt.subplot(2,1,1)
plt.title('Demodulation of AM Signal')
plt.xlabel('Time')
plt.ylabel('Magnitude')
plt.plot(t, demSig)
plt.grid(True)


# Plot the Fourier demodulation magnitude
plt.subplot(2,1,2)
plt.title('Fourier Transform of demodulated Signal')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.plot(frequencies[:len(frequencies)//2], np.abs(fourier_transform)[:len(frequencies)//2])
plt.grid(True)
plt.tight_layout()
plt.show()
recont = (demSig * 2 )/(1+np.cos(2*wc*t))

plt.subplot(2,1,1)
plt.title('Demodulation of AM Signal')
plt.xlabel('Time')
plt.ylabel('Magnitude')
plt.plot(t, recont)
plt.grid(True)

plt.subplot(2,1,2)
plt.title('Data of AM Signal')
plt.xlabel('Time')
plt.ylabel('Magnitude')
plt.plot(t, data)
plt.grid(True)
plt.tight_layout()
plt.show()
demodulated_signal = np.int16(recont * 32767)
sd.play(data,sample_rate)
sd.wait()
sd.play(recont,sample_rate)
sd.wait()