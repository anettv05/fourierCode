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
fourier_transform = np.fft.fft(data)
# Compute the frequencies corresponding to the FFT
frequencies = np.fft.fftfreq(len(data), 1/sample_rate)


# Plot the Fourier transform magnitude
plt.figure(figsize=(10, 4))
plt.title('Fourier Transform of Audio Signal')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.plot(frequencies[:len(frequencies)//2], np.abs(fourier_transform)[:len(frequencies)//2])
plt.grid(True)
plt.show()

