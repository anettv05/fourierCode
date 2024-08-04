import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

# Load the audio file
audio_file = '/Users/anettvarghese/Downloads/Test rec.wav' #add the path of the audio file it should be in .wav format
sample_rate, data = wavfile.read(audio_file)

# Convert stereo to mono if the audio is stereo
if len(data.shape) > 1:
    data = data.mean(axis=1)

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
