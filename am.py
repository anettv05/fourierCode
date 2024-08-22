import numpy as np
import sounddevice as sd
from scipy.io import wavfile
from scipy.signal import butter, lfilter

# Step 1: Read the audio file
sample_rate, data = wavfile.read('/Users/anettvarghese/Downloads/Test rec.wav')

# Step 2: Demodulate the signal (assuming AM demodulation)
fc = 90 * 10**6  # Carrier frequency
wc = 2 * np.pi * fc
t = np.linspace(0, len(data) / sample_rate, len(data))

# Multiply by the carrier and apply a low-pass filter to extract the original signal
demodulated_signal = data * np.cos(wc * t)

# Step 3: Normalize the demodulated signal
demodulated_signal = demodulated_signal / np.max(np.abs(demodulated_signal))

# Step 4: Convert to 16-bit integers
demodulated_signal = np.int16(demodulated_signal * 32767)

# Step 5: Play the demodulated signal
sd.play(demodulated_signal, sample_rate)
sd.wait()  # Wait until the playback is finished
