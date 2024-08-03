'''
import numpy as np
import matplotlib.pyplot as plt

# Time vector
t = np.linspace(0, 2*np.pi, 1000)

# Square wave function
def square_wave(t):
    return 0.5 * (1 + np.sign(np.sin(t)))

# Plot square wave
plt.figure(figsize=(10, 6))
plt.subplot(3, 1, 1)
plt.plot(t, square_wave(t))
plt.title('Square Wave $u(t)$')
plt.xlabel('Time')
plt.ylabel('Amplitude')

# Sin wave function
def sin_wave(t):
    return np.sin(t)

# Plot sin wave
plt.subplot(3, 1, 2)
plt.plot(t, sin_wave(t))
plt.title('Sin Wave $sin(t)$')
plt.xlabel('Time')
plt.ylabel('Amplitude')

# 4/pi * sin wave function
def scaled_sin_wave(t):
    return (4/np.pi) * np.sin(t)

# Plot scaled sin wave
plt.subplot(3, 1, 3)
plt.plot(t, scaled_sin_wave(t))
plt.title('$\\frac{4}{\pi} sin(t)$')
plt.xlabel('Time')
plt.ylabel('Amplitude')

plt.tight_layout()
plt.show()
'''
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft

# Function to generate a square wave signal
def square_wave(t, period):
    return 0.5 * (1 + np.sign(np.sin(2 * np.pi * t / period)))

# Time vector
t = np.linspace(0, 10, 1000)  # Define time from 0 to 10 seconds
period = 2  # Period of the square wave

# Generate the square wave signal
signal = square_wave(t, period)
plt.figure(figsize=(10, 6))
plt.subplot(2,1,1)
plt.plot(t, signal)
plt.title(' Square Wave')
plt.xlabel('Time')
plt.ylabel('Magnitude')
# Compute the Fourier transform
fourier_transform = fft(signal)

# Compute the frequencies corresponding to the Fourier coefficients
freq = np.fft.fftfreq(len(t), t[1] - t[0])

# Plot the magnitude of the Fourier coefficients
plt.figure(figsize=(10, 6))
plt.subplot(2,1,2)
plt.plot(freq, np.abs(fourier_transform))
plt.title('Fourier Spectral Series of Square Wave')
plt.xlabel('Frequency')
plt.ylabel('Magnitude')

plt.tight_layout()
plt.show()
'''
'''
import numpy as np
import matplotlib.pyplot as plt
def ustep(t):
    """
    Unit step function.
    
    Args:
    t (float or numpy.ndarray): Time variable.
    
    Returns:
    float or numpy.ndarray: Value of the unit step function at t.
    """
    # Ensure t is real-valued
    t = np.real(t)
    
    # Initialize y with zeros
    y = np.zeros_like(t)
    
    # Set y to 1 where t >= 0
    y[t >= 0] = 1
    
    return y
t = np.linspace(0, 100, 1000)
plt.figure(figsize=(10, 6))
plt.plot(t,ustep(t))
plt.title('Unit Step')
plt.xlabel('Time')
plt.ylabel('Magnitude')
plt.grid(True)
plt.show()

import numpy as np
import matplotlib.pyplot as plt
def rect(t):
    """
    Rectangular function.
    
    Args:
    t (float or numpy.ndarray): Time variable.
    
    Returns:
    float or numpy.ndarray: Value of the rectangular function at t.
    """
    # Ensure t is real-valued
    t = np.real(t)
    
    # Compute the rectangular function
    y = (np.sign(t + 0.5) - np.sign(t - 0.5) > 0).astype(float)
    
    return y
t=np.linspace(-10,10,1000)
plt.figure(figsize=(10,6))
plt.plot(t,rect(t))
plt.title('Rectangle function')
plt.xlabel('time')
plt.ylabel('amplitude')
plt.grid(True)
plt.show()

import numpy as np
import matplotlib.pyplot as plt

def triangular(t, width=1):
    """
    Triangular function.
    
    Args:
    t (float or numpy.ndarray): Time variable.
    width (float): Width of the triangular function.
    
    Returns:
    float or numpy.ndarray: Value of the triangular function at t.
    """
    # Ensure t is real-valued
    t = np.real(t)
    
    # Compute the triangular function
    y = np.maximum(0, 1 - np.abs(t) / width)
    
    return y

# Generate a time array
t = np.linspace(-2, 2, 1000)

# Compute the triangular function values
y = triangular(t)

# Plot the triangular function
plt.plot(t, y)
plt.title('Triangular Function')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()

import matplotlib.pyplot as plt
import numpy as np

freq = np.arange(-2*np.pi, -1*np.pi, 0.01)
amp = 2 * np.exp((-4 * np.pi * freq) + (0.5 * np.pi)) * (np.sinc(4 * np.pi * freq))
time = np.arange(-5, 10, 0.1)
x = np.fft.ifft(amp)

plt.plot(time, np.real(x))  # Take the real part of the inverse Fourier transform
plt.title("Inverse Fourier Transform")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()
'''
'''
import matplotlib.pyplot as plt
import numpy as np
from scipy import fft
s=0.01
t = np.arange(-5,5,s)
amp=0.5*(1+np.sign(t+0.5))-0.5*(1+np.sign(t-0.5))
ampf=fft.fft(amp)
freq = fft.fftfreq(len(t),s)
plt.figure(figsize=(10,6))
plt.subplot(4,1,1)
plt.plot(t,amp)
plt.title("unit fn")
plt.xlabel("time")
plt.ylabel("value")
plt.grid(True)

plt.subplot(4,1,2)
plt.plot(freq,np.abs(ampf)*s)
plt.xlabel("freq")
plt.ylabel("value")
plt.title("fourier")
plt.grid(True)

plt.subplot(4,1,3)
plt.plot(freq,np.angle(ampf))
plt.xlabel("freq")
plt.ylabel("value")
plt.title("fourier")
plt.grid(True)

x=np.sinc(freq)
plt.subplot(4,1,4)
plt.plot(freq,x)
plt.xlabel('time')
plt.ylabel('sinc')
plt.grid(True)
plt.tight_layout()
plt.show()
'''
'''
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft

# Sample data
samples = 1000
t = np.arange(0, 1, 1/samples)
f = 5  # frequency
signal = np.sin(2 * np.pi * f * t)
# Compute FFT
fs = fft(signal)
fi = np.fft.fftfreq(samples, 1/samples)

# Plot the FFT result
plt.plot(fi, np.abs(fs) * 2 / samples, color="#fdb462")
plt.title("FFT of Sinusoidal Signal")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()

'''
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy import fft
t = np.arange(-5,5,0.1)
m = np.sin(2*np.pi*t)
c = np.sin(4*np.pi*t)
mf=fft.fft(m)
cf=fft.fft(c)
f=np.fft.fftfreq(len(t),0.1)
muf = mf * [0.5 + 0.5*np.sign(f)]
mlf = mf * [0.5 - 0.5*np.sign(f)]

plt.figure(figsize=(10,8))
plt.subplot(3,1,1)
plt.plot(f,np.abs(mf)*0.02)
plt.title("Fourier of message")
plt.xlabel("freq")
plt.ylabel("magnitude")

plt.subplot(3,1,2)
plt.plot(f,np.abs(muf.flatten())*0.02)
plt.title("upper band Fourier of message")
plt.xlabel("freq")
plt.ylabel("magnitude")

plt.subplot(3,1,3)
plt.plot(f,np.abs(mlf.flatten())*0.02)
plt.title("lower band Fourier of message")
plt.xlabel("freq")
plt.ylabel("magnitude")
plt.tight_layout()
plt.show()
'''
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

# Parameters
T = 1.0  # Time period
Fs = 100.0  # Sampling frequency
t = np.arange(0, T, 1/Fs)  # Time vector
f = 5.0  # Frequency of the sine wave

# Generate sine wave
sin_wave = np.sin(2 * np.pi * f * t)

# Compute DFT
dft = fft(sin_wave)

# Compute frequencies
N = len(sin_wave)
frequencies = fftfreq(N, 1/Fs)[:N//2]

# Plot the magnitude spectrum
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.stem(frequencies, np.abs(dft)[:N//2])
plt.title('Magnitude Spectrum of Sine Wave')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.grid(True)

# Plot the phase spectrum
plt.subplot(1, 2, 2)
plt.stem(frequencies, np.angle(dft)[:N//2])
plt.title('Phase Spectrum of Sine Wave')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Phase (radians)')
plt.grid(True)

plt.tight_layout()
plt.show()
'''