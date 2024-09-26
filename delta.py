import numpy as np
import soundfile as sf
import sounddevice as sd
import matplotlib.pyplot as plt

# Function to perform delta modulation
def delta_modulate(audio, step_size):
    delta_signal = np.zeros_like(audio)
    modulated_signal = np.zeros_like(audio)
    
    # Initial prediction value (quantized value)
    prediction = 0
    
    for i in range(1, len(audio)):
        # Calculate the difference between the current sample and the prediction
        error = audio[i] - prediction
        
        # Delta modulation (if error is positive, increase prediction, otherwise decrease)
        if error >= 0:
            delta_signal[i] = 1
            prediction += step_size  # Increase by step size
        else:
            delta_signal[i] = -1
            prediction -= step_size  # Decrease by step size
        
        modulated_signal[i] = prediction  # Store the modulated signal
    
    return delta_signal, modulated_signal

# Function to perform delta demodulation
def delta_demodulate(delta_signal, step_size):
    demodulated_signal = np.zeros_like(delta_signal)
    prediction = 0
    
    for i in range(1, len(delta_signal)):
        # Demodulate the signal by accumulating the changes
        if delta_signal[i] > 0:
            prediction += step_size
        else:
            prediction -= step_size
            
        demodulated_signal[i] = prediction
    
    return demodulated_signal

# Function to play audio using sounddevice
def play_audio(audio, sample_rate):
    sd.play(audio, samplerate=sample_rate)
    sd.wait()

# Function to plot the original, modulated, and demodulated signals
def plot_signals(audio, modulated_signal, demodulated_signal, sample_rate):
    # Create time axis
    time = np.linspace(0, len(audio) / sample_rate, num=len(audio))

    # Plot the original, modulated, and demodulated signals
    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.plot(time, audio, label="Original Audio", color='b')
    plt.title("Original Audio Signal")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")

    plt.subplot(3, 1, 2)
    plt.plot(time, modulated_signal, label="Delta Modulated Signal", color='r')
    plt.title("Delta Modulated Signal")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")

    plt.subplot(3, 1, 3)
    plt.plot(time, demodulated_signal, label="Demodulated Signal", color='g')
    plt.title("Demodulated Signal")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")

    plt.tight_layout()
    plt.show()

# Load the audio file (mono or stereo)
filename = '/Users/anettvarghese/Downloads/Test rec.wav'  # Replace with the path to your audio file
audio, sample_rate = sf.read(filename)

# If stereo, convert to mono by averaging the two channels
if len(audio.shape) == 2:
    audio = np.mean(audio, axis=1)

# Delta modulation parameters
step_size = 0.01  # Set step size for delta modulation

# Perform delta modulation and demodulation
delta_signal, modulated_signal = delta_modulate(audio, step_size)
demodulated_signal = delta_demodulate(delta_signal, step_size)

# Normalize the demodulated signal to ensure it lies between -1 and 1
demodulated_signal = demodulated_signal / np.max(np.abs(demodulated_signal))

# Play the demodulated signal
print("Playing the demodulated audio...")
play_audio(demodulated_signal, sample_rate)

# Save the demodulated signal as an output audio file
sf.write('demodulated_audio.wav', demodulated_signal, sample_rate)

# Plot the original, modulated, and demodulated signals
plot_signals(audio, modulated_signal, demodulated_signal, sample_rate)

print("Delta modulation and demodulation complete!")
