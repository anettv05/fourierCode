import numpy as np
import soundfile as sf
import sounddevice as sd
import matplotlib.pyplot as plt

# Function for adaptive delta modulation
def adaptive_delta_modulate(audio, initial_step_size, step_size_increase, step_size_decrease):
    delta_signal = np.zeros_like(audio)
    modulated_signal = np.zeros_like(audio)
    step_size = initial_step_size
    prediction = 0
    
    for i in range(1, len(audio)):
        error = audio[i] - prediction
        
        # Adjust step size based on the previous bit
        if delta_signal[i-1] == delta_signal[i-2]:  # If the last two bits are the same, increase step size
            step_size *= step_size_increase
        else:
            step_size *= step_size_decrease
        
        # Perform adaptive delta modulation
        if error >= 0:
            delta_signal[i] = 1
            prediction += step_size
        else:
            delta_signal[i] = -1
            prediction -= step_size
        
        modulated_signal[i] = prediction  # Store the modulated signal
    
    return delta_signal, modulated_signal

# Function for adaptive delta demodulation
def adaptive_delta_demodulate(delta_signal, initial_step_size, step_size_increase, step_size_decrease):
    demodulated_signal = np.zeros_like(delta_signal)
    step_size = initial_step_size
    prediction = 0
    
    for i in range(1, len(delta_signal)):
        # Adjust step size based on the previous bit
        if delta_signal[i-1] == delta_signal[i-2]:  # If the last two bits are the same, increase step size
            step_size *= step_size_increase
        else:
            step_size *= step_size_decrease

        # Demodulate based on the delta signal
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
    time = np.linspace(0, len(audio) / sample_rate, num=len(audio))

    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.plot(time, audio, label="Original Audio", color='b')
    plt.title("Original Audio Signal")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")

    plt.subplot(3, 1, 2)
    plt.plot(time, modulated_signal, label="Adaptive Delta Modulated Signal", color='r')
    plt.title("Adaptive Delta Modulated Signal")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")

    plt.subplot(3, 1, 3)
    plt.plot(time, demodulated_signal, label="Demodulated Signal", color='g')
    plt.title("Demodulated Signal")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")

    plt.tight_layout()
    plt.show()

# Load the audio file
filename = '/Users/anettvarghese/Downloads/Test rec.wav'  # Replace with the path to your audio file
audio, sample_rate = sf.read(filename)

# If stereo, convert to mono by averaging the two channels
if len(audio.shape) == 2:
    audio = np.mean(audio, axis=1)

# Adaptive Delta Modulation parameters
initial_step_size = 0.01  # Initial step size
step_size_increase = 1.1  # Factor to increase step size
step_size_decrease = 0.9  # Factor to decrease step size

# Perform adaptive delta modulation and demodulation
delta_signal, modulated_signal = adaptive_delta_modulate(audio, initial_step_size, step_size_increase, step_size_decrease)
demodulated_signal = adaptive_delta_demodulate(delta_signal, initial_step_size, step_size_increase, step_size_decrease)

# Normalize the demodulated signal
demodulated_signal = demodulated_signal / np.max(np.abs(demodulated_signal))

# Play the demodulated signal
print("Playing the demodulated audio...")
play_audio(demodulated_signal, sample_rate)

# Save the demodulated signal as an output audio file
sf.write('demodulated_adm_audio.wav', demodulated_signal, sample_rate)

# Plot the original, modulated, and demodulated signals
plot_signals(audio, modulated_signal, demodulated_signal, sample_rate)

print("Adaptive delta modulation and demodulation complete!")
