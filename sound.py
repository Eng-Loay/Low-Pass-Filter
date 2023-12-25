import pyaudio
import wave
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import speech_recognition as sr

def low_pass_filter(signal_data, cutoff_frequency, sampling_rate, order=5):
    nyquist_frequency = 0.5 * sampling_rate
    normalized_cutoff = cutoff_frequency / nyquist_frequency
    b, a = signal.butter(order, normalized_cutoff, btype='low', analog=False)
    filtered_signal = signal.filtfilt(b, a, signal_data)
    return filtered_signal.astype(np.int32)

def plot_waveform(data, title):
    plt.figure(figsize=(10, 4))
    plt.plot(data, color='b')
    plt.title(title)
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.show()

frames_per_buffer = 3200  # Record in a buffer of 3200 samples
format = pyaudio.paInt32  # 32-bit per sample
channels = 1
sample_rate = 16000

# Initialize PyAudio object
p = pyaudio.PyAudio()

# Open stream for recording
try:
    stream = p.open(
        format=format,
        channels=channels,
        rate=sample_rate,
        input=True,  # To capture audio
        frames_per_buffer=frames_per_buffer
    )
except IOError as e:
    print(f"Error opening stream: {e}")
    p.terminate()
    raise SystemExit

print("Start Recording")

seconds = 5  # Record duration
frames = []  # Initialize array to store frames

# Record audio for the specified duration
for i in range(0, int(sample_rate / frames_per_buffer * seconds)):
    try:
        data = stream.read(frames_per_buffer)  # Read 3200 frames at each iteration
        frames.append(data)
    except IOError as e:
        print(f"Error recording: {e}")
        break

print("Finished Recording")

# Save the original audio to a WAV file
wf_original = wave.open("output_original.wav", "wb")
wf_original.setnchannels(channels)
wf_original.setsampwidth(p.get_sample_size(format))
wf_original.setframerate(sample_rate)
wf_original.writeframes(b''.join(frames))
wf_original.close()

# Display waveform of the original audio
original_data = np.frombuffer(b''.join(frames), dtype=np.int32)
plot_waveform(original_data, "Original Audio Waveform")

# Apply low-pass filter
cutoff_frequency = 600  # Adjust cutoff frequency as needed
filtered_frames = [low_pass_filter(np.frombuffer(frame, dtype=np.int32), cutoff_frequency, sample_rate) for frame in frames]

# Check if the filtered audio frames are empty (filtering might have failed)
if not filtered_frames:
    print("Filtering failed. Check cutoff frequency or input signal.")
    p.terminate()
    raise SystemExit

# Save the filtered audio to a WAV file
wf_filtered = wave.open("output_filtered.wav", "wb")
wf_filtered.setnchannels(channels)
wf_filtered.setsampwidth(p.get_sample_size(format))
wf_filtered.setframerate(sample_rate)
wf_filtered.writeframes(b''.join(filtered_frames))
wf_filtered.close()

# Display waveform of the filtered audio
filtered_data = np.frombuffer(b''.join(filtered_frames), dtype=np.int32)
plot_waveform(filtered_data, "Filtered Audio Waveform")

# Speech recognition on the original audio
audio_file_original = sr.AudioFile("output_original.wav")
recognizer_original = sr.Recognizer()

with audio_file_original as source_original:
    audio_file_data_original = recognizer_original.record(source_original)

text_original = recognizer_original.recognize_google(audio_data=audio_file_data_original, language="en-US")
print("Original Text:", text_original)

# Speech recognition on the filtered audio
audio_file_filtered = sr.AudioFile("output_filtered.wav")
recognizer_filtered = sr.Recognizer()

with audio_file_filtered as source_filtered:
    audio_file_data_filtered = recognizer_filtered.record(source_filtered)

text_filtered = recognizer_filtered.recognize_google(audio_data=audio_file_data_filtered, language="en-US")
print("Filtered Text:", text_filtered)

# Close the PyAudio stream
stream.stop_stream()
stream.close()
p.terminate()