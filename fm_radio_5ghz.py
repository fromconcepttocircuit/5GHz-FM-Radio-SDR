import sys
import threading
import time
import numpy as np
import adi
import sounddevice as sd
from scipy.signal import butter, lfilter, firwin, resample_poly
from fractions import Fraction
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore
import librosa
import soundfile as sf
import queue

########################################
# Define Sample Rates for RX and TX
########################################
sdr_rx_sample_rate = int(1e6)  # e.g., 1 MHz for RX
sdr_tx_sample_rate = int(1e6)  # e.g., 1 MHz for TX
audio_sample_rate = 48000      # Audio output sample rate

########################################
# SDR Configuration - RX (IP 192.168.2.1)
########################################
sdr = adi.ad9361(uri="ip:192.168.2.1")
sdr.rx_enabled_channels = [0]
sdr.rx_lo = int(5.8e9)
sdr.sample_rate = sdr_rx_sample_rate
sdr.rx_rf_bandwidth = int(1e6)
sdr.rx_buffer_size = 2**15
sdr.gain_control_mode = "manual"
sdr.rx_gain = 0

########################################
# SDR Configuration - TX (IP 192.168.2.2)
########################################
tx = adi.ad9361(uri="ip:192.168.2.2")
tx.tx_enabled_channels = [0]
tx.tx_lo = int(5.8e9)
tx.sample_rate = sdr_tx_sample_rate
tx.tx_rf_bandwidth = int(1e6)
tx.tx_hardwaregain_chan0 = 0
tx.tx_cyclic_buffer = False
try:
    tx.tx_buffer_size = 2**20
except Exception as e:
    print("Could not set tx_buffer_size:", e)

########################################
# Audio and Filter Setup
########################################
# Compute exact resampling factors from sdr_rx_sample_rate to audio_sample_rate.
r = Fraction(audio_sample_rate, sdr_rx_sample_rate).limit_denominator()
up_factor = r.numerator
down_factor = r.denominator
print(f"Resampling factors: up={up_factor}, down={down_factor}")

# Improved FIR anti-aliasing filter for IQ processing:
fir_taps = firwin(numtaps=201, cutoff=75e3, fs=sdr_rx_sample_rate, window=('kaiser', 8.6))

# De-emphasis filter (75 µs).
tau = 75e-6
alpha = np.exp(-1 / (audio_sample_rate * tau))
b_de = [1 - alpha]
a_de = [1, -alpha]

# Audio low-pass filter with cutoff=20 kHz to preserve the 19 kHz pilot.
b_audio, a_audio = butter(20, 20000, fs=audio_sample_rate, btype='low')

########################################
# DC Reject Filter Function
########################################
def dc_block_filter(data, cutoff=2e3, fs=audio_sample_rate):
    b, a = butter(1, cutoff / (fs / 2), btype='highpass')
    return lfilter(b, a, data)

########################################
# Global Buffers and Locks
########################################
audio_buffer = np.array([], dtype=np.float32)  # Processed audio from RX processing thread.
buffer_lock = threading.Lock()

playback_buffer = np.array([], dtype=np.float32)  # Data fed to audio playback.
playback_lock = threading.Lock()

latest_audio_block = None  # For spectrum plotting.
plot_lock = threading.Lock()

# TX Queue for double-threading TX.
tx_queue = queue.Queue(maxsize=10)

########################################
# Pilot Energy Functions
########################################
def compute_band_energy(chunk, fs, center, bandwidth):
    N = len(chunk)
    fft_data = np.fft.rfft(chunk)
    freqs = np.fft.rfftfreq(N, 1 / fs)
    band_idx = np.where((freqs >= center - bandwidth / 2) & (freqs <= center + bandwidth / 2))[0]
    energy = np.sum(np.abs(fft_data[band_idx]) ** 2)
    return energy

def pilot_ratio(chunk, fs, pilot_center=19000, control_center=10000, bandwidth=1000):
    energy_pilot = compute_band_energy(chunk, fs, pilot_center, bandwidth)
    energy_control = compute_band_energy(chunk, fs, control_center, bandwidth)
    ratio = energy_pilot / (energy_control + 1e-12)
    return ratio

########################################
# FM Demodulation Function
########################################
def fm_demodulate(iq_samples):
    epsilon = 1e-12
    iq_norm = iq_samples / (np.abs(iq_samples) + epsilon)
    phase = np.angle(iq_norm)
    dphase = np.diff(phase)
    return np.unwrap(dphase)

########################################
# SDR RX Thread: Read IQ, process, and append to audio_buffer
########################################
def sdr_rx_thread():
    global audio_buffer, latest_audio_block
    while True:
        iq_samples = sdr.rx()
        if isinstance(iq_samples, (list, tuple)):
            iq_samples = iq_samples[0]
        iq_samples = np.array(iq_samples, dtype=np.complex64).flatten()
        demodulated = fm_demodulate(iq_samples)
        filtered = lfilter(fir_taps, 1.0, demodulated)
        # Resample using up_factor/down_factor for exact conversion.
        audio_data = resample_poly(filtered, up=up_factor, down=down_factor)
        # Apply de-emphasis filter
        audio_data = lfilter(b_de, a_de, audio_data)
        # Apply DC blocking filter to remove any residual DC offset.
        audio_data = dc_block_filter(audio_data)
        # Final audio low-pass filter
        audio_data = lfilter(b_audio, a_audio, audio_data).astype(np.float32)
        with plot_lock:
            latest_audio_block = audio_data.copy()
        with buffer_lock:
            audio_buffer = np.concatenate((audio_buffer, audio_data))

########################################
# Rebuffer Thread: Transfer processed audio to playback_buffer
# Discard chunk if pilot ratio is below threshold.
########################################
def rebuffer_thread():
    global audio_buffer, playback_buffer
    chunk_size = 4096
    relative_threshold = 2.0  # Pilot energy must be >= 2x control energy
    while True:
        chunk = None
        with buffer_lock:
            if len(audio_buffer) >= chunk_size:
                chunk = audio_buffer[:chunk_size]
                audio_buffer = audio_buffer[chunk_size:]
        if chunk is not None:
            ratio = pilot_ratio(chunk, audio_sample_rate)
            if ratio >= relative_threshold:
                with playback_lock:
                    playback_buffer = np.concatenate((playback_buffer, chunk))
        else:
            time.sleep(0.005)

########################################
# Audio Playback Thread and Callback
########################################
def audio_playback_thread():
    with sd.OutputStream(channels=1,
                         samplerate=audio_sample_rate,
                         blocksize=2048,
                         callback=audio_callback):
        while True:
            sd.sleep(1000)

def audio_callback(outdata, frames, time_info, status):
    global playback_buffer, recording, recorded_audio
    with playback_lock:
        if len(playback_buffer) >= frames:
            chunk = playback_buffer[:frames]
            playback_buffer = playback_buffer[frames:]
        else:
            chunk = np.zeros(frames, dtype=np.float32)
    outdata[:] = chunk.reshape(-1, 1)
    # Append to recording buffer if recording is enabled.
    if recording:
        recorded_audio.append(chunk.copy())

########################################
# Transmitter: Pre-calculate Entire FM Signal (Audio + Pilot)
########################################
fm_signal_global = None

def prepare_fm_signal():
    global fm_signal_global
    pilot_freq = 19000
    pilot_amplitude = 0.5
    frequency_deviation = 75e3

    audio_file = "Night Falls - Everet Almond.mp3"
    try:
        audio_data, audio_fs = librosa.load(audio_file, sr=None, mono=True)
        print(f"Loaded {audio_file}, fs={audio_fs}, duration={len(audio_data) / audio_fs:.2f}s")
    except Exception as e:
        print("Error loading audio file:", e)
        return

    # Upsample entire audio file to match TX sample rate.
    upsampled_audio = resample_poly(audio_data, up=tx.sample_rate, down=audio_fs)
    total_samples = len(upsampled_audio)
    print(f"Upsampled audio length: {total_samples} samples, duration: {total_samples / tx.sample_rate:.2f}s")

    composite_audio = upsampled_audio
    N = len(composite_audio)
    print(f"Using entire audio: {N} samples ({N / tx.sample_rate:.2f} seconds) for TX signal.")

    t = np.arange(N) / tx.sample_rate
    pilot_tone = pilot_amplitude * np.sin(2 * np.pi * pilot_freq * t)
    composite_signal = composite_audio + pilot_tone
    phase = 2 * np.pi * frequency_deviation * np.cumsum(composite_signal) / tx.sample_rate
    fm_signal = (2 ** 14) * np.exp(1j * phase).astype(np.complex64)

    # Removed crossfade processing for simplicity.
    fm_signal_global = fm_signal
    print("FM modulation complete and FM signal prepared without crossfade.")

########################################
# TX Preparation Thread: Prepare TX chunks and push them into a queue
########################################
def tx_preparation_thread():
    global fm_signal_global
    prepare_fm_signal()
    if fm_signal_global is None:
        print("FM signal preparation failed. Exiting TX preparation thread.")
        return

    try:
        chunk_size = tx.tx_buffer_size
    except AttributeError:
        chunk_size = 32768

    total = len(fm_signal_global)
    pointer = 0
    print("Starting TX preparation thread to fill transmission queue...")
    while True:
        if pointer + chunk_size <= total:
            chunk = fm_signal_global[pointer:pointer + chunk_size]
            pointer += chunk_size
        else:
            remainder = total - pointer
            chunk = np.concatenate((fm_signal_global[pointer:], fm_signal_global[:chunk_size - remainder]))
            pointer = chunk_size - remainder
        tx_queue.put(chunk)  # Blocks if the queue is full

########################################
# TX Transmission Thread: Send chunks from the queue
########################################
def tx_transmission_thread():
    print("Starting TX transmission thread to send chunks from the queue...")
    while True:
        chunk = tx_queue.get()  # Blocks until a chunk is available
        tx.tx(chunk)
        time.sleep(0.001)

########################################
# GUI with Start/Stop Recording
########################################
recording = False
recorded_audio = []

class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FM RX/TX with Pilot-Based RX Post-Processing")
        self.resize(1000, 600)
        layout = QtWidgets.QVBoxLayout()
        self.setLayout(layout)

        # Frequency control (slider in MHz)
        freq_layout = QtWidgets.QHBoxLayout()
        self.minus_button = QtWidgets.QPushButton("-")
        self.minus_button.clicked.connect(self.decrease_frequency)
        freq_layout.addWidget(self.minus_button)
        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.setRange(300, 5000)
        self.slider.setSingleStep(1)
        self.slider.setValue(sdr.rx_lo // int(1e6))
        self.slider.valueChanged.connect(self.update_frequency)
        freq_layout.addWidget(self.slider)
        self.plus_button = QtWidgets.QPushButton("+")
        self.plus_button.clicked.connect(self.increase_frequency)
        freq_layout.addWidget(self.plus_button)
        layout.addLayout(freq_layout)

        self.freq_label = QtWidgets.QLabel(f"LO Frequency: {sdr.rx_lo / 1e6:.3f} MHz")
        layout.addWidget(self.freq_label)

        # Step input (in MHz)
        step_layout = QtWidgets.QHBoxLayout()
        step_label = QtWidgets.QLabel("Step (MHz):")
        step_layout.addWidget(step_label)
        self.step_textbox = QtWidgets.QLineEdit("1")
        step_layout.addWidget(self.step_textbox)
        layout.addLayout(step_layout)

        # Recording buttons
        self.start_recording_button = QtWidgets.QPushButton("Start Recording")
        self.start_recording_button.clicked.connect(self.start_recording)
        layout.addWidget(self.start_recording_button)
        self.stop_recording_button = QtWidgets.QPushButton("Stop Recording")
        self.stop_recording_button.clicked.connect(self.stop_recording)
        layout.addWidget(self.stop_recording_button)

        # Spectrum plot
        self.plot_widget = pg.PlotWidget(title="Audio Spectrum")
        self.curve = self.plot_widget.plot(pen='y')
        self.plot_widget.setLabel('bottom', "Frequency (Hz)")
        self.plot_widget.setLabel('left', "Magnitude (dB)")
        layout.addWidget(self.plot_widget)

        # Timer for spectrum updates
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_spectrum)
        self.timer.start(100)

    def update_frequency(self, value):
        sdr.rx_lo = value * int(1e6)
        self.freq_label.setText(f"LO Frequency: {sdr.rx_lo / 1e6:.3f} MHz")

    def increase_frequency(self):
        try:
            step = int(self.step_textbox.text())
        except ValueError:
            step = 1
        new_val = self.slider.value() + step
        if new_val > self.slider.maximum():
            new_val = self.slider.maximum()
        self.slider.setValue(new_val)

    def decrease_frequency(self):
        try:
            step = int(self.step_textbox.text())
        except ValueError:
            step = 1
        new_val = self.slider.value() - step
        if new_val < self.slider.minimum():
            new_val = self.slider.minimum()
        self.slider.setValue(new_val)

    def update_spectrum(self):
        global latest_audio_block
        with plot_lock:
            data = latest_audio_block if (latest_audio_block is not None and len(latest_audio_block) > 0) else None
        if data is not None:
            N = len(data)
            window = np.hanning(N)
            fft_data = np.fft.rfft(data * window)
            mag = np.abs(fft_data)
            mag_db = 20 * np.log10(mag + 1e-12)
            freqs = np.fft.rfftfreq(N, 1.0 / audio_sample_rate)
            self.curve.setData(freqs, mag_db)

    def start_recording(self):
        global recording, recorded_audio
        recording = True
        recorded_audio = []
        print("Recording started.")

    def stop_recording(self):
        global recording, recorded_audio
        recording = False
        if recorded_audio:
            data = np.concatenate(recorded_audio)
            sf.write("recorded_output.wav", data, audio_sample_rate)
            print("Recording saved to recorded_output.wav.")
        else:
            print("No audio recorded.")

########################################
# Main Function
########################################
def main():
    threading.Thread(target=sdr_rx_thread, daemon=True).start()
    threading.Thread(target=audio_playback_thread, daemon=True).start()
    threading.Thread(target=rebuffer_thread, daemon=True).start()
    threading.Thread(target=tx_preparation_thread, daemon=True).start()
    threading.Thread(target=tx_transmission_thread, daemon=True).start()
    app = QtWidgets.QApplication(sys.argv)
    main_win = MainWindow()
    main_win.show()
    app.exec()

if __name__ == "__main__":
    print("Starting integrated FM RX/TX without crossfade...")
    main()
