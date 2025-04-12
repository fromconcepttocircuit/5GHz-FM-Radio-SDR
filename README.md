# DIY FM Radio Station at 5.8 GHz with ADALM Pluto SDR

This repository provides a complete Python-based implementation of an FM transmitter and receiver operating at 5.8 GHz using the ADALM Pluto software-defined radio (SDR). You can use this code to broadcast audio wirelessly and then receive and playback your transmission in real-time.


## 📡 See It In Action (Video Demo)

🎬 **Watch the full video demo and tutorial on YouTube**:  
👉 https://youtu.be/TpBiCjSYmqY


## Features

- Full implementation of FM modulation and demodulation.
- Detailed explanation of FM theory and math.
- GUI interface with PyQtGraph for spectrum visualization and frequency control.
- Real-time audio transmission and reception using ADALM Pluto SDR.
- Complete audio processing, including filtering and resampling.

## Requirements

### Hardware

- 2x ADALM Pluto SDRs (one for TX, one for RX).
- Antennas suitable for 5.8 GHz operation.

### Software Dependencies

- Python 3.x
- libxcb-cursor-dev (Linux users)
- adi (PyADI-IIO)
- numpy
- scipy
- pyqtgraph
- PyQt5
- librosa
- sounddevice
- soundfile
- pyadi-iio

### Installation Instructions

1. **Clone the Repository**
```bash
git clone https://github.com/your-username/5GHz-FM-Radio-SDR.git
cd 5GHz-FM-Radio-SDR
```

2. **Set up Python Environment** (recommended)
```bash
python -m venv env
source env/bin/activate  # On Windows: .\env\Scripts\activate
```

3. **Install System Dependencies** (for Linux)
```bash
sudo apt-get install libxcb-cursor-dev
```

4. **Install Python Dependencies**
```bash
pip install adi pyadi-iio numpy scipy pyqtgraph PyQt5 librosa sounddevice soundfile
```

5. **ADALM Pluto Setup**
- Ensure your Pluto SDR devices are connected and configured correctly.
- Default IP addresses used:
  - RX Pluto: `192.168.2.1`
  - TX Pluto: `192.168.2.2`
- Adjust the code if your Pluto devices have different IPs.

## Running the Application

Start the application by executing:
```bash
python fm_radio_5ghz.py
```

### Usage
- Adjust frequency using the slider or step controls in the GUI.
- Start/stop audio recording using provided buttons.
- Spectrum analyzer displays the audio spectrum in real-time.

## Notes
- Replace the provided audio file (`Night Falls - Everet Almond.mp3`) with your preferred audio track for customized transmissions.
- The code transmits audio at 5.8 GHz—ensure compliance with your local regulations regarding RF transmissions.

## Disclaimer
Use responsibly and always comply with your local regulatory authority regarding radio frequency transmissions.

## License

MIT License. See the LICENSE file for more details.

## 🔗 Connect

- **YouTube Channel**: [From Concept To Circuit](https://www.youtube.com/@fromconcepttocircuit)

### ☕ Support My Work  
If you find this project helpful, consider supporting me:  
[Buy Me a Coffee](https://buymeacoffee.com/concepttoco)
- **GitHub**: [https://github.com/fromconcepttocircuit](https://github.com/fromconcepttocircuit)
- More RF & SDR projects coming soon!

---

Enjoy exploring wireless communication with ADALM Pluto SDR!
