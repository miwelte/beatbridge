#!/bin/bash
# BEATBRIDGE


# Python environment setup
sudo apt update



cd /opt/beatbridge
source .venv/bin/activate



# Install PortAudio (and Headers)
sudo apt update
sudo apt install -y libportaudio2 portaudio19-dev

source .venv/bin/activate
python -c "import sounddevice as sd; print(sd.query_devices())"
