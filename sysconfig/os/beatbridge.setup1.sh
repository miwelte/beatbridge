#!/bin/bash
# BEATBRIDGE

# Create additional system user and directories
sudo adduser --system --group --home /opt/beatbridge --shell /usr/sbin/nologin beatbridge
sudo mkdir -p /opt/beatbridge 
sudo mkdir -p /etc/beatbridge
sudo mkdir -p /opt/pyenv

sudo chown -R beatbridge:beatbridge /opt/beatbridge

sudo groupadd -f pyenv
sudo usermod -aG pyenv beatbridge
sudo chgrp -R pyenv /opt/pyenv
sudo chmod -R 2775 /opt/pyenv



sudo mkdir -p /opt/beatbridge/.log

sudo chgrp beatbridge /opt/beatbridge/.log
sudo usermod -aG beatbridge sysop






# Enable/Start the service
sudo systemctl daemon-reload
sudo systemctl enable beatbridge.service
sudo systemctl start beatbridge.service
sudo journalctl -u beatbridge.service -f



sudo hostnamectl set-hostname beatbridge-dev01


# Soudcard setup (/boot/firmware/config.txt)
sudo grep -nE "dtparam=audio|dtoverlay=hifiberry|dtoverlay=i2s" /boot/firmware/config.txt
sudo nano /boot/firmware/config.txt
# -> dtparam=audio=off
# -> dtoverlay=hifiberry-dacplusadc
sudo reboot

# Soundcard recognition check
aplay -l
arecord -l
cat /proc/asound/cards

# Set Hifyberry as default soundcard
sudo tee /etc/asound.conf >/dev/null <<'EOF'
defaults.pcm.card 2
defaults.ctl.card 2
EOF

# Ckeck default soundcard setting
sudo nano /etc/asound.conf

# Soundcard input raw recording check
alsamixer

# └─ Card select (F6) → HiFiBerry
# └─ Rec-Input select
# └─ Set Gain/Level not to cli
# └─ Record audio from the card

# Terminal level display
sudo apt install sox

# Check soundcard input level
rec -c 2 -r 48000 -n stat

# Record check of audio from the card
arecord -D plughw:2,0 -f S16_LE -c 2 -r 48000 -d 10 /opt/beatbridge/.log/audio-input-rec-test.wav




