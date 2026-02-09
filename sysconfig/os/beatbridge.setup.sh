
# Create system user and directories
sudo useradd --system --no-create-home --shell /usr/sbin/nologin bpm
sudo mkdir -p /opt/beatbridge /etc/beatbridge
sudo chown -R bpm:bpm /opt/beatbridge

# Enable/Start the service
sudo systemctl daemon-reload
sudo systemctl enable beatbridge.service
sudo systemctl start beatbridge.service
sudo journalctl -u beatbridge.service -f





sudo hostnamectl set-hostname beatbridge-dev01
