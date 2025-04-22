import pandas as pd
import socket
import json
import time

# Load preprocessed normalized dataset (without labels if possible)
df = pd.read_csv('datasets/test_dataset.csv')  # Use your normalized test CSV path

# Remove label if present
features_only = df.drop(columns=['label'], errors='ignore')

# Set Raspberry Pi IP and port
target_ip = "192.168.1.52"  # Replace with your Raspberry Pi's IP
target_port = 5050

# Set up UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Send rows one-by-one to simulate real-time feed
for _, row in features_only.iterrows():
    payload = json.dumps(row.to_dict())
    sock.sendto(payload.encode(), (target_ip, target_port))
    print(f"Sent: {payload}")
    time.sleep(0.1)  # Simulate ~10Hz sensor frequency