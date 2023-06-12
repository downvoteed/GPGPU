import pandas as pd
import matplotlib.pyplot as plt

# Load the data
gpu_data = pd.read_csv('gpu_usage.csv')
ram_data = pd.read_csv('ram_usage.csv')
vram_data = pd.read_csv('vram_usage.csv')

# Create subplots
plt.figure(figsize=(10,12))

# First subplot for RAM and VRAM usage
plt.subplot(2, 1, 1)
plt.plot(ram_data['Time'], ram_data['RAM_Usage'], label='RAM Usage (MiB)')
plt.plot(vram_data['Time'], vram_data['VRAM_Usage'], label='VRAM Usage (MiB)')
plt.title('RAM and VRAM Usage Over Time')
plt.xlabel('Time')
plt.ylabel('Usage (MiB)')
plt.legend()
plt.grid(True)

# Second subplot for GPU usage
plt.subplot(2, 1, 2)
plt.plot(gpu_data['Time'], gpu_data['GPU_Usage'], label='GPU Usage (%)')
plt.title('GPU Usage Over Time')
plt.xlabel('Time')
plt.ylabel('Usage (%)')
plt.legend()
plt.grid(True)

# Adjust layout and display
plt.tight_layout()
plt.show()

