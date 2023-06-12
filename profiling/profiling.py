import subprocess
import psutil
import time
import csv
import nvidia_smi

# Start your program
process = subprocess.Popen(["./build/src/gpu/gpu", "-i", "dataset/tokyo.mp4", "-o", "dataset/tokyo_output.mp4"])

# Open the output files
with open('ram_usage.csv', 'w', newline='') as ram_file, open('gpu_usage.csv', 'w', newline='') as gpu_file, open('vram_usage.csv', 'w', newline='') as vram_file:
    ram_writer = csv.writer(ram_file)
    gpu_writer = csv.writer(gpu_file)
    vram_writer = csv.writer(vram_file)

    # Write headers
    ram_writer.writerow(['Time', 'RAM_Usage'])
    gpu_writer.writerow(['Time', 'GPU_Usage'])
    vram_writer.writerow(['Time', 'VRAM_Usage'])

    # Initialize NVML for GPU monitoring
    nvidia_smi.nvmlInit()
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)

    # While the process is running
    while process.poll() is None:
        # Get the process info using psutil for RAM
        proc = psutil.Process(process.pid)
        mem_info = proc.memory_info()
        ram_usage = mem_info.rss / (1024 ** 2)  # Convert to MB
        ram_writer.writerow([time.time(), ram_usage])

        # Get GPU info using nvidia_smi
        gpu_info = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
        gpu_usage = gpu_info.gpu  # This is a percentage
        gpu_writer.writerow([time.time(), gpu_usage])

        # Get VRAM info
        vram_info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        vram_usage = vram_info.used / (1024 ** 2)  # Convert to MB
        vram_writer.writerow([time.time(), vram_usage])

        # Sleep for a bit to prevent overwhelming the system
        time.sleep(1)

