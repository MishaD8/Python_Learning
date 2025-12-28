"""
EC2 Resource Monitor
Checks CPU, Memory, Disk usage and alerts if too high
"""

import psutil
import time
from datetime import datetime

# Thresholds (adjust as needed)

CPU_THRESHOLD = 80    # Alert if CPU > 80 %
MEMORY_THRESHOLD = 85 # Alert if Memory > 85 %
DISK_THRESHOLD = 95   # Alert if Disk > 90 %

def get_size(bytes):
    """Convert bytes to human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes < 1024:
            return f"{bytes:.2f} {unit}"
        bytes /= 1024

def check_resources():
    """Check system resources and alert if thresholds exceeded"""

    print("=" * 70)
    print(f"EC2 Resource Monitor - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    alerts = []

    # CPU Check
    cpu_percent = psutil.cpu_percent(interval=1)
    cpu_status = "HIGH" if cpu_percent > CPU_THRESHOLD else "OK"
    print(f"\n CPU Usage: {cpu_percent}% {cpu_status}")
    if cpu_percent > CPU_THRESHOLD:
        alerts.append(f"CPU usage is high: {cpu_percent}%")

    # Memory Check
    memory = psutil.virtual_memory()
    mem_status = "HIGH" if memory.percent > MEMORY_THRESHOLD else "OK"
    print(f"Memory: {get_size(memory.used)}/{get_size(memory.total)} ({memory.percent}%) {mem_status}")
    if memory.percent > MEMORY_THRESHOLD:
        alerts.append(f"Memory usage is high: {memory.percent}%")

    # Disk Check
    disk = psutil.disk_usage('/')
    disk_status = "HIGH" if disk.percent > DISK_THRESHOLD else "OK"
    print(f"Disk: {get_size(disk.used)}/{get_size(disk.total)} ({disk.percent}%) {disk_status}")
    if disk.percent > DISK_THRESHOLD:
        alerts.append(f"Disk usage is high: {disk.percent}%")

    # Top Processes by CPU
    print(f"\n Top 5 Processes by CPU:")
    processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cpu_percent']):
        try:
            processes.append(proc.info)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    top_cpu = sorted(processes, key=lambda x: x['cpu_percent'], reverse=True)[:5]
    for proc in top_cpu:
         print(f"  . {proc['name']:<20} (PID {proc['pid']:<6}): {proc['cpu_percent']:.1f}%")
 
    # Top Processes by Memory
    print(f"\n Top 5 Processes by Memory:")
    for proc in psutil.process_iter(['pid', 'name', 'memory_percent']):
        try:
            processes.append(proc.info)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    top_mem = sorted(processes, key=lambda x: x.get('memory_percent', 0), reverse=True)[:5]
    for proc in top_mem:
        print(f"  . {proc['name']:<20} (PID {proc.get('pid', 'N/A'):<6}): {proc.get('memory_percent', 0):.1f}%")

    # Network connections
    connections = len(psutil.net_connections())
    print(f"\n Active Network Connections: {connections}")

    # Alerts Summary
    print("\n" + "=" * 70)
    if alerts:
        print("ALERTS:")
        for alert in alerts:
            print(f"  .{alert}")
        print("\n Recommendation: Consider upgrading to t2.small or optimizing processes")
    else:
         print("All systems normal - no resource issues detected")
    print("=" * 70)

if __name__ == "__main__":
    # Install psutil if needed
    try:
       import psutil
    except ImportError:
       print("Installing psutil...")
       import subprocess
       subprocess.run(['pip3', 'install', 'psutil', '--user'], check=True)
       print("Please run the script again")
       exit(0)

    check_resources()
