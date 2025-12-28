#!/usr/bin/env python3
"""
Simple AWS EC2 Instance Monitor
Shows your instance status in a nice format
"""

import subprocess
import json
from datetime import datetime


def get_instance_status(instance_id):
    """Get EC2 instance information"""
    cmd = [
        'aws', 'ec2', 'describe-instances',
        '--instance-ids', instance_id,
        '--region', 'us-east-1',
        '--output', 'json'
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    data = json.loads(result.stdout)

    instance = data['Reservations'][0]['Instances'][0]

    return {
        'Instance ID': instance['InstanceId'],
        'State': instance['State']['Name'],
        'Type': instance['InstanceType'],
        'Public IP': instance.get('PublicIpAddress', 'N/A'),
        'Launch Time': instance['LaunchTime'],
        'OS': instance.get('PlatformDetails', 'Linux/UNIX')
    }


def main():
    instance_id = 'i-098cf64b90aa4a99d'

    print("=" * 50)
    print("AWS EC2 Instance Monitor")
    print(f"Checked at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)

    status = get_instance_status(instance_id)

    for key, value in status.items():
        print(f"{key:15}: {value}")

    print("=" * 50)


if __name__ == "__main__":
    main()
