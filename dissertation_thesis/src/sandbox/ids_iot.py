class IoTIntrusionDetectionSystem:
    def __init__(self):
        # Initialize rules or patterns for known attacks
        self.rules = {
            'MaliciousCommand': ['rm -rf', 'format C:'],
            'UnauthorizedAccess': ['ssh root@', 'admin:admin'],
            # Add more rules as needed
        }

    def detect_intrusion(self, device_id, command):
        """
        Detect potential intrusions based on predefined rules.

        Parameters:
        - device_id: ID of the IoT device sending the command
        - command: Command sent by the IoT device

        Returns:
        - True if intrusion is detected, False otherwise
        """
        for rule, keywords in self.rules.items():
            for keyword in keywords:
                if keyword.lower() in command.lower():
                    print(f"Intrusion Detected on Device {device_id}: {rule}")
                    return True
        return False

# Example of using the IoT IDS
if __name__ == "__main__":
    # Instantiate the IoT IDS
    iot_ids = IoTIntrusionDetectionSystem()

    # Simulate IoT devices sending commands
    devices_and_commands = {
        'Device1': 'ls -la',
        'Device2': 'ssh root@malicious-site.com',
        'Device3': 'admin:admin',
    }

    # Check for intrusions
    for device, command in devices_and_commands.items():
        intrusion_detected = iot_ids.detect_intrusion(device, command)

        if not intrusion_detected:
            print(f"No intrusion detected on Device {device}")
