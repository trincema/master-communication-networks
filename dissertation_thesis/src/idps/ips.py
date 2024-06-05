import platform
# AC: 0, F-T: 0, W-X: 0, Y: 1, Z-AA: 0, AD: 0, AE: 1, AF: 0.
# Dos: src_bytes, dst_bytes, count, srv_count, dst_host_count, dst_host_srv_count

class IntrusionPreventionSystem(object):
    def block_ip_address(self, target):
        # Define the domain or IP address you want to block
        target = "www.google.com"  # Change this to the domain or IP you want to block

        # Determine the path to the host file based on the operating system
        system = platform.system()
        if system == "Windows":
            hosts_path = r"C:\Windows\System32\drivers\etc\hosts"
        else:
            hosts_path = "/etc/hosts"

        # Open the host file in append mode and add an entry to block the target
        # mapping unwanted domain names to the loopback address (127.0.0.1),
        # users can block access to specific websites or redirect them to alternative addresses
        with open(hosts_path, "a") as hosts_file:
            hosts_file.write(f"\n127.0.0.1    {target}\n")

        print(f"Blocked {target} in the host file.")

        # Note: On Linux and macOS, you may need to run this script with sudo for write access to /etc/hosts.