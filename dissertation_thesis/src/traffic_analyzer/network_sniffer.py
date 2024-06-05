from scapy.all import sniff

# Define a function to process packets
def process_packet(packet):
    if packet.haslayer('IP'):
        # Extract destination IP and packet length
        dst_ip = packet['IP'].dst
        length = len(packet)

        # Return destination IP and packet length
        return dst_ip, length

# Sniff packets and process them
def capture_packets():
    # Initialize a dictionary to store destination bytes
    dst_bytes = {}

    # Define the packet processing function
    def process_packet(packet):
        if packet.haslayer('IP'):
            dst_ip = packet['IP'].dst
            length = len(packet)
            # Update the dictionary with destination IP and packet length
            dst_bytes[dst_ip] = dst_bytes.get(dst_ip, 0) + length

    # Sniff packets with a filter for IPv4 traffic
    sniff(filter="ip", prn=process_packet)

    return dst_bytes

# Capture packets
dst_bytes = capture_packets()

# Print the result
for dst_ip, bytes_sent in dst_bytes.items():
    print(f"Destination IP: {dst_ip}, Total Bytes Sent: {bytes_sent}")
