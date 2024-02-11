from scapy.all import sniff

def packet_callback(packet):
    # Print the details of each packet
    print(packet.summary())

# Use the sniff function to capture packets
# The prn parameter specifies the callback function to be called for each captured packet
# The store parameter is set to 0 to disable storing packets in memory
# You can adjust the filter parameter to capture specific types of packets (e.g., "tcp", "udp")
sniff(prn=packet_callback, store=0)
