import scapy.all as scapy
from datetime import datetime
import time

captured_packets = []
steps=20

def sniff_packets(interface):
    scapy.sniff(iface=interface, store=False, prn=process_packet, filter="tcp port 80 or tcp port 443")

def process_packet(packet):
    if packet.haslayer(scapy.IP) and packet.haslayer(scapy.TCP) and packet.haslayer(scapy.Raw):
        if packet[scapy.TCP].dport == 80 or packet[scapy.TCP].sport == 80 or packet[scapy.TCP].dport == 443 or packet[scapy.TCP].sport == 443:
            captured_packets.append(packet)
            # Extract IP packet ID (ID field is a 16-bit value that is used to uniquely identify a fragment within an IP datagram)
            ip_id = packet[scapy.IP].id
            print("IP ID:", ip_id)
            # Extract Source MAC Address
            source_mac = packet[scapy.Ether].src
            print("Source MAC Address:", source_mac)
            # Extract Destination MAC Address
            source_mac = packet[scapy.Ether].dst
            print("Source MAC Address:", source_mac)
            # Print the protocol type (TCP or UDP)
            print("Protocol Type:", "TCP" if packet.haslayer(scapy.TCP) else "UDP")
            #time.sleep(1000000)

network_interface = "Ethernet"
sniff_packets(network_interface)

# Now, the captured packets are stored in the 'captured_packets' list
print("Number of captured packets:", len(captured_packets))
