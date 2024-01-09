import scapy.all as scapy
import time

captured_packets = []

def sniff_packets(interface):
    scapy.sniff(iface=interface, store=False, prn=process_packet, filter="tcp port 80 or tcp port 443")

def process_packet(packet):
    if packet.haslayer(scapy.IP) and packet.haslayer(scapy.TCP) and packet.haslayer(scapy.Raw):
        if packet[scapy.TCP].dport == 80 or packet[scapy.TCP].sport == 80 or packet[scapy.TCP].dport == 443 or packet[scapy.TCP].sport == 443:
            captured_packets.append(packet)
            # Print all details of the packet
            print("Packet Details:")
            packet.show()
            print("\nRaw Layer Details:")
            packet[scapy.Raw].show()
            print(type(packet))
            time.sleep(1000000)

network_interface = "Ethernet"
sniff_packets(network_interface)

# Now, the captured packets are stored in the 'captured_packets' list
print("Number of captured packets:", len(captured_packets))
