from scapy.all import sniff, Ether, IP, TCP
import time

def packet_callback(packet):
    if Ether in packet and IP in packet and TCP in packet:
        src_ip = packet[IP].src
        dst_ip = packet[IP].dst
        src_port = packet[TCP].sport
        dst_port = packet[TCP].dport
        print(f"Source: {src_ip}:{src_port} -> Destination: {dst_ip}:{dst_port}")
        src_ip = len(packet['TCP'].payload)
        length = len(packet)
        print("length: " + str(length))
        time.sleep(10000)

# Start sniffing on the network interface
sniff(prn=packet_callback, store=0)

