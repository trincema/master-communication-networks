import psutil
import time

def count_syn_connections(port):
    connections = psutil.net_connections(kind='tcp')
    syn_connections = [conn for conn in connections if conn.status == psutil.CONN_SYN_RECV and conn.laddr.port == port]
    return len(syn_connections)

if __name__ == "__main__":
    port = 80  # Specify the port you want to monitor
    while True:
        syn_count = count_syn_connections(port)
        print(f"Number of SYN connections received at port {port}: {syn_count}")
        time.sleep(1)
