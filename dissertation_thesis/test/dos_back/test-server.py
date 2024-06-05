import socket
import threading
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(filename='back_attack_log.txt', level=logging.INFO, format='%(asctime)s - %(message)s')

# Dictionary to count connections from each IP
connection_count = {}

# Threshold for considering an attack
CONNECTION_THRESHOLD = 100  # Example threshold, adjust as needed
TIME_WINDOW = 60  # Time window in seconds to count connections

def handle_client(client_socket, client_address):
    ip = client_address[0]
    current_time = datetime.now()

    # Log the connection attempt
    logging.info(f"Connection attempt from {ip}")

    # Update connection count
    if ip not in connection_count:
        connection_count[ip] = []
    connection_count[ip].append(current_time)

    # Remove outdated entries
    connection_count[ip] = [timestamp for timestamp in connection_count[ip] if (current_time - timestamp).seconds < TIME_WINDOW]
    client_socket.close()

def start_server(host, port):
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((host, port))
    server.listen(5)
    logging.info(f"Server started on {host}:{port}")

    while True:
        client_socket, client_address = server.accept()
        logging.info("accepted client")
        client_handler = threading.Thread(target=handle_client, args=(client_socket, client_address))
        client_handler.start()

# Example usage
if __name__ == "__main__":
    HOST = '0.0.0.0'  # Listen on all available interfaces
    PORT = 9999         # Port to listen on
    start_server(HOST, PORT)
