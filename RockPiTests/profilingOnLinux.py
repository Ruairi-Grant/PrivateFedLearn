import time
import multiprocessing as mp
import psutil
import numpy as np
import socket

import dp_test.dp_sgd_mnist.Central_mnist as Central_mnist

def test_network_and_cpu(duration=60):
    # Define the target host and port for sending and receiving packets
    target_host = '192.168.0.10'
    target_port = 12345

    # Create a UDP socket for sending and receiving packets
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # Set a timeout for the socket operations
    sock.settimeout(1)

    # Get the start time
    start_time = time.time()

    # Initialize counters for sent and received packets
    sent_packets = 0
    received_packets = 0

    # Loop until the specified duration is reached
    while time.time() - start_time < duration:
        # Send a packet
        sock.sendto(b"TestPacket", (target_host, target_port))
        sent_packets += 1

        # Try to receive a packet
        try:
            data, _ = sock.recvfrom(1024)
            if data:
                received_packets += 1
        except socket.timeout:
            pass

        # Measure CPU load using psutil
        cpu_load = psutil.cpu_percent()

        # Print CPU load and packet statistics
        print(f"Sent: {sent_packets} | Received: {received_packets}", end='\r')

    # Close the socket
    sock.close()

    # Print a final message
    print("\nTest completed.")


def sleep_test(duration=60):
    time.sleep(duration)

def test_script():

    Central_mnist.main()

def monitor(target):
    worker_process = mp.Process(target=target)
    worker_process.start()
    p = psutil.Process(worker_process.pid)

    # log cpu usage of `worker_process` every 10 ms
    cpu_percents = []
    while worker_process.is_alive():
        cpu_percents.append(p.cpu_percent())
        time.sleep(0.1)

    worker_process.join()
    return cpu_percents


if __name__ == "__main__":
    start_pkts_sent = psutil.net_io_counters().packets_sent
    start_pkts_recv = psutil.net_io_counters().packets_recv
    cpu_percents = monitor(target=test_network_and_cpu)
    end_pkts_sent = psutil.net_io_counters().packets_sent
    end_pkts_recv = psutil.net_io_counters().packets_recv

    final_pkts_sent = end_pkts_sent - start_pkts_sent
    final_pkts_recv = end_pkts_recv - start_pkts_recv

    print(f"CPU usage: {np.mean(cpu_percents)}%")
    print(f"Average CPU usage: {psutil.getloadavg()}")
    print(f"Total packets sent: {final_pkts_sent}")
    print(f"Total packets received: {final_pkts_recv}")
