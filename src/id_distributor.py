# -*- coding: utf-8 -*-

import socket
from sys import argv

counter = int(argv[1])
print(f"counter: {counter}")
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(("0.0.0.0", 8621))
server_socket.listen(10240)
while counter > 0:
    client_socket, address = server_socket.accept()
    print(counter, end="\n")
    client_socket.send((str(counter)).encode())
    client_socket.close()
    counter -= 1
