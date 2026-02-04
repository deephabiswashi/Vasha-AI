import socket
import sys

def check_ws():
    host = "127.0.0.1"
    port = 5000
    path = "/stream_audio"

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(3)
        sock.connect((host, port))
        
        request = (
            f"GET {path} HTTP/1.1\r\n"
            f"Host: {host}:{port}\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n"
            "Sec-WebSocket-Version: 13\r\n"
            "\r\n"
        )
        
        sock.sendall(request.encode())
        response = sock.recv(4096).decode()
        
        print(f"--- Server Response ---\n{response}\n-----------------------")
        
        if "101 Switching Protocols" in response:
            print("✅ WebSocket Endpoint reachable and working!")
            return True
        else:
            print("❌ WebSocket Handshake failed.")
            return False
            
    except Exception as e:
        print(f"❌ Connection Failed: {e}")
        return False
    finally:
        sock.close()

if __name__ == "__main__":
    if check_ws():
        sys.exit(0)
    else:
        sys.exit(1)
