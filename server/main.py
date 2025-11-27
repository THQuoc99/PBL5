import time
import threading

from server_utils.video_server import VideoServer
from server_utils.command_server import CommandServer




SERVER_IP = "127.0.0.1"
UDP_SERVER_PORT = 6969
TCP_SERVER_PORT = 9696



if __name__ == "__main__":
    server = VideoServer(SERVER_IP, UDP_SERVER_PORT)
    command_server = CommandServer(server, SERVER_IP,TCP_SERVER_PORT)

    video_thread = threading.Thread(target=server.start, daemon=True)
    command_thread = threading.Thread(target=command_server.start, daemon=True)

    
    try:
        print("[SYSTEM] Servers are running. Press Ctrl+C to stop.\n")
        # Start both servers
        command_thread.start()
        video_thread.start()

        # Giữ chương trình chạy và kiểm tra tín hiệu ngắt
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nReceived keyboard interrupt, shutting down...")
    finally:
        server.stop()
        command_server.stop()
        
        command_thread.join(timeout=2.0)
        video_thread.join(timeout=2.0)
        # Optionally wait for threads to finish
        
        print("[SYSTEM] Servers shutdown completed.")