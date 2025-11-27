from .command_handler import CommandHandler
from .emotions_recorder import EmotionsRecorder
from .socket_utils import SocketHelper
import threading
import socket
from pathlib import Path
import sys
import time


ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.append(str(ROOT_DIR))
sys.path.append(str(ROOT_DIR / 'models'))

class CommandServer:
    def __init__(self,video_server, host = '0.0.0.0', port= 5000):
        self.host = host
        self.port = port
        self.video_server = video_server
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.bind((host, port))
        self._stop_event = threading.Event()
        self.server_id = CommandHandler.get_server_id()
        self.emotions_recorder = EmotionsRecorder(self.video_server, self.server_id)
        self.command_handler = CommandHandler(self.video_server, self.emotions_recorder)
        self.client_threads = []

    def handle_client(self, conn, addr):
        while not self._stop_event.is_set():
            try:
                command = SocketHelper.recv_data(conn)

                response = self.command_handler.handle(command, conn, addr)
                
                SocketHelper.send_data(conn, response)
                time.sleep(1)
            except Exception as e:
                print(f"[ERROR] Error handling client: {e}")
                break

    def start(self):
        print(f"{'='*32}")
        print(f"[SYSTEM] TCP Command  Server is running on:")
        print(f"IP: {self.host}")
        print(f"Port: {self.port}")
        print(f"{'='*32}")
        self.socket.listen(5)
        self.recorder_thread = threading.Thread(target=self.emotions_recorder.start, daemon=True)
        self.recorder_thread.start()

        while not self._stop_event.is_set():
            try:
                conn, addr = self.socket.accept()
                print(f"[COMMAND SERVER] Connection from {addr[0]} : {addr[1]}")

                client_thread = threading.Thread(target=self.handle_client, args=(conn, addr))
                self.client_threads.append(client_thread)
                client_thread.start()
            except OSError as e:
                if self._stop_event.is_set():

                    break  # Nếu đang shutdown thì break nhẹ nhàng
                print(f"[TCP] OSError during accept: {e}")
                break
        
    def stop(self):
        print("[SYSTEM] Shutting down command server...")
        self._stop_event.set()
        self.emotions_recorder.stop()

        # Đóng socket
        try:
            self.socket.shutdown(socket.SHUT_RDWR)
        except Exception:
            pass
        self.socket.close()

        # Chờ recorder_thread
        print("[SYSTEM] Waiting for recorder_thread to stop...")
        self.recorder_thread.join(timeout=5.0)
        if self.recorder_thread.is_alive():
            print("[WARNING] recorder_thread did not stop in time")

        # Chờ handle_client_thread
        for thread in self.client_threads:
            print("[SYSTEM] Waiting for handle_client_thread to stop...")
            thread.join(timeout=5.0)
            if thread.is_alive():
                print("[WARNING] handle_client_thread did not stop in time")
        
        print("[SYSTEM] Command server stopped")