import socket
import pickle
import time
import threading

class CommandCameraClient:
    def __init__(self, server_host='localhost', server_port=5001, camera_name = None, reconnect_interval=5):
        self.server_host = server_host
        self.server_port = server_port
        self.camera_name = camera_name
        self.camera_id = None

        self.reconnect_interval = reconnect_interval
        self._stop_event = threading.Event()
        self.lock = threading.Lock()
        self.socket = None
        self.registered = False
        self.connecting = True
        self.receiver_thread = threading.Thread(target=self.receive_commands, daemon=True)

    def start(self):
        self.connect()
        self.receiver_thread.start()

    def stop(self, max_retries=5):
        disconnect_command = {
            'type' : 'disconnect',
            'client_type' : 'camera',
            'camera_id' : self.camera_id,
            'presenter' : 'camera'
        }
        self.send_command(disconnect_command)
        attempt = 0
        while not self._stop_event.is_set() and attempt < max_retries:
            print("[STOP] TCP CLIENT Đang đợi phản hồi từ SERVER")
            attempt += 1
            time.sleep(1)

        self._close_socket()
        if self.receiver_thread.is_alive():
            self.receiver_thread.join(timeout=2)
        print("[STOP] TCP CLIENT đã dừng.")

    def connect(self, max_retries=5):
        attempt = 0

        while not self._stop_event.is_set() and attempt < max_retries:
            try:
                self.socket = socket.create_connection((self.server_host, self.server_port))
                if self.camera_name is None and self.socket is not None:
                    self.camera_name = f"Camera - {self.socket.getsockname()[0]}"
                print(f"[INFO] Đã kết nối TCP tới {self.server_host}:{self.server_port}")
                return
            except Exception as e:
                attempt += 1
                print(f"[WARNING] Kết nối thất bại (lần {attempt}/{max_retries}): {e}. Thử lại sau {self.reconnect_interval}s...")
                time.sleep(self.reconnect_interval)

        # Nếu vượt quá số lần thử mà vẫn không thành công
        self.connecting = False
        print(f"[ERROR] Kết nối thất bại sau {max_retries} lần thử.")

    def reconnect(self):
        self._close_socket()
        time.sleep(self.reconnect_interval)
        self.connect()

    def _close_socket(self):
        if self.socket:
            try: self.socket.close()
            except: pass
            self.socket = None

    def send_command(self, command):
        if not self.socket: return
        try:
            data = pickle.dumps(command)
            with self.lock:
                self.socket.sendall(len(data).to_bytes(4, 'big') + data)
        except Exception as e:
            print(f"[ERROR] Gửi lệnh lỗi: {e}")
            self.reconnect()

    def receive_commands(self):
        while not self._stop_event.is_set():
            try:
                raw_len = self._recv_n_bytes(4)
                if not raw_len:
                    continue
                msg_len = int.from_bytes(raw_len, 'big')
                data = self._recv_n_bytes(msg_len)
                if data:
                    self.handle_response(pickle.loads(data))
            except Exception as e:
                print(f"[ERROR] Nhận lệnh lỗi: {e}")
                self.reconnect()

    def _recv_n_bytes(self, n):
        data = bytearray()
        while len(data) < n:
            packet = self.socket.recv(n - len(data))
            if not packet: return None
            data.extend(packet)
        return data


    def connection_as_camera(self):
        connect_command = {
            'type': 'connect',
            'client_type' : 'camera',
            'camera_name' : self.camera_name,
            'presenter' : 'camera'
        }
        self.send_command(connect_command)
        # Chờ tới khi đăng ký thành công
        while not self.registered:
            print("[INFO] Đang chờ server xác nhận đăng ký...")
            time.sleep(1)   
        
        return self.camera_id



    def handle_response(self, response):
        print(f"[RECEIVED] Response: {response}")
        if isinstance(response, dict): 
            if response['presenter'] == 'camera':
                if response['type'] == 'connect' and response['status'] == 'success':
                    self.registered = True
                    self.camera_id = response['camera_id']
                if response['type'] == 'disconnect' and response['status'] == 'success':
                    self._stop_event.set()
                    self.registerd = False



    