import time
import threading
import queue
import cv2
import imagezmq

class VideoSender:
    def __init__(self, server_host='localhost', server_port=5000):
        self.server_host = server_host
        self.server_port = server_port
        self.sender = imagezmq.ImageSender(connect_to=f"tcp://{server_host}:{server_port}")
        self._stop_event = threading.Event()
        self.frame_queue = queue.Queue(maxsize=1)
        self.cam_name = "server_camera"
        self.camera_id = None
        
        # Khởi tạo camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise Exception("Không thể mở camera")
            
        # Khởi tạo các thread
        self.capture_thread = threading.Thread(target=self.capture_frame_thread, daemon=True)
        self.send_thread = threading.Thread(target=self.send_frame_thread, daemon=True)
        
    
    def capture_frame_thread(self):
        while not self._stop_event.is_set():
            try:
                ret, frame = self.cap.read()
                if not ret:
                    print("Không thể đọc frame từ camera")
                    continue
                    
                # Resize frame để giảm kích thước
                # frame = cv2.resize(frame, (640, 640))
                
                if not self.frame_queue.empty():
                    self.frame_queue.get()  # Xóa frame cũ
                self.frame_queue.put(frame)
                
            except Exception as e:
                print(f"Lỗi khi đọc frame: {e}")
                continue

    
    def send_frame_thread(self):
        while not self._stop_event.is_set():
            try:
                frame = self.frame_queue.get(timeout=1)
                _, jpg_buffer = cv2.imencode(".jpg", frame)
                self.sender.send_jpg(self.camera_id, jpg_buffer)
                time.sleep(0.04)
            except Exception as e:
                print(f"[ERROR] Error while sending frame: {e}")
                continue
    
    def start(self, camera_id):
        print(f"\n{'='*50}")
        print(f"UDP Client is running:")
        print(f"Server IP: {self.server_host}")
        print(f"Server Port: {self.server_port}")
        print(f"{'='*50}\n")

        self.camera_id = camera_id

        # Bắt đầu các thread
        self.capture_thread.start()
        self.send_thread.start()
        
    def stop(self):
        print("Đang dừng client...")
        self._stop_event.set()
        
        # Đóng camera
        self.cap.release()
        
        # Đợi các thread kết thúc
        if self.capture_thread.is_alive():
            self.capture_thread.join(timeout=1.0)
        if self.send_thread.is_alive():
            self.send_thread.join(timeout=1.0)
        
        # Đóng sender socket
        try:
            self.sender.close()  # <-- THÊM DÒNG NÀY
        except Exception as e:
            print(f"[WARNING] Không thể đóng sender: {e}")
            
        print("Client đã dừng")

        