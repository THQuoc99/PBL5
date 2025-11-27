import time
from datetime import datetime
import queue
import socket
import threading
import cv2
import numpy as np
from pathlib import Path
import sys
import imagezmq
import zmq

ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.append(str(ROOT_DIR))
sys.path.append(str(ROOT_DIR / 'models'))

from .emotion_detector_128 import EmotionDetector
from models.objects.camera import Camera

class VideoServer:
    def __init__(self, host='0.0.0.0', port=5000):
        self.host = host
        self.port = port

        # Nhận ảnh từ camera
        self.tcp_ip = f"tcp://{self.host}:{self.port}"
        try:
            self.receiver = imagezmq.ImageHub(open_port=self.tcp_ip)
        except zmq.error.ZMQError as e:
            print(f"[ERROR] Failed to initialize ImageHub on {self.tcp_ip}: {e}")
            raise

        # Gửi ảnh tới GUI
        self.gui_ip = None
        self.sender = None

        # List các camera đang kết nối
        self.cameras = {}
        self.target_camera = None


        self._stop_event = threading.Event()
        

        # Chỉ giữ queue cho kết quả đã xử lý
        self.emotion_queue = queue.Queue(maxsize=20)
        self.processed_frames_queue = queue.Queue(maxsize=10)
        self.received_frames_queue = queue.Queue(maxsize=20)
        
        # Khởi tạo detector
        self.detector = EmotionDetector()
        
        # Thêm biến đếm frames
        self.received_frames = 0
        self.processed_frames = 0
        self.last_stats_minute = time.time()

        
        # Thêm biến đếm cảm xúc
        self.emotion_counts_per_10s = {
            "Happy": 0,
            "Neutral": 0,
            "Surprise": 0,
            "Negative": 0
        }


    def receive_frame_thread(self):
        poller = zmq.Poller()
        poller.register(self.receiver.zmq_socket, zmq.POLLIN)
        while not self._stop_event.is_set():
            try:
                socks = dict(poller.poll(timeout=100))  # Timeout 0.1 giây
                if self.receiver.zmq_socket in socks and socks[self.receiver.zmq_socket] == zmq.POLLIN:
                    camera_id, jpg_buffer = self.receiver.recv_jpg()

                    # Giải mã buffer JPEG thành mảng NumPy
                    frame = cv2.imdecode(np.frombuffer(jpg_buffer, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if frame is None:
                        print("[ERROR] Failed to decode JPEG frame")
                        self.receiver.send_reply(b'OK')  # Vẫn gửi phản hồi
                        continue

                    self.received_frames += 1
                    
                    # Đưa frame (mảng NumPy) vào queue
                    # Thay thế frame cũ nhất nếu queue đầy
                    try:
                        self.received_frames_queue.put_nowait((camera_id, frame))
                    except queue.Full:
                        try:
                            self.received_frames_queue.get_nowait()
                            self.received_frames_queue.put_nowait((camera_id, frame)) 
                        except queue.Empty:
                            print("[ERROR] Queue empty unexpectedly, skipping frame")
                        except Exception as e:
                            print(f"[ERROR] Error replacing frame in queue: {e}")

                    self.receiver.send_reply(b'OK')  # Gửi phản hồi cho client
            
            except Exception as e:
                print(f"[ERROR] Error in receive_frame_thread: {e}")
                continue
    
    def process_frame_thread(self):
        while not self._stop_event.is_set():
            try:
                cam_id, frame = self.received_frames_queue.get(timeout=1)
            except queue.Empty:
                continue

            # Phân tích cảm xúc
            processed_frame ,emotion = self.detector.process_frame(frame)
            self.processed_frames += 1

            if emotion is not None:
                for emo in emotion:
                    self.emotion_counts_per_10s[emo] += 1
                    self.cameras[cam_id]["emotion_counts"][emo] +=1
            

            # Lưu cảm xúc
            # Lưu cảm xúc vào emotion_queue
            try:
                self.emotion_queue.put_nowait((cam_id, emotion))
            except queue.Full:
                try:
                    self.emotion_queue.get_nowait()
                    self.emotion_queue.put_nowait((cam_id, emotion))
                except queue.Empty:
                    print("[ERROR] emotion_queue empty unexpectedly")
                except Exception as e:
                    print(f"[ERROR] Error replacing emotion in emotion_queue: {e}")

            # Nếu là camera GUI đang xem → cho vào queue hiển thị
            if cam_id == self.target_camera:
                try:
                    self.processed_frames_queue.put_nowait(processed_frame)
                except queue.Full:
                    try:
                        self.processed_frames_queue.get_nowait()
                        self.processed_frames_queue.put_nowait(processed_frame)
                    except queue.Empty:
                        print("[ERROR] processed_frames_queue empty unexpectedly")
                    except Exception as e:
                        print(f"[ERROR] Error replacing frame in processed_frames_queue: {e}")


            self.print_frame_stats()

  
    def send_frame_thread(self):
        while not self._stop_event.is_set():
            if self.sender is not None:
                try:
                    frame = self.processed_frames_queue.get(timeout=1)

                    # Encode JPEG
                    success, jpeg = cv2.imencode('.jpg', frame)
                    if not success:
                        print("[ERROR] Failed to encode JPEG frame")
                        continue

                    # Gửi qua ImageZMQ
                    self.sender.send_jpg("ProcessedFrame", jpeg)

                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"[ERROR] Failed to send frame: {e}")
                    continue
            time.sleep(0.04)


    def register_client(self, data):
        if data["client_type"] == "observer" and self.gui_ip is None:
            self.gui_ip = f"{data['host']}:{data['port']}"
            self.sender = imagezmq.ImageSender(connect_to=f"tcp://{data['host']}:{data['port']}")
            print(f"[SYSTEM] Observer connected: {data['host']}:{data['port']}")
            return "success"
        
        
        if data["client_type"] == "camera" and data['camera_id'] not in self.cameras:
            print(f"[CONNECTION] Registering camera client... {data['camera_id']}")
            try:
                self.cameras[data["camera_id"]] = {
                    'type': 'camera',
                    'camera_name' : data['camera_name'],
                    'emotion_counts' : {
                        "Happy": 0,
                        "Neutral": 0,
                        "Surprise": 0,
                        "Negative": 0
                    }
                }
                return "success"
            except Exception as e:
                print(f"[ERROR] Error saving Camera record to DB: {e}")
        return "fail"

    def unregister_client(self, data):
        if data["client_type"] == "observer" and self.sender is not None and self.gui_ip == f"{data['host']}:{data['port']}":
            self.sender.close()
            self.sender = None
            if self.remove_target_ip(data) == 'success':
                self.gui_ip = None
                print(f"[SYSTEM] Observer disconnected: {data['host']}:{data['port']}")
                return "success"
        elif data["client_type"] == "camera":
            print(f"[CONNECTION] Unregistering camera client... {data['camera_id']}")
            try:
                del self.cameras[data["camera_id"]]
                return "success"
            except Exception as e:
                print(f"[ERROR] Error deleting Camera record from DB: {e}")
        return "fail"

    def update_target_ip(self, data):
        gui_ip = f"{data['host']}:{data['port']}"
        target_camera = data["target_camera"]

        print(f"[OBSERVATION] Updating target IP for client {gui_ip} to {target_camera}")
        if gui_ip == self.gui_ip:
            self.target_camera = target_camera
            print(f"[OBSERVATION] Target : {self.target_camera}")
            return "success"
        return "fail"
    
    def remove_target_ip(self, data):
        gui_ip = f"{data['host']}:{data['port']}"
        if gui_ip == self.gui_ip:
            print(f"[OBSERVATION] Removing target IP for client {gui_ip}")
            self.target_camera = None
            return "success"
        return "fail"

    def start(self):
        print(f"{'='*32}")
        print(f"[SYSTEM] Video Server is running on:")
        print(f"IP: {self.host}")
        print(f"Port: {self.port}")
        print(f"TCP IP : {self.tcp_ip}")
        print(f"{'='*32}")

        self.receiver_thread = threading.Thread(target=self.receive_frame_thread, daemon=True)
        self.processor_thread = threading.Thread(target=self.process_frame_thread, daemon=True)
        self.sender_thread = threading.Thread(target=self.send_frame_thread, daemon=True)


        self.receiver_thread.start()
        self.processor_thread.start()
        self.sender_thread.start()

    def stop(self):
        print(f"{'='*32}")
        print("[SYSTEM] Video Server is stopping...")
        print(f"{'='*32}")
        self._stop_event.set()
        
        print(f"[SYSTEM] Waiting for receiver_thread to stop...")
        self.receiver_thread.join(timeout=2.0)
        if self.receiver_thread.is_alive():
            print("[WARNING] receiver_thread did not stop in time")
        print(f"[SYSTEM] Waiting for processor_thread to stop...")
        self.processor_thread.join(timeout=2.0)
        if self.processor_thread.is_alive():
            print("[WARNING] processor_thread did not stop in time")
        print(f"[SYSTEM] Waiting for sender_thread to stop...")
        self.sender_thread.join(timeout=2.0)
        if self.sender_thread.is_alive():
            print("[WARNING] sender_thread did not stop in time")
        print(f"[SYSTEM] Active threads after stop: {threading.active_count()}")

        if self.receiver is not None:
            print("[SYSTEM] Closing imagezmq receiver...")
            self.receiver.close()
            self.receiver = None
        if self.sender is not None:
            print("[SYSTEM] Closing imagezmq sender...")
            self.sender.close()
            self.sender = None
        print("[SYSTEM] Video Server stopped.")

    def print_frame_stats(self):
        current_time = time.time()
        elapsed = current_time - self.last_stats_minute
        if elapsed >= 60:
            fps_received = self.received_frames / elapsed
            fps_processed = self.processed_frames / elapsed

            print(f"\n{'='*10}")
            print(f"Statistics (last {elapsed:.1f} seconds):")
            print(f"Frames received: {self.received_frames} ({fps_received:.2f} fps)")
            print(f"Frames processed: {self.processed_frames} ({fps_processed:.2f} fps)")
            print(f"{'='*10}\n")
            
            # Reset counters
            self.received_frames = 0
            self.processed_frames = 0
            
            self.last_stats_minute = current_time
        
    def reset_counter_per_10s(self):
        for emotion, count in self.emotion_counts_per_10s.items():
                self.emotion_counts_per_10s[emotion] = 0
    
    def reset_counter_per_minute(self):
        for _ , camera in self.cameras.items():
            for emotion, count in camera["emotion_counts"].items():
                camera["emotion_counts"][emotion] = 0
    



