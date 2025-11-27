import socket
import pickle
import cv2
import numpy as np
import imagezmq

from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QImage

class VideoReceiver(QThread):
    change_pixmap_signal = pyqtSignal(QImage)

    def __init__(self, host='127.0.0.1', port=6699):
        super().__init__()
        self.host = host
        self.port = port
        self.receiver = imagezmq.ImageHub(open_port=f"tcp://{host}:{port}")
        self._run_flag = True

    def run(self):
        print(f"[SYSTEM] VideoReceiver listening on tcp://{self.host}:{self.port}")
        while self._run_flag:
            try:
                # Nhận frame JPEG từ server
                cam_id, jpg_buffer = self.receiver.recv_jpg()

                # Giải mã JPEG thành frame
                frame = cv2.imdecode(np.frombuffer(jpg_buffer, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    print("[ERROR] Failed to decode JPEG frame")
                    self.receiver.send_reply(b'OK')
                    continue

                # Chuyển frame thành QImage
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                if qt_image.isNull():
                    print("Invalid QImage created")
                    self.receiver.send_reply(b'OK')
                    continue
                self.change_pixmap_signal.emit(qt_image)
                self.receiver.send_reply(b'OK')
            except Exception as e:
                print(f"[ERROR] Error in video thread: {e}")
                continue

    def stop(self):
        self._run_flag = False
        self.receiver.close()
        print(f"[STOP] VideoReceiver stopped.")
        self.wait()