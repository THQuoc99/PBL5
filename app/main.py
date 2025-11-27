import sys
from PyQt6.QtWidgets import QApplication
from core.app_controller import AppController
from core.command_gui_client import CommandGUIClient
from core.video_receiver import VideoReceiver

from core.config import VIDEO_RECEIVER_CONFIG, COMMAND_SERVER_CONFIG, STYLE_CONFIG

import threading
import time

def main():
    command_gui_client = None
    video_receiver = None

    try:
        command_gui_client = CommandGUIClient(VIDEO_RECEIVER_CONFIG['PORT'], COMMAND_SERVER_CONFIG['IP'], COMMAND_SERVER_CONFIG['PORT'])
        command_gui_client.start()

        wait_count = 0
        while command_gui_client.socket is None and command_gui_client.connecting:
            time.sleep(0.5)
            wait_count += 1
            if wait_count >= 40:  # 20s timeout
                raise RuntimeError("Timeout khi kết nối tới Command Server")
        
        if not command_gui_client.connecting:
            raise RuntimeError("Timeout khi kết nối tới Command Server") 

        video_receiver = VideoReceiver(command_gui_client.receiver_host,VIDEO_RECEIVER_CONFIG['PORT'])
        video_receiver_thread = threading.Thread(target=video_receiver.run, daemon=True)
        video_receiver_thread.start()

        # Gửi lệnh đăng ký sau khi đã kết nối
        command_gui_client.connection_as_observer()


        app = QApplication(sys.argv)
        
        controller = AppController(command_gui_client, video_receiver)

        sys.exit(app.exec())
    except KeyboardInterrupt:
        print("\nReceived keyboard interrupt, shutting down...")
    except Exception as e:
        print(f"[ERROR] Lỗi xảy ra trong quá trình kết nối : {e}")
    finally:
        print('[SYSTEM] Client shutting down')
        if command_gui_client is not None :
            command_gui_client.stop()
            
        if video_receiver is not None :
            video_receiver.stop()
            video_receiver_thread.join(timeout=2.0)
        
        # Optionally wait for threads to finish
        
        print("[SYSTEM] Servers shutdown completed.")

if __name__ == "__main__":
    main()