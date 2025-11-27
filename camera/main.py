import time
from core.video_sender import VideoSender
from core.command_camera_client import CommandCameraClient


from core.config import VIDEO_SERVER_CONFIG, COMMAND_SERVER_CONFIG, CAMERA_NAME




if __name__ == "__main__":
    video_client = None
    command_client = None
    try:
        video_client = VideoSender(VIDEO_SERVER_CONFIG["IP"], VIDEO_SERVER_CONFIG["PORT"])
        command_client = CommandCameraClient(COMMAND_SERVER_CONFIG["IP"], COMMAND_SERVER_CONFIG["PORT"], CAMERA_NAME)
        command_client.start()

        wait_count = 0
        while command_client.socket is None and command_client.connecting:
            time.sleep(0.5)
            wait_count += 1
            if wait_count >= 40:  # 20s timeout
                raise RuntimeError("Timeout khi kết nối tới Command Server")
        
        if not command_client.connecting:
            raise RuntimeError("Timeout khi kết nối tới Command Server") 


        camera_id = command_client.connection_as_camera()

        print("[INFO] Bắt đầu UDP Video Client...")
        video_client.start(camera_id)

        # Giữ chương trình chạy
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n[CTRL+C] Đang dừng tất cả...")
    except Exception as e:
        print(f"\n[ERROR] Lỗi trong quá trình kết nối: {e}")
    finally:
        if video_client is not None: 
            video_client.stop()
        if command_client is not None :
            command_client.stop()