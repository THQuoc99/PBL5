# services/camera_service.py
from models.objects.camera import Camera
import uuid

class CameraService:
    @staticmethod
    def create_camera(name, ip_address=None, location=None, description=None):
        # Kiểm tra điều kiện trước khi tạo camera
        if not name:
            raise ValueError("Camera name is required")

        # Tạo camera mới với UUID tự động
        camera = Camera(
            camera_id=str(uuid.uuid4()), 
            name=name, 
            ip_address=ip_address, 
            location=location, 
            description=description
        )
        
        # Lưu camera vào cơ sở dữ liệu
        camera.save()
        
        return camera
    
    @staticmethod
    def get_or_create(name, ip_address):
        try:
            camera_rc = Camera(
                camera_id = None,
                name = name,
                ip_address = ip_address,
                location = "Không",
                description = "Không"
            )
            camera_id = camera_rc.get_or_create()
            return camera_id
        except Exception as e:
                print(f"[ERROR] Error saving Camera record to DB: {e}")

    @staticmethod
    def get_camera_by_id(camera_id):
        # Lấy camera theo camera_id
        camera = Camera.get_camera_by_id(camera_id)
        return camera.to_dict() if camera else None

    @staticmethod
    def get_all_cameras():
        # Lấy tất cả các camera
        return Camera.get_all()

    @staticmethod
    def update_camera(camera_id, name=None, ip_address=None, location=None, description=None):
        # Lấy camera theo id
        camera = Camera.get_by_id(camera_id)
        if camera is None:
            raise ValueError(f"No camera found with id: {camera_id}")
        
        # Cập nhật thông tin camera
        camera.name = name if name else camera.name
        camera.ip_address = ip_address if ip_address else camera.ip_address
        camera.location = location if location else camera.location
        camera.description = description if description else camera.description

        # Lưu lại camera đã cập nhật
        camera.save()

        return camera

    @staticmethod
    def delete_camera(camera_id):
        # Lấy camera theo id
        camera = Camera.get_by_id(camera_id)
        if camera is None:
            raise ValueError(f"No camera found with id: {camera_id}")

        # Xóa camera
        camera.delete()

        return {"message": "Camera deleted successfully"}
    
    @staticmethod
    def get_server_id():
        return Camera.get_camera_id_by_ip('0.0.0.0')
