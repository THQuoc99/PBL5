class CameraSerializer:
    def __init__(self, camera_id=None, name=None, ip_address=None, location=None, description=None):
        self.camera_id = camera_id 
        self.name = name
        self.ip_address = ip_address
        self.location = location
        self.description = description
    
    def to_dict(self):
        return {
            "camera_id": self.camera_id,
            "name": self.name,
            "ip_address": self.ip_address,
            "location": self.location,
            "description": self.description
        }
    
    @staticmethod
    def from_dict(data):
        return CameraSerializer(**data)