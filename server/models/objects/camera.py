import uuid
import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from db.db_helper import DBHelper


class Camera:
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
        return Camera(**data)


    @classmethod
    def get_all_cameras(cls):
        db_helper = DBHelper()
        query = "SELECT * FROM camera"
        result = db_helper.fetch_all(query)
        return [cls(**camera) for camera in result]

    @classmethod
    def get_camera_by_id(cls, camera_id):
        db_helper = DBHelper()
        query = "SELECT * FROM camera WHERE camera_id = %s"
        result = db_helper.fetch_one(query, (camera_id,))
        if result:
            return cls(**result)
        return None
    
    @classmethod 
    def get_camera_id_by_ip(cls, ip_address):
        db_helper = DBHelper()
        query = "SELECT camera_id FROM camera WHERE ip_address = %s"
        result = db_helper.fetch_one(query, (ip_address,))
        if result:
            return result["camera_id"]
        else:
            return None
    
    def get_or_create(self):
        db_helper = DBHelper()
        exists = db_helper.fetch_one("SELECT camera_id, name FROM camera WHERE ip_address = %s", (self.ip_address,))
        if exists:
            camera_id = exists['camera_id']
            existing_name = exists['name']
            
            # Nếu name khác thì cập nhật
            if self.name and self.name != existing_name:
                update_query = "UPDATE camera SET name = %s WHERE camera_id = %s"
                db_helper.execute(update_query, (self.name, camera_id))

            return camera_id
        else:
            self.camera_id = str(uuid.uuid4()) 
            query = """INSERT INTO camera (camera_id, name, ip_address, location, description) 
                       VALUES (%s, %s, %s, %s, %s)"""
            db_helper.execute(query, (self.camera_id, self.name, self.ip_address, self.location, self.description)) 
            return self.camera_id


    
    def save(self):
        db_helper = DBHelper()
        # Check xem camera đã có trong DB hay chưa
        exists = db_helper.fetch_one("SELECT 1 FROM camera WHERE camera_id = %s", (self.camera_id,))
        if exists:
            query = """UPDATE camera 
                       SET name = %s, ip_address = %s, location = %s, description = %s 
                       WHERE camera_id = %s"""
            db_helper.execute(query, (self.name, self.ip_address, self.location, self.description, self.camera_id))
        else:
            self.camera_id = str(uuid.uuid4()) 
            query = """INSERT INTO camera (camera_id, name, ip_address, location, description) 
                       VALUES (%s, %s, %s, %s, %s)"""
            db_helper.execute(query, (self.camera_id, self.name, self.ip_address, self.location, self.description))
    def delete(self):
        db_helper = DBHelper()
        query = "DELETE FROM camera WHERE camera_id = %s"
        db_helper.execute(query, (self.camera_id,))
