import uuid
import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from db.db_helper import DBHelper

class Emotions:
    def __init__(self, id, timestamp, camera_id, negative, surprise, happy, neutral):
        self.id = id
        self.timestamp = timestamp
        self.camera_id = camera_id
        self.negative = negative
        self.surprise = surprise
        self.happy = happy
        self.neutral = neutral
    
    def to_dict(self):
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "camera_id": self.camera_id,
            "negative": self.negative,
            "surprise": self.surprise,
            "happy": self.happy,
            "neutral": self.neutral
        }
    
    def to_dict_with_ratios(self):
        total_emotion_count = self.happy + self.neutral + self.negative + self.surprise
        if total_emotion_count == 0:
            return {
                "id": self.id,
                "timestamp": self.timestamp,
                "camera_id": self.camera_id,
                "negative": 0,
                "surprise": 0,
                "happy": 0,
                "neutral": 0
            }
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "camera_id": self.camera_id,
            "negative": self.negative / total_emotion_count,
            "surprise": self.surprise / total_emotion_count,
            "happy": self.happy / total_emotion_count,
            "neutral": self.neutral / total_emotion_count
        }
    
    @staticmethod
    def from_dict(data):
        return Emotions(**data)


    @classmethod
    def get_all_emotions(cls):
        db_helper = DBHelper()
        query = "SELECT * FROM emotions"
        result = db_helper.fetch_all(query)
        return [cls(**emotion) for emotion in result]

    @classmethod
    def get_emotion_by_id(cls, id):
        db_helper = DBHelper()
        query = "SELECT * FROM emotions WHERE id = %s"
        result = db_helper.fetch_one(query, (id,))
        if result:
            return cls(**result)
        return None
    @classmethod
    def get_camera_emotions_by_time_range(cls, start_time, end_time, camera_id):
        db_helper = DBHelper()
        query = "SELECT * FROM emotions WHERE camera_id = %s AND `timestamp` BETWEEN %s AND %s"
        results = db_helper.fetch_all(query, (camera_id, start_time, end_time))
        return [cls(**emotion) for emotion in results]
    
    @classmethod
    def get_camera_list_emotions_by_time_range(cls, start_time, end_time, server_id):
        db_helper = DBHelper()
        query = "SELECT * FROM emotions WHERE camera_id != %s AND `timestamp` BETWEEN %s AND %s"
        results = db_helper.fetch_all(query, (server_id, start_time, end_time))
        return [cls(**emotion) for emotion in results]

    @classmethod
    def get_emotion_by_time_and_camera_id(cls, timestamp, camera_id):
        db_helper = DBHelper()
        query = "SELECT * FROM emotions WHERE timestamp = %s AND camera_id = %s"
        result = db_helper.fetch_one(query, (timestamp, camera_id))
        if result:
            return cls(**result)
        return None

    def save(self):
        db_helper = DBHelper()
        if self.id is not None:
            query = """UPDATE emotions 
                       SET timestamp = %s, camera_id = %s, negative = %s, surprise = %s, happy = %s, neutral = %s 
                       WHERE id = %s"""
            db_helper.execute(query, (self.timestamp, self.camera_id, self.negative, self.surprise, self.happy, self.neutral, self.id))
        else:
            self.id = str(uuid.uuid4())
            query = """INSERT INTO emotions (id, timestamp, camera_id, negative, surprise, happy, neutral) 
                       VALUES (%s, %s, %s, %s, %s, %s, %s)"""
            db_helper.execute(query, (self.id, self.timestamp, self.camera_id, self.negative, self.surprise, self.happy, self.neutral))

    def delete(self):
        db_helper = DBHelper()
        query = "DELETE FROM emotions WHERE id = %s"
        db_helper.execute(query, (self.id,))
