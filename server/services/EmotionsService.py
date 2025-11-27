# services/emotion_service.py

import uuid
from datetime import datetime,timedelta
from models.objects.emotions import Emotions
from statistics import mean
class EmotionsService:
    @staticmethod
    def get_all_emotions():
        return Emotions.get_all_emotions()

    @staticmethod
    def get_emotion_by_id(emotion_id):
        return Emotions.get_emotion_by_id(emotion_id)

    @staticmethod
    def save_emotion(timestamp, camera_id,negative, surprise, happy, neutral):
        latest_record = Emotions.get_emotion_by_time_and_camera_id(timestamp, camera_id)
        if latest_record:
            # Cập nhật bản ghi hiện tại
            latest_record.negative += negative
            latest_record.surprise += surprise
            latest_record.happy += happy
            latest_record.neutral += neutral
            latest_record.save()
            return latest_record
        else:
            # Tạo bản ghi mới nếu không có
            emotion = Emotions(
                id=None,
                timestamp=timestamp or datetime.now(),
                camera_id=camera_id,
                negative=negative,
                surprise=surprise,
                happy=happy,
                neutral=neutral
            )
            emotion.save()
            return emotion
        
    #Lấy tỉ lệ cảm xúc trong k ngày gần nhất
    @staticmethod
    def get_camera_emotions_in_days(days, camera_id, end_time = None):
        now = datetime.now()
        if end_time is not None:
            now = end_time
        start_time = now - timedelta(days=days)
        end_time = now
        emotions = Emotions.get_camera_emotions_by_time_range(start_time, end_time, camera_id)

        emotion_dict_list = [emotion.to_dict_with_ratios() for emotion in emotions]
        return emotion_dict_list
    
    @staticmethod
    def get_camera_list_emotions_in_days(days, server_id, end_time = None):
        now = datetime.now()
        if end_time is not None:
            now = end_time
        start_time = now - timedelta(days=days)
        end_time = now
        emotions = Emotions.get_camera_list_emotions_by_time_range(start_time, end_time, server_id)

        emotion_dict_list = [emotion.to_dict_with_ratios() for emotion in emotions]
        return emotion_dict_list
