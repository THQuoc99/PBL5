# services/daily_emotion_service.py

import uuid
from datetime import datetime,timedelta
from models.objects.daily_emotions import DailyEmotions
from statistics import mean

class DailyEmotionsService:
    @staticmethod
    def get_all_emotions():
        daily_emotions_dict_list = [{daily_emotions.to_dict()} for daily_emotions in DailyEmotions.get_all_daily_emotions()]
        return {
            "type" : "daily_emotions_list",
            "list" : daily_emotions_dict_list
        }

    @staticmethod
    def get_emotions_list_by_camera_id(camera_id):
        daily_emotions = DailyEmotions.get_daily_emotions_by_camera_id(camera_id)
        daily_emotions_dict_list = [{daily_emotions.to_dict()} for daily_emotions in daily_emotions]
        return {
            "type" : "daily_emotions_list",
            "camera_id" : camera_id,
            "list" : daily_emotions_dict_list
        }

    @staticmethod
    def get_emotion_by_id(emotion_id):
        return DailyEmotions.get_daily_emotion_by_id(emotion_id)

    @staticmethod
    def create_emotion(timestamp, camera_id, negative, surprise, happy, neutral):
        emotion = DailyEmotions(
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
    
    @staticmethod
    def get_camera_emotions_in_minutes(minutes, camera_id, end_time= None):
        now = datetime.now()
        if end_time is not None:
            now = end_time
        start_time = now - timedelta(minutes=minutes)
        # Thay vì lấy tất cả cảm xúc và lọc, bạn sử dụng get_emotions_by_time_range
        emotions = DailyEmotions.get_camera_emotions_by_time_range(start_time, now, camera_id)
        emotion_dict_list = [emotion.to_dict_with_ratios() for emotion in emotions]
        return emotion_dict_list
    
    @staticmethod
    def get_camera_list_emotions_in_minutes(minutes, server_id, end_time= None):
        now = datetime.now()
        if end_time is not None:
            now = end_time
        start_time = now - timedelta(minutes=minutes)
        # Thay vì lấy tất cả cảm xúc và lọc, bạn sử dụng get_emotions_by_time_range
        emotions = DailyEmotions.get_camera_list_emotions_by_time_range(start_time, now, server_id)
        emotion_dict_list = [emotion.to_dict_with_ratios() for emotion in emotions]
        return emotion_dict_list
    