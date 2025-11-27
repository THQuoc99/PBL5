from models.objects.monthly_emotions import MonthlyEmotions
from datetime import datetime, timedelta
from statistics import mean
class MonthlyEmotionsService:
    
    @staticmethod
    def save_emotion(timestamp, camera_id,negative, surprise, happy, neutral):
        latest_record = MonthlyEmotions.get_emotion_by_time_and_camera_id(timestamp, camera_id)
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
            emotion = MonthlyEmotions(
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
    def get_emotion_by_id(emotion_id):
        return MonthlyEmotions.get_by_id(emotion_id)

    @staticmethod
    def get_monthly_emotions():
        emotions_list = MonthlyEmotions.objects.all().order_by('-timestamp')
        emotions_dict_list = [{emotions.to_dict()} for emotions in emotions_list]
        return {
            "type" : "monthly_emotions_list",
            "list" : emotions_dict_list
        }

    @staticmethod
    def update_emotion(emotion_id, **kwargs):
        emotion = MonthlyEmotions.get_by_id(emotion_id)
        if not emotion:
            return None
        
        # Update fields if present in kwargs
        for field in ['timestamp', 'camera_id', 'negative', 'surprise', 'happy', 'neutral']:
            if field in kwargs:
                setattr(emotion, field, kwargs[field])
        
        emotion.save()
        return emotion

    @staticmethod
    def delete_emotion(emotion_id):
        emotion = MonthlyEmotions.get_by_id(emotion_id)
        if emotion:
            emotion.delete()
            return True
        return False
    @staticmethod
    def _calculate_average_emotions(emotions):
        if not emotions:
            return None

        # Calculate the average values of each emotion using mean
        return {
            "negative": mean([emotion.negative for emotion in emotions]),
            "surprise": mean([emotion.surprise for emotion in emotions]),
            "happy": mean([emotion.happy for emotion in emotions]),
            "neutral": mean([emotion.neutral for emotion in emotions])
        }
    #Lấy tỉ lệ cảm xúc trong k giờ gần nhất 
    @staticmethod
    def get_average_emotions_in_last_hours(hours):
        now = datetime.now()
        start_time = now - timedelta(hours=hours)
        emotions = MonthlyEmotions.get_emotions_by_time_range(start_time, now)
        return MonthlyEmotionsService._calculate_average_emotions(emotions)
    
    #Lấy tỉ lệ cảm xúc trong k giờ gần nhất 
    @staticmethod
    def get_camera_emotions_in_hours(hours, camera_id, end_time= None):
        now = datetime.now()
        if end_time is not None:
            now = end_time
        start_time = now - timedelta(hours=hours)
        emotions = MonthlyEmotions.get_camera_emotions_by_time_range(start_time, now, camera_id)
        emotion_dict_list = [emotion.to_dict_with_ratios() for emotion in emotions]
        return emotion_dict_list
    
    @staticmethod
    def get_camera_list_emotions_in_hours(hours, server_id, end_time= None):
        now = datetime.now()
        if end_time is not None:
            now = end_time
        start_time = now - timedelta(hours=hours)
        emotions = MonthlyEmotions.get_camera_list_emotions_by_time_range(start_time, now, server_id)
        emotion_dict_list = [emotion.to_dict_with_ratios() for emotion in emotions]
        return emotion_dict_list
    