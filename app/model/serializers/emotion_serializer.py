class EmotionsSerializer:
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

    @staticmethod
    def from_dict(data):
        return EmotionsSerializer(**data)