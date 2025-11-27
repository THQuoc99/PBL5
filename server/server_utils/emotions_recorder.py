import time
from datetime import datetime, timedelta
import threading
import copy

from services.DailyEmotionsService import DailyEmotionsService
from services.MonthlyEmotionsService import MonthlyEmotionsService
from services.EmotionsService import EmotionsService
from .socket_utils import SocketHelper

from collections import defaultdict


class EmotionsRecorder:
    def __init__(self,video_server, server_id):
        self.video_server = video_server
        self.observer_conn = None
        self.server_id = server_id

        # Thêm biến lưu trữ thời gian
        self.last_stats_time = time.time()
        self.last_stats_minute = datetime.now().replace(second=0, microsecond=0)
        self.last_stats_hour = datetime.now().replace(minute=0, second=0, microsecond=0)
        self.last_stats_day = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

        # Thêm biến đếm số lần cảnh báo liên tiếp
        self.negative_alert = defaultdict(int)
        self.neutral_alert = defaultdict(int)
        self.unhappy_alert = defaultdict(int)

        self._stop_event = threading.Event()
    
    def set_observer_conn(self,conn):
        self.observer_conn = conn
    
    def remove_observer_conn(self):
        self.observer_conn = None
        
    def send_response(self, type, data=None, presenter= 'dashboard_presenter'):
        if self.observer_conn is not None:
            try:
                if (data) :
                    response_packet = {
                        'type': type,
                        'data': data,
                        'presenter' : presenter,
                        'status' : 'success'
                    }
                    SocketHelper.send_data(self.observer_conn, response_packet)
            except Exception as e:
                print(f"Error sending response to {self.observer_conn}: {e}")
    
    def send_alert(self,type, camera):
        if type == 'negative':
            self.negative_alert[camera] += 1
            self.neutral_alert[camera] = 0
            self.unhappy_alert[camera] = 0
            count = self.negative_alert[camera]
        elif type == 'neutral':
            self.negative_alert[camera] = 0
            self.neutral_alert[camera] += 1
            self.unhappy_alert[camera] = 0
            count = self.neutral_alert[camera]
        elif type == 'unhappy':
            self.negative_alert[camera] = 0
            self.neutral_alert[camera] = 0
            self.unhappy_alert[camera] += 1
            count = self.unhappy_alert[camera]
        else:
            return  # nếu type không hợp lệ, không làm gì cả

        # Chỉ gửi thông báo nếu đạt ngưỡng 3 lần liên tiếp
        if count >= 2 and self.observer_conn is not None:
            try:
                response_packet = {
                    'type': type,
                    'presenter': 'alert_presenter',
                    'camera': camera,
                }
                SocketHelper.send_data(self.observer_conn, response_packet)

                # Reset sau khi gửi để tránh spam
                self.negative_alert[camera] = 0
                self.neutral_alert[camera] = 0
                self.unhappy_alert[camera] = 0

            except Exception as e:
                print(f"Error sending response to {self.observer_conn}: {e}")

    def send_stats(self):
        current_time = time.time()
        elapsed = current_time - self.last_stats_time
        
        if elapsed >= 10:  # Mỗi 10s in một lần
            emotion_counts_per_10s = self.video_server.emotion_counts_per_10s.copy()
            self.video_server.reset_counter_per_10s()
            # Tính tổng số cảm xúc đã nhận diện
            total_emotions = sum(emotion_counts_per_10s.values())
            
            # Tính tỉ lệ phần trăm cho mỗi cảm xúc
            emotion_ratios = defaultdict(float)
            for emotion, count in emotion_counts_per_10s.items():
                if total_emotions > 0:
                    ratio = (count / total_emotions) 
                else:
                    ratio = 0
                emotion_ratios[emotion] = ratio
            
            if self.observer_conn is not None:
                try:
                    if (emotion_ratios) :
                        emotion_ratios_packet = {
                            'type': 'total_emotion_ratios',
                            'emotion_ratios': emotion_ratios,
                            'presenter' : 'home_presenter',
                            'status' : 'success'
                        }
                        SocketHelper.send_data(self.observer_conn, emotion_ratios_packet)
                except Exception as e:
                    print(f"Error sending emotion ratios to {self.observer_conn}: {e}")

            self.last_stats_time = current_time
    
    def handle_save_stats_per_minute(self):
        current_time = datetime.now()
        last_minute = self.last_stats_minute
        if current_time >= last_minute + timedelta(minutes=1):
            camera_list = copy.deepcopy(self.video_server.cameras)
            self.video_server.reset_counter_per_minute()

            # Lưu bản ghi theo phút
            self.save_stats(camera_list)

            # Lưu bản ghi theo giờ
            self.save_stats_per_hour(camera_list)
            last_hour = self.last_stats_hour
            if current_time >= last_hour + timedelta(hours=1):
                self.last_stats_hour = current_time.replace(minute=0, second=0, microsecond=0)
            
            # Lưu bản ghi theo ngày
            self.save_stats_per_day(camera_list)
            last_day = self.last_stats_day
            if current_time >= last_day + timedelta(days=1):
                self.last_stats_day = current_time.replace(hour=0, minute=0, second=0, microsecond=0)

            self.last_stats_minute = current_time.replace(second=0, microsecond=0)

    def save_stats(self,camera_list):  # Mỗi phút in một lần
        emotion_counts_all = defaultdict(int)
        emotion_ratios = defaultdict(float)
        heatmap_data = defaultdict(dict)
        current_time = self.last_stats_minute
        
        for camera_id , camera in camera_list.items():
            total_emotions = sum(camera["emotion_counts"].values())
            for emotion, count in camera["emotion_counts"].items():
                if total_emotions > 0:
                    emotion_counts_all[emotion] += count
                    ratio = (count / total_emotions) 
                else:
                    ratio = 0
                emotion_ratios[emotion] = ratio

            heatmap_data[camera['camera_name']] = emotion_ratios.copy()
            
            try:
                emotion_record = DailyEmotionsService.create_emotion(
                    timestamp= current_time,
                    camera_id = camera_id,  # đảm bảo self.camera_id đã được định nghĩa
                    negative=camera["emotion_counts"].get('Negative', 0),
                    surprise=camera["emotion_counts"].get('Surprise', 0),
                    happy=camera["emotion_counts"].get('Happy', 0),
                    neutral=camera["emotion_counts"].get('Neutral', 0)
                )
                # Gợi ý cảnh báo cảm xúc
                if emotion_ratios.get('Negative', 0) >= 0.6:
                    self.send_alert('negative', camera['camera_name'])

                elif emotion_ratios.get('Neutral', 0) >= 0.9:
                    self.send_alert('neutral', camera['camera_name'])

                elif emotion_ratios.get('Happy', 0) <= 0.2:
                    self.send_alert('unhappy', camera['camera_name'])
                
                if camera_id == self.video_server.target_camera:
                    print("[DEBUG] Gửi dữ liệu tới observer")
                    self.send_response('emotion_per_minute', emotion_record.to_dict_with_ratios(), 'observe_presenter')

                print(f"[DATABASE] Emotion record saved to DB (1m): {emotion_record.id}")
            except Exception as e:
                print(f"Error saving emotion record to DB (1m): {e}")

        try:
            heatmap_data_packet = {
                'timestamp' : current_time,
                'heatmap_data' : heatmap_data,
            }
            self.send_response('heatmap_data_per_minute', heatmap_data_packet)
            emotion_record = DailyEmotionsService.create_emotion(
                timestamp= current_time,
                camera_id = self.server_id,  # đảm bảo self.camera_id đã được định nghĩa
                negative=emotion_counts_all.get('Negative', 0),
                surprise=emotion_counts_all.get('Surprise', 0),
                happy=emotion_counts_all.get('Happy', 0),
                neutral=emotion_counts_all.get('Neutral', 0)
            )
            self.send_response('emotion_per_minute', emotion_record.to_dict_with_ratios())
            print(f"[DATABASE] Emotion record saved to DB (1m): {emotion_record.id}")
        except Exception as e:
            print(f"Error saving emotion record to DB (1m): {e}")

    def save_stats_per_hour(self, camera_list):
        current_time = self.last_stats_hour
        emotion_counts_all = defaultdict(int)
        emotion_ratios = defaultdict(float)
        heatmap_data = defaultdict(dict)

        for camera_id , camera in camera_list.items():
            total_emotions = sum(camera["emotion_counts"].values())
            for emotion, count in camera["emotion_counts"].items():
                if total_emotions > 0:
                    emotion_counts_all[emotion] += count
                    ratio = (count / total_emotions) 
                else:
                    ratio = 0
                emotion_ratios[emotion] = ratio

            heatmap_data[camera['camera_name']] = emotion_ratios.copy()
            try:
                # Tạo bản ghi mới nếu không có
                emotion_record = MonthlyEmotionsService.save_emotion(
                    timestamp=current_time,
                    camera_id= camera_id,
                    negative=camera["emotion_counts"].get('Negative', 0),
                    surprise=camera["emotion_counts"].get('Surprise', 0),
                    happy=camera["emotion_counts"].get('Happy', 0),
                    neutral=camera["emotion_counts"].get('Neutral', 0)
                )
                print(f"[DATABASE] Emotion record saved in DB (1H): {emotion_record.id}")
            except Exception as e:
                print(f"Error updating/saving emotion record to DB (1H): {e}")

        try:
            # Tạo bản ghi mới nếu không có
            heatmap_data_packet = {
                'timestamp' : current_time,
                'heatmap_data' : heatmap_data,
            }
            self.send_response('heatmap_data_per_hour', heatmap_data_packet)
            emotion_record = MonthlyEmotionsService.save_emotion(
                timestamp=current_time,
                camera_id= self.server_id,
                negative=emotion_counts_all.get('Negative', 0),
                surprise=emotion_counts_all.get('Surprise', 0),
                happy=emotion_counts_all.get('Happy', 0),
                neutral=emotion_counts_all.get('Neutral', 0)
            )
            self.send_response('emotion_per_hour', emotion_record.to_dict_with_ratios())
            print(f"[DATABASE] Emotion record created in DB (1H): {emotion_record.id}")
        except Exception as e:
            print(f"Error updating/saving emotion record to DB (1H): {e}")
    
    def save_stats_per_day(self, camera_list):
        current_time = self.last_stats_day
        emotion_counts_all = defaultdict(int)
        emotion_ratios = defaultdict(float)
        heatmap_data = defaultdict(dict)

        for camera_id , camera in camera_list.items():
            total_emotions = sum(camera["emotion_counts"].values())
            for emotion, count in camera["emotion_counts"].items():
                if total_emotions > 0:
                    emotion_counts_all[emotion] += count
                    ratio = (count / total_emotions) 
                else:
                    ratio = 0
                emotion_ratios[emotion] = ratio

            heatmap_data[camera['camera_name']] = emotion_ratios.copy()
            try:
                # Tạo bản ghi mới nếu không có
                emotion_record = EmotionsService.save_emotion(
                    timestamp=current_time,
                    camera_id= camera_id,
                    negative=camera["emotion_counts"].get('Negative', 0),
                    surprise=camera["emotion_counts"].get('Surprise', 0),
                    happy=camera["emotion_counts"].get('Happy', 0),
                    neutral=camera["emotion_counts"].get('Neutral', 0)
                )
                print(f"[DATABASE] Emotion record saved in DB (1H): {emotion_record.id}")
            except Exception as e:
                print(f"Error updating/saving emotion record to DB (1H): {e}")

        try:
            # Tạo bản ghi mới nếu không có
            heatmap_data_packet = {
                'timestamp' : current_time,
                'heatmap_data' : heatmap_data,
            }
            self.send_response('heatmap_data_per_day', heatmap_data_packet)
            emotion_record = EmotionsService.save_emotion(
                timestamp=current_time,
                camera_id= self.server_id,
                negative=emotion_counts_all.get('Negative', 0),
                surprise=emotion_counts_all.get('Surprise', 0),
                happy=emotion_counts_all.get('Happy', 0),
                neutral=emotion_counts_all.get('Neutral', 0)
            )
            self.send_response('emotion_per_day', emotion_record.to_dict_with_ratios())
            print(f"[DATABASE] Emotion record saved in DB (1D): {emotion_record.id}")
        except Exception as e:
            print(f"Error updating/saving emotion record to DB (1D): {e}")
    
    
    def start(self):
        print(f"{'='*32}")
        print(f"[SYSTEM] Recorder is running.")
        print(f"{'='*32}")

        while not self._stop_event.is_set():
            self.send_stats()
            self.handle_save_stats_per_minute()
            time.sleep(1)

    def stop(self):
        self._stop_event.set()
        print(f"{'='*32}")
        print(f"[SYSTEM] Recorder is stopped.")
        print(f"{'='*32}")
