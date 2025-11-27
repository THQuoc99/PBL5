from .enums import CommandType
from services.AccountService import AccountService
from services.DailyEmotionsService import DailyEmotionsService
from services.MonthlyEmotionsService import MonthlyEmotionsService
from services.EmotionsService import EmotionsService
from services.CameraService import CameraService
from collections import defaultdict
from datetime import datetime, timedelta





class CommandHandler:
    def __init__(self, video_server, emotions_recorder):
        self.video_server = video_server
        self.emotions_recorder = emotions_recorder
        self.commands = {
            CommandType.CONNECT: self.handle_connect,
            CommandType.DISCONNECT: self.handle_disconnect,
            CommandType.LOGIN: self.handle_login,
            CommandType.GET_CAMERA_LIST: self.handle_get_camera_list,
            CommandType.START_WATCH: self.handle_start_watch,
            CommandType.STOP_WATCH: self.handle_stop_watch,
            CommandType.GET_EMOTIONS_IN_LAST_MINUTES: self.handle_get_emotions_in_last_minutes,
            CommandType.GET_CAMERA_EMOTIONS_IN_LAST_MINUTES: self.handle_get_camera_emotions_in_last_minutes,
            CommandType.GET_EMOTIONS_IN_LAST_HOURS: self.handle_get_emotions_in_last_hours,
            CommandType.GET_EMOTIONS_IN_LAST_DAYS: self.handle_get_emotions_in_last_days,
            CommandType.GET_EMOTIONS_CUSTOM: self.handle_get_emotions_custom,
            CommandType.GET_HEATMAP_DATA_IN_LAST_MINUTES: self.handle_get_heatmap_data_in_last_minutes,
            CommandType.GET_HEATMAP_DATA_IN_LAST_HOURS: self.handle_get_heatmap_data_in_last_hours,
            CommandType.GET_HEATMAP_DATA_IN_LAST_DAYS: self.handle_get_heatmap_data_in_last_days,
            CommandType.GET_HEATMAP_DATA_CUSTOM: self.handle_get_heatmap_data_custom,
        }
        self.server_id = CameraService.get_server_id()
    
    @staticmethod
    def get_server_id():
        return CameraService.get_server_id()

    def handle(self, command, conn, addr):
        command_type = CommandType(command['type'])

        handler = self.commands.get(command_type)
        if handler:
            response = handler(command, conn, addr)
            response["presenter"] = command["presenter"]
            return response
        return {"type": "error", "message": "Unknown command"}
    
    def handle_login(self, command, conn, addr):
        username = command['username']
        password = command['password']
        account = AccountService.login(username, password)
        return {
            "type": "account",
            "status" : "success" if account else "fail", 
            "account": account,}

    def handle_start_watch(self, command, conn, addr):
        status = self.video_server.update_target_ip(command)
        return {
            "type" : "start_watch",
            "status" : status
        }

    def handle_stop_watch(self, command, conn, addr):
        status = self.video_server.remove_target_ip(command)
        return {
            "type" : "stop_watch",
            "status" : status
        }

    
    def handle_connect(self,command, conn, addr):
        if command["client_type"] == "camera":
            camera_id = CameraService.get_or_create(command['camera_name'],addr[0])
            command['camera_id'] = camera_id
        
        status = self.video_server.register_client(command)
        response = {
            "type" : "connect",
            "status" : status
        }
        if status == 'success':
            if command["client_type"] == 'observer':
                self.emotions_recorder.set_observer_conn(conn)
            else:
                response['camera_id'] = camera_id

        return response
    
    def handle_disconnect(self, command, conn, addr):
        status = self.video_server.unregister_client(command)
        if command["client_type"] == 'observer' and status == 'success':
            self.emotions_recorder.remove_observer_conn()
        return {
            "type" : "disconnect",
            "status" : status
        }
    
    def handle_get_camera_list(self,command,conn, addr):
        raw_camera_list = list(self.video_server.cameras.keys())
        if not raw_camera_list:
            return {
                "type": "camera_list",
                "status": "fail",
            }
        camera_list = [CameraService.get_camera_by_id(camera) for camera in raw_camera_list]
        return {
            "type": "camera_list", 
            "list": camera_list,
            "status" : "success"
            }
    
    def handle_get_emotions_in_last_minutes(self, command, conn, addr):
        
        minutes = command['minutes']
        
        emotions = DailyEmotionsService.get_camera_emotions_in_minutes(minutes, self.server_id)
        # Tổng hợp trung bình cho mỗi cảm xúc tại mỗi mốc thời gian
        result = {
            "happy": defaultdict(float),
            "neutral": defaultdict(float),
            "negative": defaultdict(float),
            "surprise": defaultdict(float)
        }

        for e in emotions:
            time_key = e["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
            result["happy"][time_key] = e["happy"]
            result["neutral"][time_key] = e["neutral"]
            result["negative"][time_key] = e["negative"]
            result["surprise"][time_key] = e["surprise"]

        return {
            'type' : 'emotions_in_last_minutes',
            'emotions_dict': result,
            'status' : 'success'
        }
    
    def handle_get_camera_emotions_in_last_minutes(self, command, conn, addr):
        minutes = command['minutes']
        camera_id = command['camera_id']
        
        emotions = DailyEmotionsService.get_camera_emotions_in_minutes(minutes, camera_id)
        # Tổng hợp trung bình cho mỗi cảm xúc tại mỗi mốc thời gian
        result = {
            "happy": defaultdict(float),
            "neutral": defaultdict(float),
            "negative": defaultdict(float),
            "surprise": defaultdict(float)
        }

        for e in emotions:
            time_key = e["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
            result["happy"][time_key] = e["happy"]
            result["neutral"][time_key] = e["neutral"]
            result["negative"][time_key] = e["negative"]
            result["surprise"][time_key] = e["surprise"]

        return {
            'type' : 'emotions_in_last_minutes',
            'emotions_dict': result,
            'status' : 'success'
        }
    
    def handle_get_emotions_in_last_hours(self, command, conn, addr):
        hours = command['hours']
        emotions = MonthlyEmotionsService.get_camera_emotions_in_hours(hours, self.server_id)

        # Tổng hợp trung bình cho mỗi cảm xúc tại mỗi mốc thời gian
        result = {
            "happy": defaultdict(float),
            "neutral": defaultdict(float),
            "negative": defaultdict(float),
            "surprise": defaultdict(float)
        }

        for e in emotions:
            time_key = e["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
            result["happy"][time_key] = e["happy"]
            result["neutral"][time_key] = e["neutral"]
            result["negative"][time_key] = e["negative"]
            result["surprise"][time_key] = e["surprise"]

        return {
            'type' : 'emotions_in_last_hours',
            'emotions_dict': result,
            'status' : 'success'
        }
    
    def handle_get_emotions_in_last_days(self, command, conn, addr):
        days = command['days']
        emotions = EmotionsService.get_camera_emotions_in_days(days, self.server_id)

        # Tổng hợp trung bình cho mỗi cảm xúc tại mỗi mốc thời gian
        result = {
            "happy": defaultdict(float),
            "neutral": defaultdict(float),
            "negative": defaultdict(float),
            "surprise": defaultdict(float)
        }

        for e in emotions:
            time_key = e["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
            result["happy"][time_key] = e["happy"]
            result["neutral"][time_key] = e["neutral"]
            result["negative"][time_key] = e["negative"]
            result["surprise"][time_key] = e["surprise"]

        return {
            'type' : 'emotions_in_last_days',
            'emotions_dict': result,
            'status' : 'success'
        }

    def handle_get_emotions_custom(self, command, conn, addr):
        end_time = datetime.strptime(command["end_time"], "%Y-%m-%d %H:%M:%S")
        time_range = end_time - datetime.strptime(command["start_time"], "%Y-%m-%d %H:%M:%S")
        time_unit = command['time_unit']
        if time_unit == "minute":
            minutes = int(time_range.total_seconds() / 60)
            emotions = DailyEmotionsService.get_camera_emotions_in_minutes(minutes, self.server_id, end_time)
        elif time_unit == "hour":
            hours = int(time_range.total_seconds() / 3600)
            emotions = MonthlyEmotionsService.get_camera_emotions_in_hours(hours, self.server_id, end_time)
        elif time_unit == "day":
            days = time_range.days
            emotions = EmotionsService.get_camera_emotions_in_days(days, self.server_id, end_time)
        else:
            return {
                'type': 'emotions_custom',
                'status': 'error',
                'message': 'Invalid time unit'
            }
        

        # Tổng hợp trung bình cho mỗi cảm xúc tại mỗi mốc thời gian
        result = {
            "happy": defaultdict(float),
            "neutral": defaultdict(float),
            "negative": defaultdict(float),
            "surprise": defaultdict(float)
        }

        for e in emotions:
            time_key = e["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
            result["happy"][time_key] = e["happy"]
            result["neutral"][time_key] = e["neutral"]
            result["negative"][time_key] = e["negative"]
            result["surprise"][time_key] = e["surprise"]

        return {
            'type' : 'emotions_custom',
            'emotions_dict': result,
            'status' : 'success'
        }

    def handle_get_heatmap_data_in_last_minutes(self,command ,conn, addr):
        
        minutes = command['minutes']
        
        emotions = DailyEmotionsService.get_camera_list_emotions_in_minutes(minutes, self.server_id)
        # Tổng hợp trung bình cho mỗi cảm xúc tại mỗi mốc thời gian
        result = {
            "happy": defaultdict(dict),
            "neutral": defaultdict(dict),
            "negative": defaultdict(dict),
            "surprise": defaultdict(dict)
        }

        camera_list = defaultdict(dict)

        for e in emotions:
            if e['camera_id'] not in camera_list:
                camera = CameraService.get_camera_by_id(e["camera_id"])
                camera_list[e['camera_id']] = camera
            time_key = e["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
            camera_name = camera_list[e['camera_id']]['name']
            result["happy"][time_key][camera_name] = e["happy"]
            result["neutral"][time_key][camera_name] = e["neutral"]
            result["negative"][time_key][camera_name] = e["negative"]
            result["surprise"][time_key][camera_name] = e["surprise"]

        return {
            'type' : 'heatmap_data_in_last_minutes',
            'heatmap_data': result,
            'status' : 'success'
        }
    
    def handle_get_heatmap_data_in_last_hours(self,command ,conn, addr):
        
        hours = command['hours']
        
        emotions = MonthlyEmotionsService.get_camera_list_emotions_in_hours(hours, self.server_id)
        # Tổng hợp trung bình cho mỗi cảm xúc tại mỗi mốc thời gian
        result = {
            "happy": defaultdict(dict),
            "neutral": defaultdict(dict),
            "negative": defaultdict(dict),
            "surprise": defaultdict(dict)
        }

        camera_list = defaultdict(dict)

        for e in emotions:
            if e['camera_id'] not in camera_list:
                camera = CameraService.get_camera_by_id(e["camera_id"])
                camera_list[e['camera_id']] = camera
            time_key = e["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
            camera_name = camera_list[e['camera_id']]['name']
            result["happy"][time_key][camera_name] = e["happy"]
            result["neutral"][time_key][camera_name] = e["neutral"]
            result["negative"][time_key][camera_name] = e["negative"]
            result["surprise"][time_key][camera_name] = e["surprise"]

        return {
            'type' : 'heatmap_data_in_last_hours',
            'heatmap_data': result,
            'status' : 'success'
        }
    
    def handle_get_heatmap_data_in_last_days(self,command ,conn, addr):
        
        days = command['days']
        
        emotions = EmotionsService.get_camera_list_emotions_in_days(days, self.server_id)
        # Tổng hợp trung bình cho mỗi cảm xúc tại mỗi mốc thời gian
        result = {
            "happy": defaultdict(dict),
            "neutral": defaultdict(dict),
            "negative": defaultdict(dict),
            "surprise": defaultdict(dict)
        }

        camera_list = defaultdict(dict)

        for e in emotions:
            if e['camera_id'] not in camera_list:
                camera = CameraService.get_camera_by_id(e["camera_id"])
                camera_list[e['camera_id']] = camera
            time_key = e["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
            camera_name = camera_list[e['camera_id']]['name']
            result["happy"][time_key][camera_name] = e["happy"]
            result["neutral"][time_key][camera_name] = e["neutral"]
            result["negative"][time_key][camera_name] = e["negative"]
            result["surprise"][time_key][camera_name] = e["surprise"]

        return {
            'type' : 'heatmap_data_in_last_days',
            'heatmap_data': result,
            'status' : 'success'
        }

    def handle_get_heatmap_data_custom(self,command ,conn, addr):
        end_time = datetime.strptime(command["end_time"], "%Y-%m-%d %H:%M:%S")
        time_range = end_time - datetime.strptime(command["start_time"], "%Y-%m-%d %H:%M:%S")
        time_unit = command['time_unit']
        if time_unit == "minute":
            minutes = int(time_range.total_seconds() / 60)
            emotions = DailyEmotionsService.get_camera_list_emotions_in_minutes(minutes, self.server_id, end_time)
        elif time_unit == "hour":
            hours = int(time_range.total_seconds() / 3600)
            emotions = MonthlyEmotionsService.get_camera_list_emotions_in_hours(hours, self.server_id, end_time)
        elif time_unit == "day":
            days = time_range.days
            emotions = EmotionsService.get_camera_list_emotions_in_days(days, self.server_id, end_time)
        else:
            return {
                'type': 'heatmap_data_custom',
                'status': 'error',
                'message': 'Invalid time unit'
            }
        
        result = {
            "happy": defaultdict(dict),
            "neutral": defaultdict(dict),
            "negative": defaultdict(dict),
            "surprise": defaultdict(dict)
        }

        camera_list = defaultdict(dict)

        for e in emotions:
            if e['camera_id'] not in camera_list:
                camera = CameraService.get_camera_by_id(e["camera_id"])
                camera_list[e['camera_id']] = camera
            time_key = e["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
            camera_name = camera_list[e['camera_id']]['name']
            result["happy"][time_key][camera_name] = e["happy"]
            result["neutral"][time_key][camera_name] = e["neutral"]
            result["negative"][time_key][camera_name] = e["negative"]
            result["surprise"][time_key][camera_name] = e["surprise"]

        return {
            'type' : 'heatmap_data_custom',
            'heatmap_data': result,
            'status' : 'success'
        }
