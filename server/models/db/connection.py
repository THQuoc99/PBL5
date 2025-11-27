import mysql.connector
from mysql.connector import Error

class DatabaseConnection:
    def __init__(self):
        self.connection = None
        self.connect()

    def connect(self):
        try:
            self.connection = mysql.connector.connect(
                host="localhost",
                user="root",  # Thay bằng username MySQL của bạn
                password="",  # Thay bằng password MySQL của bạn
                database="sentio_db"
            )
            if self.connection.is_connected():
                print("✅ Kết nối thành công")
        except Error as e:
            print(f"❌ Lỗi kết nối: {e}")

    def get_connection(self):
        """Trả về kết nối hiện tại nếu đang kết nối, hoặc kết nối lại nếu mất kết nối."""
        if not self.connection or not self.connection.is_connected():
            self.connect()
        return self.connection

    def close(self):
        """Đóng kết nối cơ sở dữ liệu."""
        if self.connection and self.connection.is_connected():
            self.connection.close()
            print("🛑 Đã đóng kết nối")
