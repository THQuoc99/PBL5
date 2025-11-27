from db.connection import DatabaseConnection

class DBHelper:
    def __init__(self):
        self.db = DatabaseConnection()

    def fetch_all(self, query, params=None):
        """Truy vấn tất cả các dòng dữ liệu từ DB"""
        conn = self.db.get_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute(query, params or ())
        result = cursor.fetchall()
        cursor.close()
        return result

    def fetch_one(self, query, params=None):
        """Truy vấn một dòng dữ liệu"""
        conn = self.db.get_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute(query, params or ())
        result = cursor.fetchone()
        cursor.close()
        return result

    def execute(self, query, params=None):
        """Thực thi các câu lệnh như INSERT, UPDATE, DELETE"""
        conn = self.db.get_connection()
        cursor = conn.cursor()
        cursor.execute(query, params or ())
        conn.commit()
        cursor.close()
