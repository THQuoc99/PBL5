import pickle

class SocketHelper:
    @staticmethod
    def recv_n_bytes(conn, n):
        data = bytearray()
        while len(data) < n:
            packet = conn.recv(n - len(data))
            if not packet:
                return None
            data.extend(packet)
        return data
    
    @staticmethod
    def recv_data(conn):
        """Nhận data đầy đủ và giải mã pickle nếu có."""
        try:
            raw_len = SocketHelper.recv_n_bytes(conn, 4)
            if not raw_len:
                return None
            data_len = int.from_bytes(raw_len, 'big')
            raw_data = SocketHelper.recv_n_bytes(conn, data_len)
            if raw_data is None:
                return None
            try:
                return pickle.loads(raw_data)
            except pickle.PickleError:
                return raw_data.decode()  # fallback nếu là string thuần
        except Exception as e:
            print(f"[ERROR] Recv error: {e}")
            return None

    @staticmethod
    def send_data(conn, data):
        try:
            if isinstance(data, str):
                data = data.encode()
            elif isinstance(data, dict) or hasattr(data, '__dict__'):
                data = pickle.dumps(data)

            conn.sendall(len(data).to_bytes(4, 'big'))
            conn.sendall(data)
        except Exception as e:
            print(f"[ERROR] Send error: {e}")