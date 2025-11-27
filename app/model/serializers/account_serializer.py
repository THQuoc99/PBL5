class AccountSerializer:
    def __init__(self, account_id, username, password, email, role, created_at=None):
        self.account_id = account_id
        self.username = username
        self.password = password
        self.email = email
        self.role = role
        self.created_at = created_at

    def to_dict(self):
        return {
            "account_id": self.account_id,
            "username": self.username,
            "password": self.password,
            "email": self.email,
            "role": self.role,
            "created_at": self.created_at
        }

    @staticmethod
    def from_dict(data):
        return AccountSerializer(**data)