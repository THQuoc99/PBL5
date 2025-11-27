import sys
import os
import uuid

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from db.db_helper import DBHelper

class Account:
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
        return Account(**data)

    @classmethod
    def get_all_accounts(cls):
        db_helper = DBHelper()
        query = "SELECT * FROM account"
        result = db_helper.fetch_all(query)
        return [cls(**account) for account in result]

    @classmethod
    def get_account_by_id(cls, account_id):
        db_helper = DBHelper()
        query = "SELECT * FROM account WHERE account_id = %s"
        result = db_helper.fetch_one(query, (account_id,))
        if result:
            return cls(**result)
        return None
    @staticmethod
    def check_account(username=None, email=None):
        query = """
        SELECT * FROM account WHERE username = %s OR email = %s
        """
        result = DBHelper.fetch_one(query, (username, email))
        if result:
            return Account(account_id=result['account_id'], username=result['username'], password=result['password'], email=result['email'], role=result['role'])
        return None
    
    @staticmethod
    def login(username=None, password=None):
        db_helper = DBHelper()
        query = """
        SELECT * FROM account WHERE username = %s AND password = %s
        """
        result = db_helper.fetch_one(query, (username, password))
        if result:
            return Account(account_id=result['account_id'], username=result['username'], password=result['password'], email=result['email'], role=result['role'])
        return None


    def save(self):
        db_helper = DBHelper()
        if self.account_id:
            query = """UPDATE account 
                       SET username = %s, password = %s, email = %s, role = %s 
                       WHERE account_id = %s"""
            db_helper.execute(query, (self.username, self.password, self.email, self.role, self.account_id))
        else:
            self.account_id = str(uuid.uuid4())
            query = """INSERT INTO account (account_id, username, password, email, role) 
                       VALUES (%s, %s, %s, %s, %s)"""
            db_helper.execute(query, (self.account_id, self.username, self.password, self.email, self.role))

    def delete(self):
        db_helper = DBHelper()
        query = "DELETE FROM account WHERE account_id = %s"
        db_helper.execute(query, (self.account_id,))
