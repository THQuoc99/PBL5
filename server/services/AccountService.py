# services/account_service.py
from models.objects.account import Account

class AccountService:
    @staticmethod
    def get_account_by_id(account_id):
        # Truy vấn tài khoản từ cơ sở dữ liệu
        return Account.get_account_by_id(account_id)

    @staticmethod
    def delete_account(account_id):
        # Xóa tài khoản khỏi cơ sở dữ liệu
        account = Account.get_account_by_id(account_id)
        account.delete()
    
    @staticmethod 
    def login(username, password):
        existing_account = Account.login(username, password)
        if existing_account:
            return existing_account.to_dict()
        else:
            return None
