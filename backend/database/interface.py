import json
import psycopg2
from datetime import datetime
import os
import shutil
import torch
import sys
import hashlib


class UserDatabase():
    def __init__(self):
        self.conn = psycopg2.connect(host="localhost",
                                     database="ml_builder",
                                     user="postgres",
                                     password="1234",
                                     port=5432)
        self.cur = self.conn.cursor()
        self.cur.execute("""CREATE TABLE IF NOT EXISTS users (
            uuid VARCHAR(255) PRIMARY KEY,
            username VARCHAR(255)
        );
        """)

        self.cur.execute("""CREATE TABLE IF NOT EXISTS models (
            uuid INT PRIMARY KEY,
            user_uuid VARCHAR(255) REFERENCES users (uuid),
            model_dir VARCHAR(255),
            created_at TIMESTAMP
        );
        """)

        self.cur.execute("""CREATE TABLE IF NOT EXISTS datasets (
            uuid INT PRIMARY KEY,
            user_uuid VARCHAR(255) REFERENCES users (uuid),
            model_uuid INT REFERENCES models (uuid),
            dataset_name VARCHAR(255),
            created_at TIMESTAMP
        );
        """)

        self.conn.commit()
        sys_path = sys.path[0]
        while 'backend' not in os.listdir(sys_path) and 'InteractiveMlBuilder' in sys_path:
            if "InteractiveMlBuilder" not in sys_path:
                raise FileNotFoundError("Not in InteractiveMlBuilder directory?!! Noah I know this is going to be ur fault")
            sys_path = os.path.dirname(sys_path)
            
        self.user_data_root = os.path.join(sys_path, 'backend/database/user_data')
        os.makedirs(self.user_data_root, exist_ok=True)

    def hash(self, data_str: str):
        hashed = hashlib.md5(data_str.encode()).hexdigest()
        return abs(int(hashed, 16) % 1000000)
        # return abs(hash(data_str) % 1000000)

    def close(self):
        self.cur.close()
        self.conn.close()

    def add_user(self, uuid: str, username: str):
        # make sure uuid is unique
        if self.get_user_name(uuid):
            return False
        # hash the username as the unique id int
        self.cur.execute("INSERT INTO users (uuid, username) VALUES (%s, %s)", (uuid, username))
        self.conn.commit()
        return True

    # def add_user(self, username):
    #     # make sure username is unique
    #     if self.get_user_uuid(username):
    #         return False
    #     # hash the username as the unique id int
    #     user_uuid = self.hash(username)
    #     self.cur.execute("INSERT INTO users (uuid, username) VALUES (%s, %s)", (user_uuid, username))
    #     self.conn.commit()
    #     return True
    #
    # def add_user_uuid(self, user_uuid):
    #     if self.get_user_name(user_uuid):
    #         return False
    #     self.cur.execute("INSERT INTO users (uuid, username) VALUES (%s, %s)", (user_uuid, user_uuid))
    #     self.conn.commit()
    #     return True

    # def get_user_uuid(self, username):
    #     self.cur.execute("SELECT * FROM users WHERE username=%s", (username,))
    #     user = self.cur.fetchone()
    #     return user[0] if user else None

    def get_user_name(self, user_uuid: str):
        self.cur.execute("SELECT * FROM users WHERE uuid=%s", (user_uuid,))
        user = self.cur.fetchone()
        return user[1] if user else None

    def get_users(self):
        self.cur.execute("SELECT * FROM users")
        return self.cur.fetchall()

    def get_model_uuid(self, user_uuid: str, model_config: dict):
        return self.hash(user_uuid + json.dumps(model_config))

    def init_model(self, user_uuid: str, model_config: dict):
        created_at = datetime.now()
        model_uuid = self.get_model_uuid(user_uuid, model_config)
        model_dir = os.path.join(self.user_data_root, user_uuid, str(model_uuid))

        self.cur.execute("DELETE FROM datasets WHERE user_uuid=%s AND model_uuid=%s", (user_uuid, model_uuid))
        self.cur.execute("DELETE FROM models WHERE user_uuid=%s AND uuid=%s", (user_uuid, model_uuid))
        self.conn.commit()

        # save config
        os.makedirs(model_dir, exist_ok=True)
        with open(os.path.join(model_dir, 'config.json'), 'w') as f:
            f.write(json.dumps(model_config))

        # touching output.logs, loss.logs, error.logs
        open(os.path.join(model_dir, 'output.logs'), 'w').close()
        open(os.path.join(model_dir, 'loss.logs'), 'w').close()
        open(os.path.join(model_dir, 'error.logs'), 'w').close()

        # removing model.pt if it exists
        if os.path.exists(os.path.join(model_dir, 'model.pt')):
            os.remove(os.path.join(model_dir, 'model.pt'))

        self.cur.execute("INSERT INTO models (uuid, user_uuid, model_dir, created_at) VALUES (%s, %s, %s, %s)",
                         (model_uuid, user_uuid, model_dir, created_at))
        self.conn.commit()

        return model_uuid

    def get_model_dir(self, user_uuid: str, model_uuid: int):
        self.cur.execute("SELECT * FROM models WHERE user_uuid=%s AND uuid=%s", (user_uuid, model_uuid))
        model = self.cur.fetchone()
        return model[2] if model else None

    def save_model_pt(self, user_uuid: str, model_uuid: int, model: torch.nn.Module):
        model_dir = self.get_model_dir(user_uuid, model_uuid)
        if not model_dir:
            return False
        torch.save(model.state_dict(), os.path.join(model_dir, 'model.pt'))
        return True

    def save_model_logs(self, user_uuid: str, model_uuid: int, logs: str, log_type: str):
        if log_type not in ['output', 'loss', 'error']:
            raise ValueError('log_type must be one of "output", "loss", "error"')

        model_dir = self.get_model_dir(user_uuid, model_uuid)
        if not model_dir:
            return False
        with open(os.path.join(model_dir, f'{log_type}.logs'), 'a') as f:
            f.write(logs + '\n')
        return True

    def get_models(self, user_uuid: str):
        self.cur.execute("SELECT * FROM models WHERE user_uuid=%s", (user_uuid,))
        return self.cur.fetchall()

    def add_dataset(self, user_uuid: str, model_uuid: int, dataset_name: str):
        # hashing
        dataset_uuid = self.hash(user_uuid + dataset_name) + model_uuid
        created_at = datetime.now()
        self.cur.execute("DELETE FROM datasets WHERE user_uuid=%s AND model_uuid=%s AND dataset_name=%s",
                         (user_uuid, model_uuid, dataset_name))
        self.cur.execute("INSERT INTO datasets (uuid, user_uuid, model_uuid, dataset_name, created_at) VALUES (%s, %s, %s, %s, %s)",
                         (dataset_uuid, user_uuid, model_uuid, dataset_name, created_at))
        self.conn.commit()

    def get_dataset(self, user_uuid: str, model_uuid: int, dataset_name: str):
        self.cur.execute("SELECT * FROM datasets WHERE user_uuid=%s AND model_uuid=%s AND dataset_name=%s",
                         (user_uuid, model_uuid, dataset_name))
        return self.cur.fetchall()

    def get_unique_datasets(self):
        self.cur.execute("SELECT DISTINCT dataset_name FROM datasets")
        return self.cur.fetchall()

    def get_models_by_dataset(self, dataset_name: str):
        self.cur.execute("SELECT user_uuid, model_uuid FROM datasets WHERE dataset_name=%s", (dataset_name,))
        return self.cur.fetchall()

    def get_datasets_by_user(self, user_uuid: str):
        self.cur.execute("SELECT DISTINCT dataset_name FROM datasets WHERE user_uuid=%s", (user_uuid,))
        return self.cur.fetchall()



    def clear(self):
        # DO NOT RUN THIS IN PRODUCTION
        shutil.rmtree(self.user_data_root, ignore_errors=True)
        os.makedirs(self.user_data_root, exist_ok=True)
        self.cur.execute("DELETE FROM datasets")
        self.cur.execute("DELETE FROM models")
        self.cur.execute("DELETE FROM users")
        self.conn.commit()

    def delete(self):
        # DO NOT RUN THIS IN PRODUCTION
        # deletes all tables
        self.clear()
        self.cur.execute("DROP TABLE datasets")
        self.cur.execute("DROP TABLE models")
        self.cur.execute("DROP TABLE users")
        self.conn.commit()
