import psycopg2
import time

class UserDatabase():
    def __init__(self):
        self.conn = psycopg2.connect(host="localhost",
                                     database="postgres",
                                     user="postgres",
                                     password="1234",
                                     port=5432)
        self.cur = self.conn.cursor()
        self.cur.execute("""CREATE TABLE IF NOT EXISTS users (
            uuid INT PRIMARY KEY,
            username VARCHAR(255)
        );
        """)

        self.cur.execute("""CREATE TABLE IF NOT EXISTS models (
            uuid INT PRIMARY KEY,
            user_uuid INT REFERENCES users (uuid),
            model_path VARCHAR(255),
            config_path VARCHAR(255),
            created_at TIMESTAMP
        );
        """)

        self.cur.execute("""CREATE TABLE IF NOT EXISTS datasets (
            uuid INT PRIMARY KEY,
            user_uuid INT REFERENCES users (uuid),
            dataset_path VARCHAR(255),
            created_at TIMESTAMP
        );
        """)

        self.conn.commit()

    def hash(self, string):
        return abs(hash(string) % 1000000)

    def close(self):
        self.cur.close()
        self.conn.close()

    def add_user(self, username):
        # make sure username is unique
        if self.get_user(username):
            return False
        # hash the username as the unique id int
        user_uuid = self.hash(username)
        self.cur.execute("INSERT INTO users (uuid, username) VALUES (%s, %s, %s)", (user_uuid, username))
        self.conn.commit()
        return True

    def get_user(self, username):
        self.cur.execute("SELECT * FROM users WHERE username=%s", (username,))
        return self.cur.fetchone()

    def get_users(self):
        self.cur.execute("SELECT * FROM users")
        return self.cur.fetchall()

    def add_model(self, username, model_path, config_path):
        user = self.get_user(username)
        user_uuid = user[0]
        created_at = time.time()
        self.cur.execute("INSERT INTO models (user_uuid, model_path, config_path, created_at) VALUES (%s, %s, %s, %s)", (user_uuid, model_path, config_path, created_at))
        self.conn.commit()

    def get_models(self, username):
        user = self.get_user(username)
        user_uuid = user[0]
        self.cur.execute("SELECT * FROM models WHERE user_uuid=%s", (user_uuid))
        return self.cur.fetchall()

    def add_dataset(self, username, dataset_name, dataset_path):
        # hashing
        user = self.get_user(username)
        user_uuid = user[0]
        dataset_uuid = self.hash(dataset_name + username)
        created_at = time.time()
        self.cur.execute("INSERT INTO datasets (dataset_uuid, user_uuid, dataset_name, created_at) VALUES (%s, %s, %s)", (dataset_uuid, user_uuid, dataset_path, created_at))
        self.conn.commit()

    def get_dataset(self, username, dataset_name):
        user = self.get_user(username)
        user_uuid = user[0]
        self.cur.execute("SELECT * FROM datasets WHERE user_uuid=%s AND dataset_name=%s", (user_uuid, dataset_name))
        return self.cur.fetchone()

    def clear(self):
        self.cur.execute("DELETE FROM users")
        self.cur.execute("DELETE FROM models")
        self.cur.execute("DELETE FROM datasets")
        self.conn.commit()