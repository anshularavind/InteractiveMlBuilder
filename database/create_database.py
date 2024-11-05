import psycopg2

conn = psycopg2.connect(host="localhost",
                        database="postgres",
                        user="postgres",
                        password="1234",
                        port=5432)

cur = conn.cursor()

# create tables
cur.execute("""CREATE TABLE IF NOT EXISTS users (
    uuid INT PRIMARY KEY,
    username VARCHAR(255),
    password VARCHAR(255)
);
""")

cur.execute("""CREATE TABLE IF NOT EXISTS models (
    uuid INT PRIMARY KEY,
    user_uuid INT,
    model_type VARCHAR(255),
    model_path VARCHAR(255),
    config_path VARCHAR(255),
    created_at TIMESTAMP
);
""")

cur.execute("""CREATE TABLE IF NOT EXISTS datasets (
    uuid INT PRIMARY KEY,
    user_uuid INT,
    dataset_name VARCHAR(255),
    created_at TIMESTAMP
);
""")

conn.commit()

cur.close()
conn.close()