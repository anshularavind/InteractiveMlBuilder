from interface import UserDatabase

def test_add_users():
    db = UserDatabase()
    db.clear()
    db.add_user("test3")
    print(db.get_user_uuid("test3"))
    db.add_user("test4")
    print(db.get_users())
    db.close()

def test_add_models():
    db = UserDatabase()
    db.clear()
    db.add_user("test3")
    user_uuid = db.get_user_uuid("test3")
    model_uuid = db.init_model(user_uuid, '{"model_size": 100}')
    print(db.get_models(user_uuid))
    print(db.get_model_dir(user_uuid, model_uuid))
    db.close()

def test_add_datasets():
    db = UserDatabase()
    db.clear()
    db.add_user("test3")
    user_uuid = db.get_user_uuid("test3")
    db.add_dataset(user_uuid, "dataset1", "path/to/dataset1")
    print(db.get_dataset(user_uuid, "path/to/dataset1"))
    db.close()