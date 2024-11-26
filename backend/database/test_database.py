from interface import UserDatabase

def test_add_users():
    db = UserDatabase()
    db.clear()
    db.add_user("test3")
    print(db.get_user("test3"))
    db.add_user("test4")
    print(db.get_users())
    db.close()

def test_add_models():
    db = UserDatabase()
    db.clear()
    db.add_user("test3")
    model_uuid = db.init_model(db.get_user("test3"), '{"model_size": 100}')
    print(db.get_models("test3"))
    print(db.get_model_dir(db.get_user("test3"), model_uuid))
    db.close()

def test_add_datasets():
    db = UserDatabase()
    db.clear()
    db.add_user("test3")
    db.add_dataset("test3", "dataset1", "path/to/dataset1")
    print(db.get_dataset("test3", "path/to/dataset1"))
    db.close()