from interface import UserDatabase

def test_add_users():
    db = UserDatabase()
    db.clear()
    db.add_user("test3")

    print(db.get_user_uuid("test3"))
    db.add_user("test4")
    config_json = {
        "username": "test_user3",
        "model_config": {
            "input": [3, 224, 224],
            "output": [10],
            "dataset": "cifar10",
            "lr": 0.001,
            "batch_size": 32,
            "blocks": [
                {
                    "block": "Conv2D",
                    "params": {
                        "channels": 64,
                        "kernel_size": 3,
                        "output_size": [64, 222, 222]
                    }
                }
            ]
        },
        "dataset": "cifar10"
    }
    db.init_model(db.get_user_uuid("test3"), config_json)
    print(db.get_user_name(db.get_user_uuid("test3")))
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

if __name__ == "__main__":
    test_add_models()
    test_add_datasets()