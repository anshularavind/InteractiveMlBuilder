from interface import UserDatabase

def test_add_users():
    db = UserDatabase()
    db.clear()
    db.add_user("test3", "test3")
    print(db.get_user("test"))
    db.add_user("test4", "test4")
    print(db.get_users())
    db.close()

if __name__ == "__main__":
    test_add_users()