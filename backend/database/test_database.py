from interface import UserDatabase

def test_add_users():
    db = UserDatabase()
    db.clear()
    db.add_user("test3")
    print(db.get_user("test3"))
    db.add_user("test4")
    print(db.get_users())
    db.close()