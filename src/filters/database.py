import sqlite3

bandb = sqlite3.connect(".cache/bannedusers.db")
cur = bandb.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    age INTEGER
)
""")
