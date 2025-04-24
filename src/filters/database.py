import sqlite3


def isBanned(username, cur):
    cur.execute("SELECT COUNT(*) FROM users WHERE name = ?",
                (username,)
                )
    return cur.fetchone()[0] != 0

def ban(username, cur):
    if isBanned(username, cur):
        return False
    cur.execute("INSERT INTO users (name) VALUES (?)", 
                (username))
    cur.commit()
    return True

if __name__ == "__main__":
    bandb = sqlite3.connect(".cache/bannedusers.db")
    cur = bandb.cursor()
    
    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL
    )
    """)
    cur.commit()
    cur.close()
