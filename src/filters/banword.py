import sqlite3
import database.py


banworddb = sqlite3.connect(".cache/banwords.db")
banwordcur = bandb.cursor()
banwords = database.getBanwords(banworddb)

def isBanwordContained(comment):
    wordlist = comment.split(" ")
    for k in wordlist:
        if k in banwords:
            return True
    return False
