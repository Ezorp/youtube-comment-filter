import sqlite3
import database.py

def isBanwordContained(comment, banwords):
    wordlist = comment.split(" ")
    for k in wordlist:
        if k in banwords:
            return True
    return False
