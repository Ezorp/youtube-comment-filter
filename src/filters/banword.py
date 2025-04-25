import sqlite3
import filters.database

def isBanwordContained(comment, banwords):
    wordlist = comment.split(" ")
    for k in wordlist:
        if k in banwords:
            return True
    return False
