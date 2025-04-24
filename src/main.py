from filters/banword.py import *
from filters/Comment.py import *
from filters/database.py import *
from filters/full-caps.py import *
from filters/regex-filter.py import *
from training/trainingScript.py import *
import csv

if __name__ == "__main__":

    promt = "This is a prompt"
    bandb = sqlite3.connect(".cache/bannedusers.db")
    cur = bandb.cursor()
    
    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL
    )
    """)

    banworddb = sqlite3.connect(".cache/banwords.db")
    cur2 = banworddb.cursor()
    
    cur.execute("""
    CREATE TABLE IF NOT EXISTS words (
        id INTEGER PRIMARY KEY,
        word TEXT NOT NULL
    )
    """)

    banwords = getBanwords(cur2)
    banworddb.commit()
    banworddb.close()

    #batch is a list of Comment s that are the one threated by the script.
    batch = []
    with open("batch.csv", newline='', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=';')
        next(reader)  # skip header
        for row in reader:
            data.append(Comment(video=row[0], author=row[1], date=row[2], content=row[3]))

    # banning user if posted more than 4 comments on a single video (anti spam)
    spamdict = {("example","default-name"):0}
    for k in batch:
        if (k.video,k.author) not in spamdict:
            spamdict[(k.video,k.author)] = 0
        else:
            spamdict[(k.video,k.author)] += 1
    for key in spamdict:
        if spamdict[key] > 4:

            # add user to the banned database.
            ban(key[1], cur)

    # begin comment per comment filtering
    for comment in batch:
        #regex filtering
        if containLink(comment):
            comment.flag = True
            # automatic ban of the author of a spam comment.
            ban(comment.author, cur)

        if (not comment) and isFullCaps(comment):
            comment.flag = True
            # automatic ban :gigachad:
            ban(comment.author, cur)

        if (not comment) and isBanwordContained(comment.content, banwords):
            comment.flag = True
            ban(comment.author, cur)

        if (not comment) and isBanned(comment.author, cur):
            comment.flag = True
        
        if not comment:
            result = query_ollama(promptbuilding(comment))
            if "yes" in result:
                comment.flag = True
                ban(comment.authoq)







    bandb.commit()
    bandb.close()
