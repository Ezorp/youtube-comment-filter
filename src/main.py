from filters.banword import *
from filters.Comment import *
from filters.database import *
from filters.fullcaps import *
from filters.regexfilter import *
from llm.ollamaapilink import *
import csv

def promptbuilding(comment):
    return f"You are an AI agend made to identify comments made under a youtube video. Your task is to return \"Yes\" if you consider that the linked comment should be deleted and \"No\" otherwize.\nYour output must be strictly be \"Yes\" or \"No\".\n\nTo do so, you have acces to the author name, the date the comment was posted and the content of the message.\nShould be deleted any filter evasion, hate speach, advertisement to something, and so on.\nremember that a negative comment is not necessary spam and that trolling comments should not be consider as spam. \n\nAuthor name: {comment.author}\n\nComment date: {comment.date}\n\nBEGIN COMMENT CONTENT\n{comment.content}\nEND COMENT CONTENT"

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
        reader = csv.reader(f, delimiter=',')
        next(reader)  # skip header
        for row in reader:
            batch.append(Comment(video=row[0], author=row[1], date=row[2], content=row[3]))

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
            ban(comment.author, cur, bandb)

        if (not comment) and isFullCaps(comment):
            comment.flag = True
            # automatic ban :gigachad:
            ban(comment.author, cur, bandb)

        if (not comment) and isBanwordContained(comment.content, banwords):
            comment.flag = True
            ban(comment.author, cur, bandb)

        if (not comment) and isBanned(comment.author, cur):
            comment.flag = True
        
        if not comment:
            #print("AI")
            result = query_ollama(promptbuilding(comment))
            #print(comment.content + "\n" + result)
            if "Yes" in result:
                comment.flag = True
                ban(comment.author, cur, bandb)

    outputls = [["VIDEO","AUTHOR","DATE","TEXT","CLASS"]]
    while len(batch) > 0:
        c = batch.pop()
        outputls.append([c.video, c.author, c.date, c.content, 1 if c else 0])
    with open("output.csv", "w", newline="") as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerows(outputls)

    bandb.commit()
    bandb.close()
