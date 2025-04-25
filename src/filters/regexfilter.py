from Comment import Comment
import re

def containLink(comment):
    return re.search(comment.content, r"https?:\/\/[^\s]+")
