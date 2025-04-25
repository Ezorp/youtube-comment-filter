from filters.Comment import Comment
import re

def containLink(comment):
    #return re.search(comment.content, r"https?:\/\/[^\s]+")
    #return "https://" in comment.content or "http://" in comment.content
    pattern = r'https?://\S+|www\.\S+'
    return re.search(pattern, comment.content) is not None
