from Comment.py import Comment


def isFullCaps(c):
    if c.flag:
        return True

    t = "".join(l for l in c.content if l.isalpha())
    (cap, nocap) = (0,0)

    for k in t:
        if k>="A":
            cap+=1
        else:
            nocap+=1
    return cap >= nocap
