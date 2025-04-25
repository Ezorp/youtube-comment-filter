from filters.Comment import Comment


def isFullCaps(c):
    if c.flag:
        return True

    t = "".join(l for l in c.content if l.isalpha())
    cap = 0

    for k in t:
        if k<="Z":
            cap+=1
    # consider FullCaps if more than 70% of the content is capitalize
    return (cap/len(t)) > 0.7
