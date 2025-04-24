class Comment:
    def __init__(self, author="default-author", date="1013-07-12T22:33:27.916", video="pRpeEdMkmQ0", content="Awsome video :)"):
        self.author = author
        self.date = date
        self.content = content
        self.video = video
        self.flag = False
        return

    def flag(self):
        self.flag = True
        return

    def __bool__:
        return self.flag
