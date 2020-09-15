class TweetCriteria:
    def __init__(self, username=None, start=None, end=None, querySearch=None, maxTweets=0, topTweets=None, near=None, within="15mi"):
        self.username=username
        if self.username:
            self.username=self.username.strip("'\"")
        self.since=start
        self.until=end
        self.querySearch=querySearch
        if not maxTweets:
            maxTweets=1000000
        self.maxTweets=maxTweets
        self.remaining = self.maxTweets
        self.topTweets=topTweets
        self.near=near
        self.within=within

    def __repr__(self):
        return str(self.__dict__)

    def get_data(self):
        data = []
        if self.username:
            data.append("from:{}".format(self.username))
        if self.querySearch:
            data.append(self.querySearch)
        if self.near and self.within:
            data.append(
                    "&near:{} within:{}".format(
                        self.near, 
                        self.within))
        if self.since:
            data.append("since:{}".format(self.since))
        if self.until:
            data.append("until:{}".format(self.until))
        return " ".join(data)

    def url(self):
        if self.topTweets:
            return "https://twitter.com/i/search/timeline?q=%s&src=typd&max_position=%s" 
        else:
            return "https://twitter.com/i/search/timeline?f=tweets&q=%s&src=typd&max_position=%s"


    
    def setUsername(self, username):
        if username:
            username = username.strip("\"'") 
        self.username = username 
        return self    
    
    def setSince(self, since):
        self.since = since
        return self    

    def setUntil(self, until):
        self.until = until
        return self    

    def setQuerySearch(self, querySearch):
        self.querySearch = querySearch
        return self    

    def setMaxTweets(self, maxTweets):
        self.maxTweets = maxTweets
        return self

    def setTopTweets(self, topTweets):
        self.topTweets = topTweets
        return self

    def setNear(self, near):
        self.near = near
        return self

    def setWithin(self, within):
        self.within = within
        return self

