import re

class Tweet:
    def __init__(self, pq):
        self.username = Tweet.get_username(pq)
        self.id = Tweet.get_id(pq)
        self.author_id = Tweet.get_authorid(pq)
        self.text = Tweet.get_text(pq)
        
        self.retweets = Tweet.get_retweets(pq)
        self.favorites = Tweet.get_favorites(pq)
        self.comments = Tweet.get_comments(pq)

        self.mentions = Tweet.get_mentions(pq)
        self.hashtags = " ".join(re.compile('(#\\w*)').findall(self.text))
        self.quotes = Tweet.get_quoted_tweet(pq)
        self.conversation = Tweet.get_conversation_id(pq)

        self.geo = Tweet.get_geospan(pq)
        self.date = Tweet.get_date(pq)

        self.permalink = Tweet.get_permalink(pq)

        self.retweeter = Tweet.get_retweeter(pq)
        self.retweet_id = Tweet.get_retweetid(pq)

    
    @staticmethod
    def get_username(pq):
        #return pq("span:first.username.u-dir b").text()
        return pq.attr('data-screen-name')
    
    @staticmethod
    def get_conversation_id(pq):
        return pq.attr('data-conversation-id')
        
    @staticmethod
    def get_authorid(pq):
        return pq.attr("data-user-id")

    @staticmethod
    def get_retweeter(pq):
        return pq.attr("data-retweeter")

    @staticmethod
    def get_mentions(pq):
        mentions = pq.attr('data-mentions')
        if mentions:
            return mentions.split(" ")
        return []

    @staticmethod
    def get_quoted_tweet(pq):
        """Quoted tweets look different to retweets"""
        res = pq("a.QuoteTweet-link").attr('href')
        return res

    @staticmethod
    def get_retweetid(pq):
        return pq.attr("data-retweet-id")

    @staticmethod
    def get_text(pq):
        raw = pq("p.js-tweet-text").text().replace('# ', '#').replace('@ ', '@')
        return re.sub(r"\s+", " ", raw)

    @staticmethod
    def _get_count(pq, path):
        return int(pq(path).attr("data-tweet-stat-count").replace(",", ""))

    @staticmethod
    def get_retweets(pq):
        path = "span.ProfileTweet-action--retweet span.ProfileTweet-actionCount"
        return Tweet._get_count(pq, path)

    @staticmethod
    def get_comments(pq):
        path = "span.ProfileTweet-action--reply span.ProfileTweet-actionCount"
        return Tweet._get_count(pq, path)

    @staticmethod
    def get_favorites(pq):
        path = "span.ProfileTweet-action--favorite span.ProfileTweet-actionCount"
        return Tweet._get_count(pq, path)

    @staticmethod
    def get_date(pq):
        return int(pq("small.time span.js-short-timestamp").attr("data-time"))

    @staticmethod
    def get_id(pq):
        return pq.attr("data-tweet-id")

    @staticmethod
    def get_permalink(pq):
        pl = pq.attr("data-permalink-path")
        if pl:
            return "https://twitter.com/" + pl
        return None

    @staticmethod
    def get_geospan(pq):
        geo_span = pq("span.Tweet-geo")
        if geo_span:
            return geo_span.attr('title')
        return ""

def tweet_to_dict(tweet):
    #print(tweet.username, tweet.text)
    res = {
                "username": tweet.username,
                "id_str": tweet.id,
                "retweeter": tweet.retweeter,
                "retweet_id": tweet.retweet_id,
                "author": tweet.author_id,
                "geo": tweet.geo,
                "content": {
                    "text": tweet.text,
                    #{}"urls": tweet.urls,
                    "hashtags": tweet.hashtags,
                    "quotes": tweet.quotes,
                    "conversation": tweet.conversation
                },
                "interactions": {
                    "retweets": tweet.retweets,
                    "mentions": tweet.mentions,
                    "favorites": tweet.favorites
                },
                "permalink": tweet.permalink,
                "date": tweet.date
            }
    return res


