import sys
from tweepy import API
from tweepy import OAuthHandler

def get_twitter_auth():
    try:
        consumer_key = "KXa9QLdmOqrYs8ZqOW9jDCodR"
        consumer_secret = "MnNIRKCls8OhnJC0zQpOVDAuZ9xKKAcuCsmP89jDzWXSiqSGDp"
        access_token = "3165550007-mC284CmDO2RR60yUJvJBCCj5JXW9NCVcgetb8WH"
        access_secret = "yfRaOF70HbETNPYHnwHxA2oHhQibEV7tMkyiXqMj3PlSn"
    except KeyError:
        sys.stderr.write("TWITTER_* environment variables not set\n")
        sys.exit(1)
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_secret)
    return auth

def get_twitter_client():
    auth = get_twitter_auth()
    client = API(auth)
    return client
