import sys, re, os, time, random
import progressbar
import json as jsonlib
from lxml import etree

if sys.version_info[0] < 3:
    import cookielib
    import urllib2 as rq
    from urllib import quote as uquote
else:
    from urllib.parse import quote as uquote
    import http.cookiejar as cookielib
    import urllib.request as rq

from pyquery import PyQuery
from .Tweet import Tweet, tweet_to_dict

# ETPARSER = etree.ETCompatXMLParser()


class TweetManager:
    @staticmethod
    def make_outdir(name):
        if os.path.isdir(name):
            raise ValueError("Output directory {} exists.  Pick another".format(name))
        os.mkdir(name)

    @staticmethod
    def write_batch(out, data, batchnum):
        with open(os.path.join(out, "batch_{}.json".format(batchnum)), 'w') as outfile:
            outfile.write(jsonlib.dumps([tweet_to_dict(d) for d in data]))

    @staticmethod
    def write_config(out, data):
        with open(os.path.join(out, "config.txt"), 'w') as outfile:
            outfile.write(data)

    def __init__(self):
        pass

    @staticmethod
    def getTweets(tweetCriteria, receiveBuffer=None, bufferLength=100, proxy=None, outdir=None, batchsize=1000, randsleep=0):
        bar = progressbar.ProgressBar(max_value=tweetCriteria.maxTweets)
        refreshCursor = ''

        results = []
        resultsAux = []
        cookieJar = cookielib.CookieJar()
        if outdir:
            TweetManager.make_outdir(outdir)
            TweetManager.write_config(outdir, tweetCriteria.get_data())
            batchnum = 0
        if randsleep:
            minsleep = randsleep - randsleep // 10
            maxsleep = randsleep + randsleep // 10

        active = True

        while active:
            bar.update(len(results) + (tweetCriteria.maxTweets - tweetCriteria.remaining))
            if outdir and len(results) > batchsize:
                TweetManager.write_batch(outdir, results, batchnum)
                batchnum += 1
                tweetCriteria.remaining -= len(results)
                results = []

            if randsleep:
                stime = random.randint(minsleep, maxsleep)
                #sys.stderr.write("sleeping: {}".format(stime))
                time.sleep(stime/1000000)


            json = TweetManager.getJsonReponse(tweetCriteria, refreshCursor, cookieJar, proxy)
            if len(json['items_html'].strip()) == 0:
                #print("nothing found in items")
                sys.stderr.write(jsonlib.dumps(json))
                break

            refreshCursor = json['min_position']
            text = json["items_html"]

            pq = PyQuery(text, parser='html')
            tweets = pq('div.js-stream-tweet')

            if len(tweets) == 0:
                #print("\n\n\nno tweets found :(")
                break

            for tweetHTML in tweets:
                tpq = PyQuery(tweetHTML, parser='html_fragments')
                tweet = Tweet(tpq)

                results.append(tweet)
                resultsAux.append(tweet)

                if receiveBuffer and len(resultsAux) >= bufferLength:
                    receiveBuffer(resultsAux)
                    resultsAux = []

                if tweetCriteria.remaining > 0 and len(results) >= tweetCriteria.remaining:
                    active = False
                    break

        if receiveBuffer and len(resultsAux) > 0:
            receiveBuffer(resultsAux)

        if outdir:
            TweetManager.write_batch(outdir, results, batchnum)
            batchnum += 1
            tweetCriteria.remaining -= len(results)
            results = []

        return results

    @staticmethod
    def getJsonReponse(tweetCriteria, refreshCursor, cookieJar, proxy):
        url, data = tweetCriteria.url(), tweetCriteria.get_data()
        url = url % (uquote(data), refreshCursor)
        headers = [
                ('Host', "twitter.com"),
                ('User-Agent', "Mozilla/5.0 (Windows NT 6.1; Win64; x64)"),
                ('Accept', "application/json, text/javascript, */*; q=0.01"),
                ('Accept-Language', "de,en-US;q=0.7,en;q=0.3"),
                ('X-Requested-With', "XMLHttpRequest"),
                ('Referer', url),
                ('Connection', "keep-alive")
        ]

        if proxy:
            opener = rq.build_opener(
                    rq.ProxyHandler({'http': proxy, 'https': proxy}), rq.HTTPCookieProcessor(cookieJar))
        else:
            opener = rq.build_opener(rq.HTTPCookieProcessor(cookieJar))
        opener.addheaders = headers
        try:
            #print( url )
            response = opener.open(url)
            jsonResponse = response.read()
        except Exception as e:
            sys.stderr.write("Twitter weird response. Try to see on browser: https://twitter.com/search?q={}&src=typd".format(uquote(data)))
            sys.exit()

        dataJson = jsonlib.loads(jsonResponse.decode())

        return dataJson
