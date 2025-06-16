import re
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

class CleanAndStemTweets(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
        self.stop_words = set(stopwords.words("english"))
        self.stemmer = PorterStemmer()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        cleaned = []
        for tweet in X:
            tweet = re.sub(r"http\S+|www\S+|https\S+", '', tweet)
            tokens = self.tokenizer.tokenize(tweet)
            tokens = [t for t in tokens if t.isalpha() and t not in self.stop_words]
            stems = [self.stemmer.stem(t) for t in tokens]
            cleaned.append(' '.join(stems))  # Option : retourne une chaîne pour compatibilité tokenizer
        return cleaned