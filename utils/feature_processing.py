from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from scipy.special import softmax
from sentence_transformers import SentenceTransformer

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

import praw
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Load models for feature pre-processing
sentiment_model_name = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
sentiment_model = {
    "tokenizer": AutoTokenizer.from_pretrained(sentiment_model_name),
    "model": AutoModelForSequenceClassification.from_pretrained(sentiment_model_name)
}
text_encoder = SentenceTransformer('paraphrase-MiniLM-L6-v2')


def get_sentiment(text: str):
    """
    Returns three scores for the text: negative, neutral, positive
    """
    def preprocess(text):
        # Preprocess text (username and link placeholders)
        new_text = []
        for t in text.split(" "):
            t = 'http' if t.startswith('http') else t
            new_text.append(t)
        # Note that Roberta only accepts up to 512 tokens
        # I have not yet figured out how to select the number of tokens, so below is a quickhack
        return " ".join(new_text[:256]) # TODO: Select tokens in a from the tokenizer instead

    text = preprocess(text)
    encoded_input = sentiment_model["tokenizer"](text, return_tensors='pt')

    # Clip to 512 tokens
    for key in encoded_input.keys():
            encoded_input[key] = encoded_input[key][:, :512]

    print(len(encoded_input))
    output = sentiment_model["model"](**encoded_input)
    scores = output[0][0].detach().numpy()
    return softmax(scores)


def get_text_embedding(text: str):
    """
    Creates an embedding for the given text using the MiniLM model.
    https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
    """
    return text_encoder.encode([text])[0]


def contains_tldr(text: str):
    for variant in ["tldr", "tl;dr", "tl dr", "tl,dr", "tl:dr"]:
        if variant in text.lower():
            return True
    return False


def extract_user_features(user: praw.models.Redditor, snapshot_time: datetime):
    """
    See the reddit docs for redditors here:
    https://praw.readthedocs.io/en/stable/code_overview/models/redditor.html#praw.models.Redditor
    """
    # Extract like history
    likes = []
    submissions = user.submissions.new(limit=50)
    for submission in submissions:
        submission_age = snapshot_time - datetime.fromtimestamp(submission.created_utc)
        if submission_age < timedelta(days=30):
            likes.append(submission.score)

    return pd.DataFrame({
            # ID columns for joins
            "user_id": user.id,
            "snapshot_time": snapshot_time.isoformat(),    # utc Timestamp of when the data was extracted
            
            # Meta data (for manual checking - not for model)
            "user_name": user.name,

            # Model features
            "comment_karma": user.comment_karma,
            "link_karma": user.link_karma,
            "is_gold": user.is_gold,                        # Whether the user has premium status
            "is_mod": user.is_mod,                          # Whether the user is a moderator of ANY subreddit
            "has_verified_email": user.has_verified_email,
            "account_age": (snapshot_time - datetime.fromtimestamp(user.created_utc)).days,
            "num_posts_last_month": len(likes),
            "likes_hist_mean": np.mean(likes),
            "likes_hist_stddev": np.std(likes),
            "likes_hist_median": np.median(likes),
            "likes_hist_80th_percentile": np.percentile(likes, 80),
            "likes_hist_20th_percentile": np.percentile(likes, 20),
        }, index=[0])


def extract_post_features(post: praw.models.Submission, snapshot_time: datetime):
    """
    See the reddit docs for submissions / posts here:
    https://praw.readthedocs.io/en/stable/code_overview/models/submission.html#praw.models.Submission
    """
    sentiment = get_sentiment(post.selftext)
    print(post.selftext)
    print(sentiment)
    has_text = len(post.selftext.strip(" \n")) > 0
    features = {
            # ID columns for joins
            "post_id": post.id,
            "user_id": post.author.id,
            "subreddit_id": post.subreddit.id,
            "snapshot_time": snapshot_time.isoformat(),

            # Meta data (for manual checking - not for model)
            "date_created": post.created_utc,
            "link": post.permalink,
            "title": post.title,
            "text": post.selftext if has_text else "",

            # Model features and labels
            "num_likes": post.score,
            "upvote_ratio": post.upvote_ratio,
            "text_length": len(post.selftext.split(" ")) if has_text else 0,
            "sentiment_negative": sentiment[0],
            "sentiment_neutral": sentiment[1],
            "sentiment_positive": sentiment[2],
            #"topic": classify_topic(post.text),                                 # TODO: Check if usable: https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english?text=I+like+you.+I+love+you+yoyo#how-to-get-started-with-the-model
            "contains_tldr": contains_tldr(post.selftext),
            "hour_of_day": datetime.fromtimestamp(post.created_utc).hour,
            "day_of_week": datetime.fromtimestamp(post.created_utc).weekday(),
        }
    df_new_post = pd.DataFrame(features, index=[0])
    df_new_post["embedding_text"] = [get_text_embedding(post.selftext)]
    df_new_post["embedding_title"] = [get_text_embedding(post.title)]
    return df_new_post

def extract_subreddit_features(subreddit: praw.models.Subreddit, snapshot_time: datetime):
    """
    See the reddit docs for subreddits here:
    https://praw.readthedocs.io/en/stable/code_overview/models/subreddit.html#praw.models.Subreddit
    """

    features = {
        # ID columns for joins
        "subreddit_id": subreddit.id,
        "snapshot_time": snapshot_time.isoformat(),

        # Meta data (for manual checking - not for model)
        "subreddit_name": subreddit.display_name,

        # Model features
        "num_subscribers": subreddit.subscribers,
        # ...
    }

    negative_sentiments = []
    neutral_sentiments = []
    positive_sentiments = []

    # get sentiment of top posts
    for post in subreddit.new(limit=30):
        sentiment = get_sentiment(post.selftext)
        negative_sentiments.append(sentiment[0])
        neutral_sentiments.append(sentiment[1])
        positive_sentiments.append(sentiment[2])
    
    # calculate mean, median, and standard deviation of sentiment
    features["sentiment_negative_mean"] = np.mean(negative_sentiments)
    features["sentiment_negative_stddev"] = np.std(negative_sentiments)
    features["sentiment_negative_median"] = np.median(negative_sentiments)
    features["sentiment_neutral_mean"] = np.mean(neutral_sentiments)
    features["sentiment_neutral_stddev"] = np.std(neutral_sentiments)
    features["sentiment_neutral_median"] = np.median(neutral_sentiments)
    features["sentiment_positive_mean"] = np.mean(positive_sentiments)
    features["sentiment_positive_stddev"] = np.std(positive_sentiments)
    features["sentiment_positive_median"] = np.median(positive_sentiments)

    # count posts in the last week
    number_of_posts = 0
    for post in subreddit.new(limit=None):
        number_of_posts += 1
    features["num_posts_last_week"] = number_of_posts

    # embedding of the description of the subreddit

    df_new_subreddit = pd.DataFrame(features, index=[0])

    df_new_subreddit["embedding_description"] = [get_text_embedding(subreddit.description)]

    return df_new_subreddit

def get_subreddit_names():
    """
    Returns a list of subreddit names to extract data from.
    """
    return [
        # High quality subreddits
        "explainlikeimfive",
        "Showerthoughts",
        "AskReddit",
        "todayilearned",
        "AskMen",
        "AskWomen",
        "Jokes",
        "WritingPrompts",
        "nosleep",
        "IAmA",
        "LifeProTips",

        # More text-based subreddits
        "Parenting",
        "legaladvice",
        "bestoflegaladvice",
        "TalesFromRetail",
        "TalesFromYourServer",
        "IDontWorkHereLady",
        "TalesFromThePizzaGuy",
        "HFY",
        "SubredditSimulator",
        "javascript",
        "ChoosingBeggars",
        "AmItheAsshole",
        "dating_advice",
        "askscience",
        "movies",
        "technology",
        ]

        # https://www.reddit.com/r/talesfromtechsupport/


class ColumnExpander(BaseEstimator, TransformerMixin):
    """
    Class to expand a column containing arrays as string to multiple columns.
    Expects the arrays to have length 384 (as returned by the the sentence transformer paraphrase-MiniLM-L6-v2)
    """
    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        assert isinstance(X, pd.DataFrame), "Can only expand columns if X is a pandas dataframe."

        columns = self.columns if self.columns is not None else X.columns
        for col in columns:
            # The embedding is stored as string and must be converted back to a numpy array
            to_array = lambda x: np.fromstring(x.strip("[]"), dtype=float, sep=",")
            X.loc[:, col] = X[col].apply(to_array)

            # Check if the array has the expected length
            array_len = len(X[col].values[0])
            assert array_len == 384, f"Expected arrays in {col} to be an embedding with size 384, but got {array_len} instead."
            
            # Expand the column into multiple ones and remove the original column
            df_expanded = pd.DataFrame(X[col].tolist(),
                                        index=X.index,
                                        columns=[f"{col}_{str(i).zfill(3)}" for i in range(384)])
            X.loc[:, df_expanded.columns] = df_expanded.values
            X.drop(columns=[col], inplace=True)
        return X


def get_preprocessor(model_type="tree"):
    """
    Defines how to process the data before feeding it to a model.
    """
    if model_type == "tree":
        return ColumnTransformer(transformers=[
                                    ("onehot_encoder", OneHotEncoder(sparse=False, handle_unknown="ignore"),["subreddit_id"]),
                                    ("column_expander", ColumnExpander(), ["embedding_text", "embedding_title"]),
                                    ("drop_columns", "drop", ["post_id", "user_id", "snapshot_time"])
                                    ],
                       remainder='passthrough',
                       verbose_feature_names_out=True)
    else:
        # NOTE: If using non-tree based models, we need to scale the data
        raise NotImplementedError(f"Preprocessing data for model type {model_type} is not implemented.")
