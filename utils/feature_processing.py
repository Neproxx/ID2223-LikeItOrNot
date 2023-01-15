from scipy.special import softmax
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import numpy as np

import praw
import numpy as np
import pandas as pd
from warnings import warn
from datetime import datetime, timedelta

# Models for sentiment analysis and text encoding
sentiment_model_name = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
sentiment_model = None
text_encoder = None


def load_sentiment_model():
    global sentiment_model
    if sentiment_model is None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        sentiment_model = {
            "tokenizer": AutoTokenizer.from_pretrained(sentiment_model_name),
            "model": AutoModelForSequenceClassification.from_pretrained(sentiment_model_name)
        }
    return sentiment_model


def load_text_encoder():
    global text_encoder
    if text_encoder is None:
        from sentence_transformers import SentenceTransformer
        text_encoder = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    return text_encoder


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
        return " ".join(new_text)

    try:
        sentiment_model = load_sentiment_model()
        text = preprocess(text)
        encoded_input = sentiment_model["tokenizer"](text, return_tensors='pt', max_length=512, truncation=True)
        output = sentiment_model["model"](**encoded_input)
        scores = output[0][0].detach().numpy()
        return softmax(scores)
    except Exception as e:
        warn(f"Could not extract sentiment for text with length {len(text.split(' '))} words: {text}")
        print(e)
        print("sentiment output:")
        print(sentiment_model["model"](**encoded_input))
        return (np.nan, np.nan, np.nan)


def get_text_embedding(text: str):
    """
    Creates an embedding for the given text using the MiniLM model.
    https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
    """
    text_encoder = load_text_encoder()
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
    # If there are no historic likes, use -999 as placeholder for the statistical features,
    # as this value is outside the range of possible values.
    has_post_history = len(likes) > 0
    try:
        return pd.DataFrame({
                # ID columns for joins
                "user_id": user.id if user.id else "unknown_user_id",
                "snapshot_time": snapshot_time.isoformat(),    # utc Timestamp of when the data was extracted
                
                # Meta data (for manual checking - not for model)
                "user_name": user.name if user.name else "unknown_user_name",

                # Model features
                "comment_karma": user.comment_karma if user.comment_karma else 0,
                "link_karma": user.link_karma if user.link_karma else 0,
                "is_gold": user.is_gold if user.is_gold else False,
                "is_mod": user.is_mod if user.is_mod else False,
                "has_verified_email": user.has_verified_email if user.has_verified_email else False,
                "account_age": (snapshot_time - datetime.fromtimestamp(user.created_utc)).days,
                "num_posts_last_month": len(likes),
                "likes_hist_mean": np.mean(likes) if has_post_history else -999,
                "likes_hist_stddev": np.std(likes) if has_post_history else -999,
                "likes_hist_median": np.median(likes) if has_post_history else -999,
                "likes_hist_80th_percentile": np.percentile(likes, 80) if has_post_history else -999,
                "likes_hist_20th_percentile": np.percentile(likes, 20) if has_post_history else -999,
            }, index=[0])
    except Exception as e:
        warn(f"Could not extract user features for user {user.name}")
        raise e


def extract_post_features(post: praw.models.Submission, snapshot_time: datetime):
    """
    See the reddit docs for submissions / posts here:
    https://praw.readthedocs.io/en/stable/code_overview/models/submission.html#praw.models.Submission
    """
    sentiment_title = get_sentiment(post.title)
    sentiment_text = get_sentiment(post.selftext)
    has_text = len(post.selftext.strip(" \n")) > 0 if post.selftext else False
    features = {
            # ID columns for joins
            "post_id": post.id if post.id else "unknown_post_id",
            "user_id": post.author.id if post.author.id else "unknown_user_id",
            "subreddit_id": post.subreddit.id if post.subreddit.id else "unknown_subreddit_id",
            "snapshot_time": snapshot_time.isoformat(),

            # Meta data (for manual checking - not for model)
            "date_created": post.created_utc if post.created_utc else "unknown_date_created",
            "link": post.permalink if post.permalink else "unknown_permalink",
            "title": post.title if post.title else "",
            "text": post.selftext if has_text else "",

            # Model features and labels
            "num_likes": post.score,
            "upvote_ratio": post.upvote_ratio,
            "text_length": len(post.selftext.split(" ")) if has_text else 0,
            "text_sentiment_negative": sentiment_text[0],
            "text_sentiment_neutral": sentiment_text[1],
            "text_sentiment_positive": sentiment_text[2],
            "title_sentiment_negative": sentiment_title[0],
            "title_sentiment_neutral": sentiment_title[1],
            "title_sentiment_positive": sentiment_title[2],
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

    negative_sentiments = []
    neutral_sentiments = []
    positive_sentiments = []

    # Compute distribution of sentiments of top posts in the last week
    for post in subreddit.top(limit=50, time_filter="week"):
        sentiment = get_sentiment(post.selftext)
        negative_sentiments.append(sentiment[0])
        neutral_sentiments.append(sentiment[1])
        positive_sentiments.append(sentiment[2])

    features = {
        # ID columns for joins
        "subreddit_id": subreddit.id,
        "snapshot_time": snapshot_time.isoformat(),

        # Meta data (for manual checking - not for model)
        "subreddit_name": subreddit.display_name,

        # Model features
        "num_subscribers": subreddit.subscribers,
        "subreddit_sentiment_negative_mean": np.mean(negative_sentiments),
        "subreddit_sentiment_negative_stddev": np.std(negative_sentiments),
        "subreddit_sentiment_negative_median": np.median(negative_sentiments),
        "subreddit_sentiment_negative_80th_percentile": np.percentile(negative_sentiments, 80),
        "subreddit_sentiment_negative_20th_percentile": np.percentile(negative_sentiments, 20),
        "subreddit_sentiment_neutral_mean": np.mean(neutral_sentiments),
        "subreddit_sentiment_neutral_stddev": np.std(neutral_sentiments),
        "subreddit_sentiment_neutral_median": np.median(neutral_sentiments),
        "subreddit_sentiment_neutral_80th_percentile": np.percentile(neutral_sentiments, 80),
        "subreddit_sentiment_neutral_20th_percentile": np.percentile(neutral_sentiments, 20),
        "subreddit_sentiment_positive_mean": np.mean(positive_sentiments),
        "subreddit_sentiment_positive_stddev": np.std(positive_sentiments),
        "subreddit_sentiment_positive_median": np.median(positive_sentiments),
        "subreddit_sentiment_positive_80th_percentile": np.percentile(positive_sentiments, 80),
        "subreddit_sentiment_positive_20th_percentile": np.percentile(positive_sentiments, 20),
    }

    # Embedding of the description of the subreddit
    df_new_subreddit = pd.DataFrame(features, index=[0])
    df_new_subreddit["embedding_description"] = [get_text_embedding(subreddit.description)]
    
    return df_new_subreddit

def get_subreddit_names(n_subreddits=10, random=False):
    """
    Returns a list of subreddit names to extract data from.
    """
    subreddits = [
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
        ]
    if random:
        np.random.shuffle(subreddits)
    return subreddits[:n_subreddits]


class ColumnExpander(BaseEstimator, TransformerMixin):
    """
    Class to expand a column containing arrays (as string) to multiple columns.
    Expects the arrays to have length 384 (as returned by the sentence transformer paraphrase-MiniLM-L6-v2)
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

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = self.columns
        return [f"{col}_{str(i).zfill(3)}" for col in input_features for i in range(384)]

class ColumnReorderer(BaseEstimator, TransformerMixin):
    """Ensures that the column order is the same during training and inference."""
    def __init__(self):
        self.column_order = None

    def fit(self, X, y=None):
        self.column_order = X.columns
        return self

    def transform(self, X):
        if self.column_order is None:
            raise ValueError("ColumnReorderer was not fitted yet.")
        try:
            return X[self.column_order]
        except KeyError as e:
            print(f"Error during column reordering: {e}")
            print(f"Expected {len(self.column_order)} columns: {self.column_order}")
            print(f"Found {len(X.columns)} columns: {X.columns}")
            raise e


def get_preprocessor(model_type="tree"):
    """
    Defines how to process the data before feeding it to a model.
    """
    if model_type == "tree":
        return ColumnTransformer(transformers=[
                                    ("onehot_encoder", OneHotEncoder(sparse=False, handle_unknown="ignore"),["subreddit_id"]),
                                    ("column_expander", ColumnExpander(), ["embedding_text", "embedding_title", "embedding_description"]),
                                    ("drop_columns", "drop", ["post_id", "user_id", "snapshot_time"])
                                    ],
                       remainder='passthrough',
                       verbose_feature_names_out=False)
    else:
        # NOTE: If using non-tree based models, we need to scale the data
        raise NotImplementedError(f"Preprocessing data for model type {model_type} is not implemented.")

def get_model_pipeline(model, model_type="tree"):
    return Pipeline(steps=[
                    ("column_reorderer", ColumnReorderer()),
                    ("preprocessor", get_preprocessor(model_type)),
                    ("model", model)
                    ])