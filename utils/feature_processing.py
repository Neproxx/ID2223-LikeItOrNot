# %%
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from scipy.special import softmax
from sentence_transformers import SentenceTransformer

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
        return " ".join(new_text)

    text = preprocess(text)
    encoded_input = sentiment_model["tokenizer"](text, return_tensors='pt')
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
            "user_id": user.id,
            "snapshot_time": snapshot_time.isoformat(),    # utc Timestamp of when the data was extracted
            "user_name": user.name,
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
    features = {
            "post_id": post.id,
            "user_id": post.author.id,
            "subreddit_id": post.subreddit.id,
            "snapshot_time": snapshot_time.isoformat(),
            "created": post.created_utc,
            "link": post.permalink,
            "num_likes": post.score,
            "upvote_ratio": post.upvote_ratio,
            "title": post.title,
            "text": post.selftext,
            "text_length": len(post.selftext.split(" ")),
            "sentiment_negative": sentiment[0],
            "sentiment_neutral": sentiment[1],
            "sentiment_positive": sentiment[2],
            #"topic": classify_topic(post.text),                                 # TODO: Check if usable: https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english?text=I+like+you.+I+love+you+yoyo#how-to-get-started-with-the-model
            "contains_tldr": contains_tldr(post.selftext),
            "hour_of_day": datetime.fromtimestamp(post.created_utc).hour,
            "day_of_week": datetime.fromtimestamp(post.created_utc).weekday(),
        }
    embedding_text = get_text_embedding(post.selftext)
    embedding_title = get_text_embedding(post.title)
    for i in range(len(embedding_text)):
        features[f"embedding_text_{str(i).zfill(3)}"] = embedding_text[i]
    for i in range(len(embedding_title)):
        features[f"embedding_title_{str(i).zfill(3)}"] = embedding_title[i]
    return pd.DataFrame(features, index=[0])


def extract_subreddit_features(subreddit: praw.models.Subreddit, snapshot_time: datetime):
    """
    See the reddit docs for subreddits here:
    https://praw.readthedocs.io/en/stable/code_overview/models/subreddit.html#praw.models.Subreddit
    """
    features = {
        "subreddit_id": subreddit.id,
        "subreddit_name": subreddit.display_name,
        "snapshot_time": snapshot_time.isoformat(),
        "num_subscribers": subreddit.subscribers,
        # ...
    }
    # TODO: Retrieve sample of top posts (e.g. 30) and compute sentiment of top posts
    # Add features:
    # - sentiment_negative_mean
    # - sentiment_negative_stddev
    # - sentiment_negative_median
    # - sentiment_neutral_mean
    # - sentiment_neutral_stddev
    # - sentiment_neutral_median
    # - sentiment_positive_mean
    # - sentiment_positive_stddev
    # - sentiment_positive_median
    # - num_users_total
    # - <activity_metric>                                   # e.g. number of posts per day, avg number of comments on posts, etc...
    # - <embedding of the description of the subreddit?>
    return pd.DataFrame(features, index=[0])


# NOTE: This method can probably be deleted, as it was created to debug creating the feature groups but did not solve the problem
def get_column_names(fg_name, non_primary_only=False):
    if fg_name == "reddit_posts":
        cols = (["post_id", "user_id", "subreddit_id", "snapshot_time", "created", "link", "num_likes", "upvote_ratio",
                 "title", "text", "text_length", "sentiment_negative", "sentiment_neutral", "sentiment_positive",
                 "contains_tldr", "hour_of_day", "day_of_week"]
              + [f"embedding_text_{str(i).zfill(3)}" for i in range(384)]
              + [f"embedding_title_{str(i).zfill(3)}" for i in range(384)]
            )

    if fg_name == "reddit_users":
        cols = ["user_id", "snapshot_time", "user_name", "is_gold", "is_mod", "has_verified_email",
                "account_age", "likes_hist_mean", "likes_hist_stddev", "likes_hist_median",
                "likes_hist_80th_percentile", "likes_hist_20th_percentile", "num_posts_last_month"]

    if fg_name == "reddit_subreddits":
        # TODO: Add more features
        cols = ["subreddit_id", "snapshot_time", "subreddit_name", "num_subscribers"]

    if non_primary_only:
        cols = cols[1:] if fg_name == "reddit_posts" else cols[2:]

    print(f"Columns for {fg_name}: {cols}")

    return cols
