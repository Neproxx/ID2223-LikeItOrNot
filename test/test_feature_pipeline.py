import pytest
import praw
import datetime
import os
import numpy as np
from utils.feature_processing import extract_post_features, extract_user_features, extract_subreddit_features

local=False
if local:
    from dotenv import load_dotenv
    load_dotenv()

reddit = praw.Reddit(
        user_agent=os.environ["REDDIT_USER_AGENT"],
        client_id=os.environ["REDDIT_CLIENT_ID"],
        client_secret=os.environ["REDDIT_CLIENT_SECRET"],
    )
# https://www.reddit.com/r/AskReddit/comments/gjbiii/whats_a_delicious_poor_mans_meal/
post = reddit.submission(id="gjbiii")
snapshot_time = datetime.datetime(2022, 12, 24)

@pytest.mark.parametrize("post, snapshot_time", [(post, snapshot_time)])
def test_extract_post_features(post, snapshot_time):
    df = extract_post_features(post, snapshot_time)

    # Check that features were extracted
    assert df is not None and df.size > 0, "The dataframe should not be empty after extracting features from post"
    assert df.shape == (1, 22), f"Expected 22 columns after extracting features from post but got {df.shape[1]}"

    # ID columns
    assert df.loc[0, "post_id"] == "gjbiii"

    # meta data columns
    assert df.loc[0, "title"] == "What's a delicious poor man's meal?"

    # model features and labels
    assert df.loc[0, "text_length"] == 0
    assert df.loc[0, "contains_tldr"] == False

@pytest.mark.parametrize("subreddit, snapshot_time", [(post.subreddit, snapshot_time)])
def test_extract_subreddit_features(subreddit, snapshot_time):
    df = extract_subreddit_features(subreddit, snapshot_time)

    assert df is not None and df.size > 0, "The dataframe should not be empty after extracting features from subreddit"
    assert df.shape == (1, 20), f"Expected 20 columns after extracting features from subreddit but got {df.shape[1]}"
    assert df.loc[0, "subreddit_name"] == "AskReddit"
    assert not np.isnan(df.loc[0, "subreddit_sentiment_negative_mean"])
    assert not np.isnan(df.loc[0, "subreddit_sentiment_negative_stddev"])
    assert not np.isnan(df.loc[0, "subreddit_sentiment_negative_median"])
    assert not np.isnan(df.loc[0, "subreddit_sentiment_negative_80th_percentile"])
    assert not np.isnan(df.loc[0, "subreddit_sentiment_negative_20th_percentile"])
    assert not np.isnan(df.loc[0, "subreddit_sentiment_neutral_mean"])
    assert not np.isnan(df.loc[0, "subreddit_sentiment_neutral_stddev"])
    assert not np.isnan(df.loc[0, "subreddit_sentiment_neutral_median"])
    assert not np.isnan(df.loc[0, "subreddit_sentiment_neutral_80th_percentile"])
    assert not np.isnan(df.loc[0, "subreddit_sentiment_neutral_20th_percentile"])
    assert not np.isnan(df.loc[0, "subreddit_sentiment_positive_mean"])
    assert not np.isnan(df.loc[0, "subreddit_sentiment_positive_stddev"])
    assert not np.isnan(df.loc[0, "subreddit_sentiment_positive_median"])
    assert not np.isnan(df.loc[0, "subreddit_sentiment_positive_80th_percentile"])

@pytest.mark.parametrize("user, snapshot_time", [(post.author, snapshot_time)])
def test_extract_user_features(user, snapshot_time):
    df = extract_user_features(user, snapshot_time)

    assert df is not None and df.size > 0, "The dataframe should not be empty after extracting features from user"
    assert df.shape == (1, 15), f"Expected 15 columns after extracting features from user but got {df.shape[1]}"

