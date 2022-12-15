import pandas as pd

def validate_samples(df_users, df_posts, df_subreddits):
    tables = {
        'users': df_users,
        'posts': df_posts,
        'subreddits': df_subreddits
    }
    inconsistencies = dict((k, {}) for k in tables.keys())
    for table_name in tables.keys():
        df = tables[table_name]

        # Check if any column contains NaN values
        if pd.isna(df).any().any():
            nan_columns = df.columns[pd.isna(df).any()].tolist()
            inconsistencies[table_name]["nan_cols"] = nan_columns
    return inconsistencies


def problems_found(inconsistencies):
    for table_name in inconsistencies.keys():
        if len(inconsistencies[table_name].keys()) > 0:
            return True
    return False


def validate_preprocessor(preprocessor, X, model_type):
    """
    :param X: A pandas dataframe containing the data to be preprocessed.
    :param preprocessor: A preprocessor as returned by get_preprocessor().
    """
    unprocessed_features = ['text_length', 'sentiment_negative', 'sentiment_neutral', 'sentiment_positive', 'contains_tldr',
                            'hour_of_day', 'day_of_week', 'comment_karma', 'link_karma', 'is_gold', 'is_mod', 'has_verified_email',
                            'account_age', 'num_posts_last_month', 'likes_hist_mean', 'likes_hist_stddev', 'likes_hist_median',
                            'likes_hist_80th_percentile', 'likes_hist_20th_percentile', 'num_subscribers']
    num_cols_expected = len(unprocessed_features)
    
    # Two embedding columns embedding_text and embedding_title that are expanded to 384 columns each
    num_cols_expected += 2*384

    # subreddit_id is one-hot encoded
    num_cols_expected += len(X["subreddit_id"].unique())

    if model_type == "tree":
        X_preprocessed = preprocessor.fit_transform(X)
        if not num_cols_expected==X_preprocessed.shape[1]:
            raise ValueError(f"Expected {num_cols_expected} columns after preprocessing, but got {X_preprocessed.shape[1]} instead.")
    else:
        raise NotImplementedError(f"Validation of preprocessor for model type {model_type} is not implemented.")
