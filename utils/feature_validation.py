import great_expectations as ge
from great_expectations.core import ExpectationConfiguration

def get_expectation_suites(df_posts, df_users, df_subreddits):
    ge_posts_df = ge.from_pandas(df_posts)
    ge_users_df = ge.from_pandas(df_users)
    ge_subreddits_df = ge.from_pandas(df_subreddits)
    posts_suite = ge_posts_df.get_expectation_suite()
    users_suite = ge_users_df.get_expectation_suite()
    subreddits_suite = ge_subreddits_df.get_expectation_suite()
    posts_suite.expectation_suite_name = "posts_suite"
    users_suite.expectation_suite_name = "users_suite"
    subreddits_suite.expectation_suite_name = "subreddits_suite"
    #### Add Expectations
    posts_suite.add_expectation(
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_unique",
            kwargs={"column": "post_id"}
        )
    )

    for feature_name in ["text_sentiment_negative", "text_sentiment_neutral", "text_sentiment_positive",
                        "title_sentiment_negative", "title_sentiment_neutral", "title_sentiment_positive",
                        "upvote_ratio"]:
        posts_suite.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_between",
                kwargs={
                    "column":feature_name,
                    "min_value":0,
                    "max_value":1
                    }
                )
            )

    for feature_name in ["subreddit_sentiment_negative_mean", "subreddit_sentiment_neutral_mean",
                        "subreddit_sentiment_positive_mean", "subreddit_sentiment_negative_stddev",
                        "subreddit_sentiment_neutral_stddev", "subreddit_sentiment_positive_stddev",
                        "subreddit_sentiment_negative_median", "subreddit_sentiment_neutral_median",
                        "subreddit_sentiment_positive_median", "subreddit_sentiment_negative_20th_percentile",
                        "subreddit_sentiment_neutral_20th_percentile", "subreddit_sentiment_positive_20th_percentile",
                        "subreddit_sentiment_negative_80th_percentile", "subreddit_sentiment_neutral_80th_percentile",
                        "subreddit_sentiment_positive_80th_percentile"]:
        subreddits_suite.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_between",
                kwargs={
                    "column":feature_name,
                    "min_value":0,
                    "max_value":1
                    }
                )
            )

    for feature_name in ["title", "text", "link"]:
        posts_suite.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_of_type",
                kwargs={
                    "column":feature_name,
                    "type_":"str"
                    }
                )
            )

    subreddits_suite.add_expectation(
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_of_type",
            kwargs={
                "column":"subreddit_name",
                "type_":"str"
                }
            )
        )

    posts_suite.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_of_type",
                kwargs={
                    "column":"contains_tldr",
                    "type_":"bool"
                    }
                )
            )

    for feature_name in ["has_verified_email", "is_gold", "is_mod"]:
        users_suite.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_of_type",
                kwargs={
                    "column":feature_name,
                    "type_":"bool"
                    }
                )
            )

    for feature_name in ["text_length", "num_likes", "hour_of_day", "day_of_week", "date_created"]:
        posts_suite.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_in_type_list",
                kwargs={
                    "column":feature_name,
                    "type_list":["int", "float"]
                }
            )
        )

    for feature_name in ["likes_hist_mean", "likes_hist_stddev", "likes_hist_median",
                        "likes_hist_20th_percentile", "likes_hist_80th_percentile",
                        "num_posts_last_month", "comment_karma", "link_karma", "account_age"]:
        users_suite.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_in_type_list",
                kwargs={
                    "column":feature_name,
                    "type_list":["int", "float"]
                }
            )
        )

    subreddits_suite.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_in_type_list",
                kwargs={
                    "column":"num_subscribers",
                    "type_list":["int", "float"]
                }
            )
        )

    for feature_name in ["embedding_title", "embedding_text"]:
        posts_suite.add_expectation(
                ExpectationConfiguration(
                    expectation_type="expect_column_value_lengths_to_equal",
                    kwargs={
                        "column":feature_name,
                        "value":384
                    }
                )
            )
    subreddits_suite.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_column_value_lengths_to_equal",
                kwargs={
                    "column":"embedding_description",
                    "value":384
                }
            )
        )

    posts_suite.add_expectation(
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_between",
            kwargs={
                "column":"hour_of_day",
                "min_value":0,
                "max_value":23
            }
        )
    )

    posts_suite.add_expectation(
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_between",
            kwargs={
                "column":"day_of_week",
                "min_value":0,
                "max_value":6
            }
        )
    )

    iso_regex = "^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{6})?$"
    for suite in [posts_suite, users_suite, subreddits_suite]:
        suite.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_match_regex",
                kwargs={
                    "column":"snapshot_time",
                    "regex": iso_regex
                }
            )
        )

    posts_suite.add_expectation(
        ge.core.ExpectationConfiguration(
            expectation_type="expect_column_values_to_match_regex",
            kwargs={
                "column":"link",
                "regex":"^/r/[^\s]+$"
            }
        )
    )

    return {
        "posts": posts_suite,
        "users": users_suite,
        "subreddits": subreddits_suite
    }



def validate_preprocessor(preprocessor, X, model_type):
    """
    :param X: A pandas dataframe containing the data to be preprocessed.
    :param preprocessor: A preprocessor as returned by get_preprocessor().
    """
    unprocessed_features = ["text_length", "text_sentiment_negative", "text_sentiment_neutral", "text_sentiment_positive", 
                            "title_sentiment_negative","title_sentiment_neutral","title_sentiment_positive", "contains_tldr",
                            "hour_of_day", "day_of_week",
                            # User features
                            "comment_karma", "link_karma", "is_gold", "is_mod", "has_verified_email", "account_age",
                            "num_posts_last_month","likes_hist_mean", "likes_hist_stddev", "likes_hist_median",
                            "likes_hist_80th_percentile", "likes_hist_20th_percentile", 
                            # Subreddit features
                            "subreddit_sentiment_negative_mean", "subreddit_sentiment_negative_stddev", 
                            "subreddit_sentiment_negative_median", "subreddit_sentiment_negative_80th_percentile",
                            "subreddit_sentiment_negative_20th_percentile", "subreddit_sentiment_neutral_mean",
                            "subreddit_sentiment_neutral_stddev", "subreddit_sentiment_neutral_median",
                            "subreddit_sentiment_neutral_80th_percentile", "subreddit_sentiment_neutral_20th_percentile",
                            "subreddit_sentiment_positive_mean", "subreddit_sentiment_positive_stddev",
                            "subreddit_sentiment_positive_median", "subreddit_sentiment_positive_80th_percentile",
                            "subreddit_sentiment_positive_20th_percentile", "num_subscribers"]
    num_cols_expected = len(unprocessed_features)
    
    # Three embedding columns embedding_text and embedding_title and embedding_description
    # that are expanded to 384 columns each
    num_cols_expected += 3*384

    # subreddit_id is one-hot encoded
    num_cols_expected += len(X["subreddit_id"].unique())

    if model_type == "tree":
        X_preprocessed = preprocessor.fit_transform(X)
        if not num_cols_expected==X_preprocessed.shape[1]:
            raise ValueError(f"Expected {num_cols_expected} columns after preprocessing, but got {X_preprocessed.shape[1]} instead.")
    else:
        raise NotImplementedError(f"Validation of preprocessor for model type {model_type} is not implemented.")
