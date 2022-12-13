def g():
    import os
    import pandas as pd
    import praw
    import hopsworks
    from datetime import datetime, timedelta
    from warnings import warn
    from tqdm import tqdm
    from utils.feature_processing import extract_post_features, extract_user_features, extract_subreddit_features, get_column_names
    from utils.feature_validation import validate_samples, problems_found

    def should_process_user(user: praw.models.Redditor, processed_user_ids: set):
        if user.id in processed_user_ids:
            return False
        return True

    def should_process_post(post: praw.models.Submission, processed_post_ids: set):
        """
        Filters out posts that would be bad samples and should be skipped.
        For example media posts do not make sense for this service or posts that are already processed.
        """
        # Fiter out posts that are younger than 48 hours, as we assume that
        # the number of like to have converged after that time
        post_age = datetime.utcnow() - datetime.fromtimestamp(post.created_utc)
        if post_age < timedelta(days=2):
            return False

        # Filter out posts that do not represent textual content but links to other pages
        if not post.is_self:
            return False

        # Skip if duplicate
        if post.id in processed_post_ids:
            return False
        return True

    def extract_samples(processed_post_ids):
        """
        :param processed_post_ids: Set of post ids that have already been processed and should be skipped
        """
        max_posts_per_subreddit = 2
        df_posts = pd.DataFrame()
        df_users = pd.DataFrame()
        df_subreddits = pd.DataFrame()
        processed_user_ids = set()

        for subreddit_name in subreddit_list:
            # TODO: Improve how we select posts to get a representative sample of good and bad posts
            posts = [p for p in reddit.subreddit(subreddit_name).top("week", limit=200)]
            snapshot_time = datetime.utcnow()

            new_subreddits = extract_subreddit_features(reddit.subreddit(subreddit_name), snapshot_time)
            df_subreddits = pd.concat([df_subreddits, new_subreddits], ignore_index=True)

            for post in tqdm(posts[:max_posts_per_subreddit]):
                if should_process_post(post, processed_post_ids):
                    new_post = extract_post_features(post, snapshot_time)
                    df_posts = pd.concat([df_posts, new_post], ignore_index=True)

                    # NOTE: If we parallelize the feature extraction, we have to process all users at once first
                    # If we do it here, we risk to process the same user multiple times
                    if should_process_user(post.author, processed_user_ids):
                        new_user = extract_user_features(post.author, snapshot_time)
                        df_users = pd.concat([df_users, new_user], ignore_index=True)
                        processed_user_ids.add(post.author.id)

            end_time = datetime.now()
            print(f"Extracted {len(df_posts)} posts from subreddit {subreddit_name} in: {end_time - snapshot_time} seconds")

        inconsistencies = validate_samples(df_users, df_posts, df_subreddits)
        return df_posts, df_users, df_subreddits, inconsistencies

    project = hopsworks.login()
    fs = project.get_feature_store()

    # NOTE:
    # Users and subreddits may be crawled several times, therefore the primary key
    # needs to include the snapshot time that connects them to the associated posts
    posts_fg = fs.get_or_create_feature_group(
        name="reddit_posts_test1",                              # TODO: Change name after bug fix
        version=1,
        primary_key=["post_id"],
        #features=get_column_names("reddit_posts"),             # TODO: Remove line after bug fix
        description="User posts crawled from Reddit")

    users_fg = fs.get_or_create_feature_group(
        name="reddit_users_test1",                              # TODO: Change name after bug fix
        version=1,
        primary_key=["user_id", "snapshot_time"],
        #features = get_column_names("reddit_users"),           # TODO: Remove line after bug fix
        description="User profiles crawled from Reddit")

    subreddits_fg = fs.get_or_create_feature_group(
        name="reddit_subreddits_test1",                         # TODO: Change name after bug fix
        version=1,
        primary_key=["subreddit_id", "snapshot_time"],
        #features = get_column_names("reddit_subreddits"),      # TODO: Remove line after bug fix
        description="Subreddit information crawled from Reddit")

    # Retrieve ids of posts that have already been processed
    try:
        df_posts_hopsworks = posts_fg.select(features=["post_id"]).read()
        processed_post_ids = set(df_posts_hopsworks["post_id"].values.tolist())
    except:
        processed_post_ids = set()

    # Extract new samples
    reddit = praw.Reddit(
        user_agent=os.environ["REDDIT_USER_AGENT"],
        client_id=os.environ["REDDIT_CLIENT_ID"],
        client_secret=os.environ["REDDIT_CLIENT_SECRET"],
    )
    subreddit_list = ["AskReddit", "todayilearned", "gaming", "explainlikeimfive", "Showerthoughts"]
    df_posts, df_users, df_subreddits, inconsistencies = extract_samples(processed_post_ids)

    if problems_found(inconsistencies):
        warn("Consistency problems have been found when extracting new samples: " + str(inconsistencies))

    print("Inserting new samples into feature store")
    # DEBUG prints, as Hopsworks currently fails to create the feature groups
    #print("df_users: ")
    #df_users.info()
    #print("df_posts: ")
    #df_posts.info()
    #print("df_subreddits: ")
    #df_subreddits.info()

    # Iterate over all values of all data frames and print the type
    #print("USERS:")
    #for col_idx in range(len(df_users.columns)):
    #    print(f"type of {df_users.columns[col_idx]}: {type(df_users.iloc[0, col_idx])}")

    #print("SUBREDDITS:")
    #for col_idx in range(len(df_subreddits.columns)):
    #    print(f"type of {df_subreddits.columns[col_idx]}: {type(df_subreddits.iloc[0, col_idx])}")

    #print("POSTS:")
    #for col_idx in range(len(df_posts.columns)):
    #    print(f"type of {df_posts.columns[col_idx]}: {type(df_posts.iloc[0, col_idx])}")


    posts_fg.insert(df_posts, write_options={"wait_for_job" : False})
    users_fg.insert(df_users, write_options={"wait_for_job" : False})
    subreddits_fg.insert(df_subreddits, write_options={"wait_for_job" : False})

    # TODO: call FeatureGroup.save_validation_report()
    #       or FeatureGroup.validate()

import modal
stub = modal.Stub()
image = modal.Image.debian_slim().pip_install(["hopsworks","joblib","praw","transformers", "sentence-transformers","torch", "pandas"]) 
@stub.function(image=image,
               schedule=modal.Period(days=1),
               secret=modal.Secret.from_name("reddit-predict"),
               mounts=[modal.Mount(remote_dir="/root/utils", local_dir="./utils")])
def f():
    g()

LOCAL=True

if __name__ == "__main__":
    if LOCAL == True :
        # NOTE: Create an .env file in the root directory and set HOPEWORKS_API_KEY=<your_api_key>
        from dotenv import load_dotenv
        load_dotenv()
        g()

    else:
        with stub.run():
            f.call()

# TODO: When creating a feature view - use "num_likes" and "upvote_ratio" as labels
