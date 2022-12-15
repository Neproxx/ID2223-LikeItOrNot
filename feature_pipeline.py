RUN_ON_MODAL=False

# Specify maximum time to crawl in seconds. Modal will have a timeout threshold of MAX_CRAWL_TIME + 15 minutes
# The additional 15 minutes are to allow for the pipeline to finish and upload everything.
MAX_CRAWL_TIME = 30*60

def g():
    import os
    import pandas as pd
    import praw
    import hopsworks
    from datetime import datetime, timedelta
    from warnings import warn
    from tqdm import tqdm
    from utils.feature_processing import extract_post_features, extract_user_features, extract_subreddit_features, get_subreddit_names
    from utils.feature_validation import validate_samples, problems_found

    def upload_to_hopsworks(df_insert, feature_group, primary_key, name):
        """
        Inserting data to hopsworks sometimes results in (one of the three tables) to be empty.
        Maybe this happens only the first time a feature group is created, but we cannot know
        and thus have to be careful.
        """
        max_retries = 3
        for cur_try in range(1,max_retries+1):
            feature_group.insert(df_insert, write_options={"wait_for_job" : True})
            df_hopsworks = feature_group.select(features=primary_key).read()

            # Check if the data is in Hopsworks or if we have to try insert again
            has_failed=False
            for key in df_insert[primary_key].values:
                if key not in df_hopsworks[primary_key].values:
                    warn(f"Failed to upload dataset {name} to Hopsworks because key {key} was not found online, retrying...")
                    has_failed=True
                    break
            if not has_failed:
                print(f"Successfully uploaded dataset {name} to Hopsworks at try number {cur_try}")
                return True
        warn(f"Failed to upload dataset {name} to Hopsworks after {max_retries} retries, aborting...")
        return False


    def is_time_over(deadline: datetime):
        return datetime.utcnow() > deadline


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
            #print("Skipping post because it is too young: " + post.id + " " + post.title + " " + post.url)
            return False

        # Filter out posts that do not represent textual content but links to other pages
        if not post.is_self:
            #print("Skipping post because it is not a self post: " + post.id + " " + post.title + " " + post.url)
            return False

        if post.id in processed_post_ids:
            #print("Skipping post because it has already been processed: " + post.id + " " + post.title + " " + post.url)
            return False

        if not hasattr(post.author, "id"):
            #print("Skipping post because author is deleted: " + post.id + " " + post.title + " " + post.url)
            return False
        return True


    def extract_samples(processed_post_ids: set, deadline: datetime):
        """
        :param processed_post_ids: Set of post ids that have already been processed and should be skipped.
        :param deadline: Time after which the function should stop processing and upload the data to Hopsworks.
                         Otherwise, modal may throw a timeout error.
        """
        max_posts_per_subreddit = 15
        df_posts = pd.DataFrame()
        df_users = pd.DataFrame()
        df_subreddits = pd.DataFrame()
        processed_user_ids = set()

        subreddit_list = get_subreddit_names()
        for subreddit_name in subreddit_list:
            # TODO: Improve how we select posts to get a representative sample of good and bad posts
            # Round robin approach to sample posts from multiple subreddits?
            posts = [p for p in reddit.subreddit(subreddit_name).top("week", limit=1000)]
            snapshot_time = datetime.utcnow()

            new_subreddits = extract_subreddit_features(reddit.subreddit(subreddit_name), snapshot_time)
            df_subreddits = pd.concat([df_subreddits, new_subreddits], ignore_index=True)

            posts_cnt = 0
            for post in tqdm(posts):
                if should_process_post(post, processed_post_ids):
                    try:
                        new_post = extract_post_features(post, snapshot_time)
                        if should_process_user(post.author, processed_user_ids):
                            new_user = extract_user_features(post.author, snapshot_time)
                            df_users = pd.concat([df_users, new_user], ignore_index=True)
                            processed_user_ids.add(post.author.id)

                        df_posts = pd.concat([df_posts, new_post], ignore_index=True)
                        processed_post_ids.add(post.id)
                        posts_cnt += 1
                    except Exception as e:
                        print("Failed to process post: " + post.id + " " + post.title + " " + post.url)
                        print(e)

                if is_time_over(deadline) or posts_cnt==max_posts_per_subreddit:
                    break

            print(f"Extracted {posts_cnt} posts from subreddit {subreddit_name} in: {(datetime.utcnow() - snapshot_time).seconds} seconds")

            if is_time_over(deadline):
                warn("Timelimit reached, stopping sample extraction.")
                break

        inconsistencies = validate_samples(df_users, df_posts, df_subreddits)
        return df_posts, df_users, df_subreddits, inconsistencies

    deadline = datetime.utcnow() + timedelta(seconds=MAX_CRAWL_TIME)
    project = hopsworks.login()
    fs = project.get_feature_store()

    # NOTE:
    # Users and subreddits may be crawled several times, therefore the primary key
    # needs to include the snapshot time that connects them to the associated posts
    posts_fg = fs.get_or_create_feature_group(
        name="reddit_posts",
        version=1,
        primary_key=["post_id"],
        description="User posts crawled from Reddit")

    users_fg = fs.get_or_create_feature_group(
        name="reddit_users",
        version=1,
        primary_key=["user_id", "snapshot_time"],
        description="User profiles crawled from Reddit")

    subreddits_fg = fs.get_or_create_feature_group(
        name="reddit_subreddits",
        version=1,
        primary_key=["subreddit_id", "snapshot_time"],
        description="Subreddit information crawled from Reddit")

    # Retrieve ids of posts that have already been processed
    try:
        df_posts_hopsworks = posts_fg.select(features=["post_id"]).read()
        processed_post_ids = set(df_posts_hopsworks["post_id"].values.tolist())
    except:
        processed_post_ids = set()

    reddit = praw.Reddit(
        user_agent=os.environ["REDDIT_USER_AGENT"],
        client_id=os.environ["REDDIT_CLIENT_ID"],
        client_secret=os.environ["REDDIT_CLIENT_SECRET"],
    )
    df_posts, df_users, df_subreddits, inconsistencies = extract_samples(processed_post_ids, deadline)

    if problems_found(inconsistencies):
        warn("Consistency problems have been found when extracting new samples: " + str(inconsistencies))

    print("Inserting new samples into feature store")
    upload_to_hopsworks(df_posts, posts_fg, primary_key=["post_id"], name="reddit_posts")
    upload_to_hopsworks(df_users, users_fg, primary_key=["user_id", "snapshot_time"], name="reddit_users")
    upload_to_hopsworks(df_subreddits, subreddits_fg, primary_key=["subreddit_id", "snapshot_time"], name="reddit_subreddits")

    # TODO: call FeatureGroup.save_validation_report()
    #       or FeatureGroup.validate()
    # Set automatic alert (e.g. via email) if validation fails
    # https://www.hopsworks.ai/post/receiving-alerts-in-slack-email-pagerduty-from-hopsworks-support-for-managing-your-feature-store
    # One such check should make sure that the preprocessor creates the correct number of features

import modal
stub = modal.Stub()
image = modal.Image.debian_slim().pip_install(["hopsworks","joblib","praw","transformers", "sentence-transformers","torch", "pandas"]) 
@stub.function(image=image,
               schedule=modal.Period(days=1),
               secret=modal.Secret.from_name("reddit-predict"),
               mounts=[modal.Mount(remote_dir="/root/utils", local_dir="./utils")],
               timeout=MAX_CRAWL_TIME+900,
               # retries=3
               )
def f():
    g()

if __name__ == "__main__":
    if RUN_ON_MODAL:
        with stub.run():
            f.call()
    else:
        # NOTE: Create an .env file in the root directory and set HOPEWORKS_API_KEY=<your_api_key>
        from dotenv import load_dotenv
        load_dotenv()
        g()
