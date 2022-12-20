# LikeItOrNot

Scalable machine learning system to predict the number of likes a reddit post or comment is going to get if the user decides to post it. The service consists of several components:

## Feature Pipeline

The feature pipeline is deployed on modal and run on a daily schedule. It scans a set of subreddits for new posts / comments, pre-processes them and adds them to the feature store on hopsworks.

A long list of subreddits are crawled on a daily basis, these include for example:

- /r/AskReddit
- /r/explainlikeimfive
- /r/Showerthoughts

The script extracts data for three entity types that are stored in their individual table namely reddit_users, reddit_posts and reddit_subreddits. Some of the features represent e.g. the raw text and are only kept for plausibilisation reasons. See the enumeration and description of all features below.

### reddit_posts

- `post_id`: Uniquely identifies the post / submission on reddit.
- `user_id`: Uniquely identifies the user that created the post.
- `subreddit_id`: Uniquely identifies the subreddit that the post was created in.
- `snapshot_time`: The time when the post was crawled. The same for associated entries in all three tables.
- `num_likes`: The number of upvotes the post received. This is the <strong>`primary label`</strong> we want to predict.
- `upvote_ratio`: The ratio of upvotes to downvotes which indicates how controversial a post is. This is a <strong>`secondary label`</strong> we want to predict.
- `date_created`: The time when the post was created on reddit.
- `link`: The URL to the post on reddit.
- `title`: The title of the post as string.
- `text`: The text of the post as string.
- `text_length`: The length of the text of the post in number of tokens (that correspond to words most of the time).
- `text_sentiment_negative`: The negative sentiment of the post text as float in the range [0, 1]. Obtained with [cardiffnlp/twitter-roberta-base-sentiment-latest](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest).
- `text_sentiment_neutral`: The neutral sentiment of the post text as float in the range [0, 1]. Obtained with [cardiffnlp/twitter-roberta-base-sentiment-latest](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest).
- `text_sentiment_positive`: The positive sentiment of the post text as float in the range [0, 1]. Obtained with [cardiffnlp/twitter-roberta-base-sentiment-latest](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest).
- `title_sentiment_negative`: The positive sentiment of the post title as float in the range [0, 1]. Obtained with [cardiffnlp/twitter-roberta-base-sentiment-latest](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest).
- `title_sentiment_neutral`: The positive sentiment of the post title as float in the range [0, 1]. Obtained with [cardiffnlp/twitter-roberta-base-sentiment-latest](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest).
- `title_sentiment_positive`: The positive sentiment of the post title as float in the range [0, 1]. Obtained with [cardiffnlp/twitter-roberta-base-sentiment-latest](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest).
- `contains_tldr`: Whether the post contains a "tldr" (too long, didn`t read) section.
- `hour_of_day`: The hour of the day when the post was created.
- `day_of_week`: The day of the week when the post was created.
- `embedding_text`: The 384-dimensional embedding of the text of the post (taking only the first 512 tokens into account). Obtained with [sentence-transformers/paraphrase-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/paraphrase-MiniLM-L6-v2).
- `embedding_title`: The 384-dimensional embedding of the title of the post (taking only the first 512 tokens into account). Obtained with [sentence-transformers/paraphrase-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/paraphrase-MiniLM-L6-v2).

### reddit_users

- `user_id`: Uniquely identifies the user on reddit.
- `snapshot_time`: The time when the user was crawled. The same for associated entries in all three tables.
- `user_name`: The name of the user on reddit.
- `comment_karma`: The number of upvotes the user received for comments.
- `link_karma`: The number of upvotes the user received for posts.
- `is_gold`: Whether the user has premium status.
- `is_mod`: Whether the user is a moderator of any subreddit.
- `has_verified_email`: Whether the user has verified their email address.
- `account_age`: The age of the account in days.
- `num_posts_last_month`: The number of posts the user created in the last month (we chose 50 to be the max).
- `likes_hist_mean`: The mean of the number of likes the user received for their posts in the last month.
- `likes_hist_stddev`: The standard deviation of the number of likes the user received for their posts in the last month.
- `likes_hist_median`: The median of the number of likes the user received for their posts in the last month.
- `likes_hist_80th_percentile`: The 80th percentile of the number of likes the user received for their posts in the last month.
- `likes_hist_20th_percentile`: The 20th percentile of the number of likes the user received for their posts in the last month.

### reddit_subreddits

- `subreddit_id`: Uniquely identifies the subreddit on reddit.
- `snapshot_time`: The time when the subreddit was crawled. The same for associated entries in all three tables.
- `subreddit_name`: The name of the subreddit on reddit.
- `num_subscribers`: The number of subscribers to the subreddit.
- `sentiment_negative_mean`: The mean of the negative sentiment of the most recent top posts in the subreddit.
- `sentiment_negative_stddev`: The standard deviation of the negative sentiment of the most recent top posts in the subreddit.
- `sentiment_negative_median`: The median of the negative sentiment of the most recent top posts in the subreddit.
- `sentiment_neutral_mean`: The mean of the neutral sentiment of the most recent top posts in the subreddit.
- `sentiment_neutral_stddev`: The standard deviation of the neutral sentiment of the most recent top posts in the subreddit.
- `sentiment_neutral_median`: The median of the neutral sentiment of the most recent top posts in the subreddit.
- `sentiment_positive_mean`: The mean of the positive sentiment of the most recent top posts in the subreddit.
- `sentiment_positive_stddev`: The standard deviation of the positive sentiment of the most recent top posts in the subreddit.
- `sentiment_positive_median`: The median of the positive sentiment of the most recent top posts in the subreddit.

### Yet to be further defined and implemented:

- `<subreddit_activity_metric>`: The activity metric of the subreddit, e.g. the number of posts created in the subreddit per day.
- `<subreddit_description_embedding>`: The embedding of the description of the subreddit. Obtained with sentence-transformers/paraphrase-MiniLM-L6-v2.
- `<post_topic_classification>`: Topic of the post as obtained with [distilbert-base-uncased-finetuned-sst-2-english](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english?text=I+like+you.+I+love+you+yoyo#how-to-get-started-with-the-model)

## Training Pipeline

We use an XGBoost regressor as the model and perform extensive hyperparameter search thanks to the efficient training process.

## Inference Pipeline / UI

An interactive UI allows the user to enter the content of his/her post or comment alongside his profile. The service will then extract the necessary features, query the model and display the number of likes to be expected.

## MLOps

At various steps along the feature and training pipelines, we add automated tests to detect problems like model drift or feature drift.

## Running the script

Firstly, if you want to run the scripts locally, make sure to have created a .env file that contains the necessary environment variables namely. If you plan on running the scripts on modal, you have to add the environment variables there. <span style="color:red">Add all of them to the same secret called "reddit-predict"</span>.

- `HOPSWORKS_API_KEY`
- `REDDIT_CLIENT_SECRET`
- `REDDIT_CLIENT_ID`
- `REDDIT_USER_AGENT`

Secondly, this repository contains a Dockerfile to build a container for this tool that runs in any local environment. In order to use it, simply run the below command which builds the container, starts it and provides you with a terminal inside the container where you can run the scripts from. In case of development, you can modify files locally and the changes will come into effect inside the container immediately. Note that building the container can take up to 15 minutes due to the large requirements.

```console
docker compose run reddit-predict
```

In case you want to run the script on `modal`, execute the following commands after you started the container and follow the instructions. It will generate a new token that the container needs to authenticate itself to modal. Also, do not forget to set `RUN_ON_MODAL=True`.

```console
modal token new
```

## Internal notes for implementation

Components that need to be created:

- feature pipeline deployed on modal
- hyperparameter search for xgboost model
- feature / models tests (MLOps)
- feature extraction
- feature selection
- Check if SHAP values can be used for ML explainability. Although probably not if the sentence embedding is our main feature :(
