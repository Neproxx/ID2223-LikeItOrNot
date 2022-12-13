# LikeItOrNot
Scalable machine learning system to predict the number of likes a reddit post or comment is going to get if the user decides to post it.

The service consists of several components:

## Feature Pipeline
The feature pipeline is deployed on modal and run on a daily schedule. It scans a set of subreddits for new posts / comments, pre-processes them and adds them to the feature store on hopsworks.

#### Post features
- Embedding of the post / comment generated with the encoder of a encoder-decoder neural network for language translation.
- Sentiment score obtained with a library like VADER
  Better model that however could take more time to query (and would thus disqualify):
  https://huggingface.co/j-hartmann/emotion-english-distilroberta-base
- topic classification using the following model (if it does not take long to query):
  https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english
- subreddit that is posted on
- length of the post
- existence of strings like "tl;dr" that make the post more readable
- time of day + day of week

#### User features
- Karma
- Account age
- most recent like history of the user
- number of posts already created on the subreddit (by the user)
- number of posts already created that day (or binary - is first post of the day or not)

#### Subreddit features
- sentiment of the top 15 posts: mean, stddev, median
- Number of users in total
- Metric of activity (to be made more precise)

Start with the following subreddits:
- /r/AskReddit
- /r/todayilearned
- /r/gaming
- /r/explainlikeimfive
- /r/Showerthoughts

Features that are going to be extracted and tested via feature selection include among others:

## Training Pipeline
We use an XGBoost regressor as the model and perform extensive hyperparameter search thanks to the efficient training process.

## Inference Pipeline / UI
An interactive UI allows the user to enter the content of his/her post or comment alongside his profile. The service will then extract the necessary features, query the model and display the number of likes to be expected. 

## MLOps
At various steps along the feature and training pipelines, we add automated tests to detect problems like model drift or feature drift.


## Internal notes for implementation
Components that need to be created:
- feature pipeline deployed on modal
- hyperparameter search for xgboost model
- feature / models tests (MLOps)
- feature extraction
- feature selection
- Check if SHAP values can be used for ML explainability. Although probably not if the sentence embedding is our main feature :(
