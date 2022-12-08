# LikeItOrNot
Scalable machine learning system to predict the number of likes a reddit post or comment is going to get if the user decides to post it.

The service consists of several components:

## Feature Pipeline
The feature pipeline is deployed on modal and run on a daily schedule. It scans a set of subreddits for new posts / comments, pre-processes them and adds them to the feature store on hopsworks.

Features that are going to be extracted and tested via feature selection include among others:
- Embedding of the post / comment generated with the encoder of a encoder-decoder neural network for language translation.
- Sentiment score obtained with a library like VADER
- User information: Karma, age of the account (think of troll accounts)
- subreddit that is posted on
- most recent like history of the user
- length of the post
- existence of strings like "tl;dr" that make the post more readable

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
