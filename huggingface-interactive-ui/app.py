import streamlit as st
import praw
import os
import datetime
import hopsworks
import pandas as pd
import joblib
import traceback
import matplotlib.pyplot as plt
from warnings import warn

# Deal with import paths that are different for the main_repo submodule files
import sys
sys.path.append(r'./main_repo')
sys.path.append(r'./main_repo/utils')
# get current directory as absolute path and add ./main_repo to path
sys.path.append(os.path.abspath(os.path.dirname(__file__)) + r'\main_repo')

from main_repo.utils.feature_processing import (extract_user_features, 
                                                extract_subreddit_features,
                                                get_text_embedding,
                                                get_sentiment,
                                                contains_tldr,
                                                get_subreddit_names,)
from main_repo.utils.training import post_process_predictions, generate_shap_forceplot

is_local=False
if is_local:
    from dotenv import load_dotenv
    load_dotenv()

MODEL_VERSION=22


def get_features(user_name: str, subreddit_name: str, post_title: str, post_text: str, post_date: datetime, post_time: datetime):
    now = datetime.datetime.utcnow()
    try:
        user_name = str(user_name).strip()
        subreddit_name = str(subreddit_name).strip()
        if user_name[:2] == "u/":
            user_name = user_name[2:]
        if subreddit_name[:2] == "r/":
            subreddit_name = subreddit_name[2:]
        redditor = reddit.redditor(user_name)
        subreddit = reddit.subreddit(subreddit_name)
    except Exception as e:
        warn(f"Could not find user {user_name} or subreddit with name {subreddit_name}")
        print(e)
        traceback.print_exc()
        return -1

    post_datetime = datetime.datetime.combine(post_date, post_time)
    try: 
        df_user = extract_user_features(redditor, now)
        df_subreddit = extract_subreddit_features(subreddit, now)

        print("post - user id: ", df_user["user_id"].values[0])
        print("post - subreddit id: ", df_subreddit["subreddit_id"].values[0])

        # Post features
        sentiment_text = get_sentiment(post_text)
        sentiment_title = get_sentiment(post_title)
        has_text = len(post_text.strip(" \n")) > 0
        post_features = {
            "snapshot_time": now.isoformat(),
            "text_length": len(post_text.split(" ")) if has_text else 0,
            "text_sentiment_negative": sentiment_text[0],
            "text_sentiment_neutral": sentiment_text[1],
            "text_sentiment_positive": sentiment_text[2],
            "title_sentiment_negative": sentiment_title[0],
            "title_sentiment_neutral": sentiment_title[1],
            "title_sentiment_positive": sentiment_title[2],
            "contains_tldr": contains_tldr(post_text),
            "hour_of_day": post_datetime.hour,
            "day_of_week": post_datetime.weekday(),

            # necessary for correct application of pipeline steps
            "post_id": "dummy_id",
            "user_id": df_user["user_id"].values[0],
            "subreddit_id": df_subreddit["subreddit_id"].values[0],
            "date_created": post_datetime.isoformat(),
            "link": "unknown_permalink",
            "title": post_title,
            "text": post_text if has_text else "",
        }
        df_post = pd.DataFrame(post_features, index=[0])
        df_post["embedding_text"] = [get_text_embedding(post_text)]
        df_post["embedding_title"] = [get_text_embedding(post_title)]
        df_final = pd.merge(df_post, df_user, on=["snapshot_time", "user_id"]).merge(df_subreddit, on=["snapshot_time", "subreddit_id"])
        
        # Preprocessor expects embedding columns to be strings as returned from feature store
        for col in df_final:
            if "embedding" in col:
                df_final[col] = df_final[col].apply(lambda a: str(a.tolist()))
    except Exception as e:
        warn(f"Could not extract features")
        print(e)
        traceback.print_exc()
        return -2

    return df_final

@st.experimental_memo
def load_model():
    project = hopsworks.login()
    mr = project.get_model_registry()
    model_hsfs = mr.get_model("reddit_predict", version=MODEL_VERSION)
    model_dir = model_hsfs.download()
    model = joblib.load(model_dir + "/reddit_model.pkl")
    return model


def query_model():
    df_features = get_features(user_name, subreddit_name, post_title, post_text, post_date, post_time)

    # Check for errors
    if isinstance(df_features, int) and df_features == -1:
        st.error("Could not find user or subreddit")
        return
    elif isinstance(df_features, int) and df_features == -2:
        st.error("Error when trying to extract features")
        return

    model = load_model()

    # Note that the order of the features is guaranteed to be correct because of the first step in the pipeline
    y_pred = model.predict(df_features)
    y_pred = post_process_predictions(y_pred)
    pred_num_likes = int(y_pred[0,0])
    pred_upvote_ratio = round(y_pred[0,1]*100, 2)

    like_label, like_description, like_emoji = get_like_category(pred_num_likes)
    ratio_label, ratio_description, ratio_emoji = get_ratio_category(pred_upvote_ratio)

    st.markdown("# Like It Or Not")
    st.markdown("A machine learning service that predicts the number of likes and upvote ratio of your Reddit post before you submit it. The initial computation may take a few seconds, as the model must be downloaded. Please be patient.")

    st.markdown("## Output")
    col1, col2 = st.columns(2)
    col1.metric("Likes", str(int(pred_num_likes)) + " " + like_emoji)
    col2.metric("Upvote Ratio", str(pred_upvote_ratio) + "% " + ratio_emoji)
    st.markdown(f"{like_description} You can expect an upvote ratio of {pred_upvote_ratio} which means that {pred_upvote_ratio}% of the people who see your post will upvote it (and {round(100-pred_upvote_ratio, 2)}% will downvote it). {ratio_description}")

    st.markdown("## Explanation")
    st.markdown("Below you can see how different features of your post affected the final prediction. " + 
                "The diagram shows the default value that would have been predicted in case no features about your post were known. " +
                "In addition, every feature is associated with a bar the color and length of which indicate the magnitude and type of impact it had on the prediction. " +
                "A long bar with red color states that the feature increased the prediction value by a large amount. " +
                "The exact meaning of the feature names and their values can be found at the [main Github repository](https://github.com/Neproxx/ID2223-LikeItOrNot). ")
    generate_shap_forceplot(model, df_features, output_dir="reddit_model", clear_figure=False)
    st.pyplot(plt.gcf(), clear_figure=False)

    st.session_state.has_predicted = True


def get_like_category(num_likes, include_emoji=True):
    # 0-10, 11-100, 101-1000, 1000+
    if num_likes <= 10:
        label = "Low"
        description = "It seems like your post will not get many likes, you should try to make it more interesting."
        emoji = "❄️"
    elif num_likes <= 100:
        label = "Medium"
        description = "It seems like your post will get a quite some attention although it will not necessarily become a top post."
        emoji = "🌡️"
    elif num_likes <= 1000:
        label = "High"
        description = "It seems like your post will get a lot of likes, you should try to make it even more interesting!"
        emoji = "🔥"
    else:
        label = "Very High"
        description = "Great job! It seems like your post will climb to the top of the subreddit and get a lot of attention!"
        emoji = "🔥🚒"
    return label, description, emoji

def get_ratio_category(upvote_ratio, include_emoji=True):
    if upvote_ratio <= 60:
        label = "negative"
        description = "This means that a majority of people will dislike your post."
        emoji = "🤬"
    elif upvote_ratio <= 85:
        label = "controversial"
        description = "This means that people will have mixed feelings about your post."
        emoji = "🗫"
    else:
        label = "positive"
        description = "This means that the overwhelming majority of people will love your post!"
        emoji = "❤️"
    return label, description, emoji


reddit = praw.Reddit(
        user_agent=os.environ["REDDIT_USER_AGENT"],
        client_id=os.environ["REDDIT_CLIENT_ID"],
        client_secret=os.environ["REDDIT_CLIENT_SECRET"],
    )

if 'has_predicted' not in st.session_state:
    st.session_state['has_predicted'] = False

# Add header only on first run
if not st.session_state['has_predicted']:
    st.markdown("# Like It Or Not")
    st.markdown("A machine learning service that predicts the number of likes and upvote ratio of your Reddit post before you submit it. The initial computation may take a few seconds, as the model must be downloaded. Please be patient.")

# Input elements
with st.sidebar:
    st.markdown("## Input")
    user_name = st.text_input("User name")
    subreddit_name = st.selectbox("Subreddit name", get_subreddit_names(n_subreddits=-1, random=False))
    post_title = st.text_input("Post title")
    post_text = st.text_area("Post text")
    post_date = st.date_input("Post date", value=datetime.datetime.now())
    post_time = st.time_input("Post time", value=datetime.datetime.now().time())
    submit_button = st.button("Predict", on_click=query_model)
