import os
import hopsworks
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from warnings import warn

is_local=False
if is_local:
    from dotenv import load_dotenv
    load_dotenv()

MODEL_VERSION=22

@st.experimental_memo
def load_data():
    project = hopsworks.login()
    fs = project.get_feature_store()

    try:
        posts_fg = fs.get_feature_group("reddit_posts", version=os.getenv("POSTS_FG_VERSION", default=1))
        users_fg = fs.get_feature_group("reddit_users", version=os.getenv("USERS_FG_VERSION", default=1))
        subreddits_fg = fs.get_feature_group("reddit_subreddits", version=os.getenv("SUBREDDITS_FG_VERSION", default=1))
        full_join = posts_fg.select(features=["post_id", "snapshot_time", "num_likes", "upvote_ratio"]).join(
                            users_fg.select(features=["user_id", "snapshot_time"]), on=["user_id", "snapshot_time"]).join(
                                subreddits_fg.select(features=["subreddit_id", "snapshot_time"]), on=["subreddit_id", "snapshot_time"])
        df = full_join.read()
    except Exception as e:
        warn("Could not load data from feature store (most likely due to Port issues with Hopsworks). Trying to load the data from the model registry instead. Full exception:")
        warn(str(e))
        df = None

    # Load model including the generated images and evaluation scores
    mr = project.get_model_registry()
    model_hsfs = mr.get_model("reddit_predict", version=MODEL_VERSION)
    model_dir = model_hsfs.download()
    print("Model directory: {}".format(model_dir))

    metric_rows = {}
    metrics_avail = [m.replace("_likes","") for m in model_hsfs.training_metrics if "_likes" in m]
    for target in ["likes", "upvote_ratio"]:
        metric_rows[target] = []
        for metric in metrics_avail:
            metric_rows[target].append(model_hsfs.training_metrics[f"{metric}_{target}"])
    df_metrics = pd.DataFrame(metric_rows, index=metrics_avail)
    
    if df is None:
        try:
            df = pd.read_pickle(os.path.join(model_dir, "df_dashboard.pkl"))
        except:
            warn("Failed to load data from both the feature store and the model directory. Please upload the data to the model directory manually.")

    plots = {
        "predictions": plt.imread(f"{model_dir}/prediction_error.png"),
        "predictions_logscale": plt.imread(f"{model_dir}/prediction_error_logscale.png"),
        "confusion_matrix": plt.imread(f"{model_dir}/confusion_matrix.png"),
        "shap_numlikes": plt.imread(f"{model_dir}/shap_summary_plot_num_likes.png"),
        "shap_upvote_ratio": plt.imread(f"{model_dir}/shap_summary_plot_upvote_ratio.png"),
        "shap_numlikes_compact": plt.imread(f"{model_dir}/shap_summary_plot_compact.png")
    }

    return df, plots, df_metrics


df, plots, df_metrics = load_data()

if df is None:
    st.error("Could not load data from feature store or model directory as Huggingface has compatibility issues with parts of the data read API from Hopsworks.")
    st.stop()

# create a distribution plot of the number of likes using seaborn
st.title("Like It or Not")
st.markdown("This is the dashboard for the Like It Or Not model that predict the number of likes and the upvote ratio that a Reddit post is going to get.")

# Data stats
st.markdown("## Data Statistics")
col1, col2, col3 = st.columns(3)
col1.metric("Unqiue Posts", str(df["post_id"].nunique()))
col2.metric("Unique Users", str(df["user_id"].nunique()))
col3.metric("Unique Subreddits", str(df["subreddit_id"].nunique()))

# Distribution of the target variables
col1, col2 = st.columns(2)
col1.markdown("### Distribution of Number of Likes")
col2.markdown("### Distribution of Upvote Ratio")
col1, col2 = st.columns(2)
fig, ax = plt.subplots()
sns.histplot(df["num_likes"], ax=ax)
ax.set_ylabel("Number of posts")
ax.set_xlabel("Number of likes (log scale)")
ax.set_xscale("log")
plt.tight_layout()
col1.pyplot(fig)

fig2, ax = plt.subplots()
sns.distplot(df["upvote_ratio"], ax=ax, kde=False)
ax.set_ylabel("Number of posts")
plt.tight_layout()
col2.pyplot(fig2)

# Performance metrics
st.markdown("## Performance Metrics")
st.markdown("The model achieved the below scores on the test set. Please keep the effect of the sample weights in mind as explained in the Github repository. These reduce for example the R2 score from 0.75 to roughly 0.05. However, despite these low scores, the model is more useful in practice as it provides a meaningful lower bound estimate of the likes to be received as opposed to overestimating every post by up to 1500")
st.dataframe(df_metrics)

# Prediction error plots
st.markdown("## Prediction Error Plots")
st.markdown("The green line indicates the perfect prediction while the blue lines show point densities. Every point represents a prediction. The model is optimized for the number of likes and provides an estimate for the minimum number of likes expected. The upvote ratio does not perform well and would profit from dedicated modeling with another objective function if it is important.")
st.markdown("### Linear Scale")
st.image(plots["predictions"])
st.markdown("### Log Scale")
st.image(plots["predictions_logscale"])

# Confusion matrix
st.markdown("## Confusion Matrix")
st.markdown("After mapping the predicted number of likes to categories, the following confusion matrix can be obtained:")
st.image(plots["confusion_matrix"])

# Shap plots
st.markdown("## Shap Evaluation")
st.markdown("Shap values are an approach to machine learning explainability where the magnitude and kind of impact (positive / negative) of all features is computed." +
"Below, you see a beeswarm plot obtained on the predictions on the test data where every point represents a sample, its color tells if the feature had a high or low value " +
"and its position tells if the feature had a positive or negative impact on the prediction.")
st.image(plots["shap_numlikes"])

st.markdown("In addition, it is possible to sum up and average the absolute impact of all features over all samples. " +
            "The result can be interpreted as the feature importance. For the embedding features, we summed the values of the individual dimensions.")
st.image(plots["shap_numlikes_compact"])
