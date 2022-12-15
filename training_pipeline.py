import xgboost
import hopsworks
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from utils.feature_processing import get_preprocessor
from utils.feature_validation import validate_preprocessor
from dotenv import load_dotenv

# TODO: Make script executable on Modal
load_dotenv()
project = hopsworks.login()
fs = project.get_feature_store()

def create_feature_view():
    posts_fg = fs.get_feature_group("reddit_posts", version=1)
    users_fg = fs.get_feature_group("reddit_users", version=1)
    subreddits_fg = fs.get_feature_group("reddit_subreddits", version=1)

    # Select features that are necessary for joins or model training and exclude the meta data
    posts_selection = posts_fg.select_except(features=["date_created","link","title","text"])
    users_selection = users_fg.select_except(features=["user_name"])
    subreddits_selection = subreddits_fg.select_except(features=["subreddit_name"])

    join_query = posts_selection.join(
                        users_selection, on=["user_id", "snapshot_time"]).join(
                            subreddits_selection, on=["subreddit_id", "snapshot_time"])

    # Note: There is hardly any documentation on which transformations are available (no onehot encoders?),
    # There is even less on how to implement custom ones and using the label_encoder we get an incomprehensible error
    # Thus, the most reasonable solution is to instead define our own sklearn pipeline.
    # OLD code:
    # Load transformation functions. Other functions that exist: "min_max_scaler", "robust_scaler"
    #label_encoder = fs.get_transformation_function(name="label_encoder", online=False)
    # Map features to transformations. Note that xgboost does not need scaling, but other algorithms would.
    #transformation_functions = {"subreddit_id": label_encoder}

    feature_view = fs.create_feature_view(name="reddit_features",
                                        version=1,
                                        description="Features and labels of the reddit dataset, excluding data that is not relevant for training.",
                                        labels=["num_likes", "upvote_ratio"],
                                        #transformation_functions=transformation_functions,
                                        query=join_query)
    return feature_view

def get_full_dataset():
    """
    Retrieves the dataset containing additional info like the raw text of the content and title of a post.
    """
    posts_fg = fs.get_feature_group("reddit_posts", version=1)
    users_fg = fs.get_feature_group("reddit_users", version=1)
    subreddits_fg = fs.get_feature_group("reddit_subreddits", version=1)
    full_join = posts_fg.select_all().join(
                        users_fg.select_all(), on=["user_id", "snapshot_time"]).join(
                            subreddits_fg.select_all(), on=["subreddit_id", "snapshot_time"])
    df_full = full_join.read()
    return df_full


def get_model_features(with_validation_set=False):
    try:
        feature_view = fs.get_feature_view(name="reddit_features", version=1)
    except:
        feature_view = create_feature_view()

    # TODO: After we have gathered enough data - let all test data come afer date x and the train data / validation data from before
    if with_validation_set:
        return feature_view.train_validation_test_split(validation_size=0.2, test_size=0.2)
    return feature_view.train_test_split(test_size=0.2)

# TODO: Bin the number of likes into ranges and try to predict the range instead of the exact number
X_train, X_test, y_train, y_test = get_model_features()
model = Pipeline(steps=[
                    ("preprocessor", get_preprocessor("tree")),
                    ("model", xgboost.XGBRegressor())
                    ])
validate_preprocessor(model.named_steps["preprocessor"], X_train, "tree")


model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred, multioutput="raw_values")
mae = mean_absolute_error(y_test, y_pred, multioutput="raw_values")
r2 = r2_score(y_test, y_pred, multioutput="raw_values")

t1_label = y_test.columns[0]
t2_label = y_test.columns[1]
print(f"MSE {t1_label}: %.2f" % mse[0])
print(f"MAE {t1_label}: %.2f" % mae[0])
print(f"R2 {t1_label}: %.2f" % r2[0])
print(f"MSE {t2_label}: %.2f" % mse[1])
print(f"MAE {t2_label}: %.2f" % mae[1])
print(f"R2 {t2_label}: %.2f" % r2[1])

# TODO: Define and upload model together with the fitted preprocessor
# %%
