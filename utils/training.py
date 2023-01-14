import os
import pandas as pd
import xgboost
import hopsworks
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from hsml.schema import Schema
from hsml.model_schema import ModelSchema
from bayes_opt import BayesianOptimization, UtilityFunction
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, confusion_matrix
from sklearn.pipeline import Pipeline
from utils.feature_processing import get_model_pipeline
from utils.feature_validation import validate_preprocessor
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from sklearn.preprocessing import OneHotEncoder

def get_full_dataset(fs):
    """
    Retrieves the dataset containing additional info like the raw text of the content and title of a post.
    """
    posts_fg = fs.get_feature_group("reddit_posts", version=os.getenv("POSTS_FG_VERSION", default=1))
    users_fg = fs.get_feature_group("reddit_users", version=os.getenv("USERS_FG_VERSION", default=1))
    subreddits_fg = fs.get_feature_group("reddit_subreddits", version=os.getenv("SUBREDDITS_FG_VERSION", default=1))
    full_join = posts_fg.select_all().join(
                        users_fg.select_all(), on=["user_id", "snapshot_time"]).join(
                            subreddits_fg.select_all(), on=["subreddit_id", "snapshot_time"])
    df_full = full_join.read()
    return df_full

def get_model_features(with_validation_set=False, version=None):
        """
        :param version: The version of the train split to use. If None, a new split is generated.
        """
        project = hopsworks.login()
        fs = project.get_feature_store()
        try:
            feature_view = fs.get_feature_view(name="reddit_features", version=os.getenv("FEATURE_VIEW_VERSION", default=1))
        except Exception as e:
            feature_view = create_feature_view(fs)

        # TODO: After we have gathered enough data - let all test data come afer date x and the train data / validation data from before
        if with_validation_set and version is not None:
            return feature_view.get_train_validation_test_splits(training_dataset_version=version)
        elif with_validation_set:
            return feature_view.train_validation_test_split(validation_size=0.2, test_size=0.2)
        elif not with_validation_set and version is not None:
            return feature_view.get_train_test_split(training_dataset_version=version)
        return feature_view.train_test_split(test_size=0.2)

def create_feature_view(fs):
    posts_fg = fs.get_feature_group("reddit_posts", version=os.getenv("POSTS_FG_VERSION", default=1))
    users_fg = fs.get_feature_group("reddit_users", version=os.getenv("USERS_FG_VERSION", default=1))
    subreddits_fg = fs.get_feature_group("reddit_subreddits", version=os.getenv("SUBREDDITS_FG_VERSION", default=1))

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
                                        version=os.getenv("FEATURE_VIEW_VERSION", default=1),
                                        description="Features and labels of the reddit dataset, excluding data that is not relevant for training.",
                                        labels=["num_likes", "upvote_ratio"],
                                        #transformation_functions=transformation_functions,
                                        query=join_query)
    return feature_view

def get_optimal_hyperparameters(n_iterations):
    """
    Uses evaluate_model as black box function to optimize the hyperparameters of the model.
    """
    print("Loading data...")
    X_train, X_test, X_val, y_train, y_test, y_val = get_model_features(with_validation_set=True, version=None)
    print("Done.")

    print("Starting Bayesian Optimization...")
    def evaluate_model(n_estimators,max_depth,eta,subsample,colsample_bytree):
        """
        Trains the model using the full training set and evaluates it on the test set.
        Uses the training and test data from the enclosing context ("closure").
        """
        model = get_model_pipeline(
            xgboost.XGBRegressor(
                            n_estimators=int(n_estimators), 
                            max_depth=int(max_depth),
                            eta=eta,
                            subsample=subsample,
                            colsample_bytree=colsample_bytree),
            "tree"
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        y_pred = post_process_predictions(y_pred)
        rmse_num_likes = mean_squared_error(y_val, y_pred, squared=False, multioutput="raw_values")[0]
        #mae = mean_absolute_error(y_test, y_pred, multioutput="raw_values")
        #r2 = r2_score(y_test, y_pred, multioutput="raw_values")
        return -rmse_num_likes
    
    # Define parameter space to explore
    pbounds = {'n_estimators': (100, 1000), 'max_depth': (3, 16), 'eta': (0.01, 0.2), 'subsample': (0.5, 1), 'colsample_bytree': (0.5, 1)}
    optimizer = BayesianOptimization(
        f=evaluate_model,
        pbounds=pbounds,
        verbose=2,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
        random_state=1,
    )

    # Utility function to maximize
    utility = UtilityFunction(kind="ucb", kappa=2.5, xi=0.0)
    for _ in range(n_iterations):
        next_point_to_probe = optimizer.suggest(utility)
        print("Next point to probe: ", next_point_to_probe)
        target = evaluate_model(**next_point_to_probe)
        print("Target: ", target)
        optimizer.register(params=next_point_to_probe, target=target)
    print("Optimization finished. Result : ", optimizer.max)

    max_params = optimizer.max
    n_estimators = int(max_params["params"]["n_estimators"])
    max_depth = int(max_params["params"]["max_depth"])
    eta = max_params["params"]["eta"]
    subsample = max_params["params"]["subsample"]
    colsample_bytree = max_params["params"]["colsample_bytree"]
    X_train = pd.concat([X_train, X_val])
    y_train = pd.concat([y_train, y_val])

    return {
        "parameters": (n_estimators, max_depth, eta, subsample, colsample_bytree),
        "data": (X_train, X_test, y_train, y_test)
    }


# get optimal hyperparameters variant 2
# use neural network as black box function
def get_optimal_hyperparameters_nn(n_iterations):
    """
    Uses evaluate_model as black box function to optimize the hyperparameters of the model.
    """

    print("Loading data...")
    X_train, X_test, X_val, y_train, y_test, y_val = get_model_features(with_validation_set=True, version=None)
    print("Done.")
   
    print("Starting Bayesian Optimization...")
    def evaluate_model(learning_rate,dropout,hidden_layer_size):
        """
        Trains the model using the full training set and evaluates it on the test set.
        Uses the training and test data from the enclosing context ("closure").
        """
        model = get_model_pipeline(
            keras.Sequential([
                layers.Dense(int(hidden_layer_size), activation="relu", input_shape=[X_train.shape[1]]),
                layers.Dropout(dropout),
                layers.Dense(1)
            ]),
            "nn"
        )

        # extract model from pipeline
        model = model.named_steps["model"]  
        model.compile(loss="mse", optimizer=keras.optimizers.Adam(learning_rate=learning_rate), metrics=["mae", "mse"])

        model.fit(X_train, y_train, epochs=100, verbose=0)
        y_pred = model.predict(X_val)
        y_pred = post_process_predictions(y_pred)
        rmse_num_likes = mean_squared_error(y_val, y_pred, squared=False, multioutput="raw_values")[0]
        #mae = mean_absolute_error(y_test, y_pred, multioutput="raw_values")
        #r2 = r2_score(y_test, y_pred, multioutput="raw_values")
        return -rmse_num_likes
    
    # Define parameter space to explore
    pbounds = {'learning_rate': (0.001, 0.1), 'dropout': (0, 0.5), 'hidden_layer_size': (10, 100)}
    optimizer = BayesianOptimization(
        f=evaluate_model,
        pbounds=pbounds,
        verbose=2,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
        random_state=1,
    )

    # Utility function to maximize
    utility = UtilityFunction(kind="ucb", kappa=2.5, xi=0.0)
    for _ in range(n_iterations):
        next_point_to_probe = optimizer.suggest(utility)
        print("Next point to probe: ", next_point_to_probe)
        target = evaluate_model(**next_point_to_probe)
        print("Target: ", target)
        optimizer.register(params=next_point_to_probe, target=target)


def train_model(bayesian_search=True, bayesian_n_iterations=15):
    if bayesian_search:
        results = get_optimal_hyperparameters(n_iterations=bayesian_n_iterations)
        n_estimators, max_depth, eta, subsample, colsample_bytree = results["parameters"]
        X_train, X_test, y_train, y_test = results["data"]
        model = get_model_pipeline(
            xgboost.XGBRegressor(
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            eta=eta,
                            subsample=subsample,
                            colsample_bytree=colsample_bytree
                ),
            "tree"
        )
    else:
        model = get_model_pipeline(
            xgboost.XGBRegressor(
                n_estimators=500,
                max_depth=9,
                eta=0.01,
                colsample_bytree=0.5
            ),
            "tree"
        )
        print("Loading data...")
        X_train, X_test, y_train, y_test = get_model_features(with_validation_set=False, version=None)
        print("Done")
    validate_preprocessor(model.named_steps["preprocessor"], X_train, "tree")
    print("Starting final training run...")
    model.fit(X_train, y_train)
    return model, X_train, X_test, y_train, y_test


# train model variant 2 (neural network)
def train_model_nn(bayesian_search=True, bayesian_n_iterations=15):
    if bayesian_search:
        results = get_optimal_hyperparameters_nn(n_iterations=bayesian_n_iterations)
        learning_rate, dropout, hidden_layer_size = results["parameters"]
        X_train, X_test, y_train, y_test = results["data"]
        model = get_model_pipeline(
            keras.Sequential([
                layers.Dense(int(hidden_layer_size), activation="relu", input_shape=[X_train.shape[1]]),
                layers.Dropout(dropout),
                layers.Dense(1)
            ]),
            "nn"
        )

        # extract model from pipeline and compile it
        model = model.named_steps["model"]  
        model.compile(loss="mse", optimizer=keras.optimizers.Adam(learning_rate=learning_rate), metrics=["mae", "mse"])
    else:
        model = get_model_pipeline(
            keras.Sequential([
                layers.Dense(64, activation="relu", input_shape=[X_train.shape[1]]),
                layers.Dropout(0.2),
                layers.Dense(1)
            ]),
            "nn"
        )
        
        model = model.named_steps["model"]  
        model.compile(loss="mse", optimizer=keras.optimizers.Adam(learning_rate=0.01), metrics=["mae", "mse"])
        print("Loading data...")
        X_train, X_test, y_train, y_test = get_model_features(with_validation_set=False, version=None)
        print("Done")
    validate_preprocessor(model.named_steps["preprocessor"], X_train, "nn")
    print("Starting final training run...")
    model.fit(X_train, y_train, epochs=100, verbose=0)
    return model, X_train, X_test, y_train, y_test


def post_process_predictions(y_pred):
    """
    Post-processes the predictions of the model to ensure that the predictions are within the valid range.
    """
    # Number of likes can be positive or negative, but upvote ratio can only be between 0 and 1
    y_pred[1] = y_pred[1].clip(0, 1)
    return y_pred


def get_metrics(y_test, y_pred):
    print("GETTING METRICS...")
    rmse = mean_squared_error(y_test, y_pred, multioutput="raw_values", squared=False)
    mae = mean_absolute_error(y_test, y_pred, multioutput="raw_values")
    r2 = r2_score(y_test, y_pred, multioutput="raw_values")
    metrics = {
        "rmse_likes": rmse[0],
        "mae_likes": mae[0],
        "r2_likes": r2[0],
        "rmse_upvote_ratio": rmse[1],
        "mae_upvote_ratio": mae[1],
        "r2_upvote_ratio": r2[1]
    }

    print("Y_TEST: ", y_test)
    print("Y_PRED: ", y_pred)

    print("METRICS: ", metrics)

    print("RMSE num_likes: %.2f" % metrics["rmse_likes"])
    print("MAE num_likes: %.2f" % metrics["mae_likes"])
    print("R2 num_likes: %.2f" % metrics["r2_likes"])
    print("RMSE upvote_ratio: %.2f" % metrics["rmse_upvote_ratio"])
    print("MAE upvote_ratio: %.2f" % metrics["mae_upvote_ratio"])
    print("R2 upvote_ratio: %.2f" % metrics["r2_upvote_ratio"])

    return metrics


def generate_prediction_plots(y_test, y_pred):
    print("Creating plots...")
    if not os.path.isdir("reddit_model"):
        os.mkdir("reddit_model")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    sns.scatterplot(x=y_test["num_likes"], y=y_pred[:,0], ax=ax1)
    sns.scatterplot(x=y_test["upvote_ratio"], y=y_pred[:,1], ax=ax2)
    ax1.set_title("Predicted vs actual number of likes")
    ax2.set_title("Predicted vs actual upvote ratio")
    ax1.set_xlabel("Actual number of likes")
    ax1.set_ylabel("Predicted number of likes")
    ax2.set_xlabel("Actual upvote ratio")
    ax2.set_ylabel("Predicted upvote ratio")
    sns.kdeplot(x=y_test["num_likes"], y=y_pred[:,0], color='blue', ax=ax1)
    sns.kdeplot(x=y_test["upvote_ratio"], y=y_pred[:,1], color='blue', ax=ax2)

    # Draw line to indicate perfect prediction
    min_likes = min(y_test["num_likes"].min(), y_pred[:,0].min())
    max_likes = min(y_test["num_likes"].max(), y_pred[:,0].max())
    min_ratio = min(y_test["upvote_ratio"].min(), y_pred[:,1].min())
    max_ratio = max(y_test["upvote_ratio"].max(), y_pred[:,1].max())

    ax1.plot([min_likes, max_likes], [min_likes, max_likes], color='green')
    ax2.plot([min_ratio, max_ratio], [min_ratio, max_ratio], color='green')

    fig.savefig("reddit_model/prediction_error.png")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    fig.savefig("reddit_model/prediction_error_logscale.png")


#def generate_shap_forceplot(model: Pipeline, X_test):
#    # TODO
#    if not os.path.isdir("reddit_model"):
#        os.mkdir("reddit_model")
#
#    import shap
#    explainer = shap.TreeExplainer(model.named_steps["model"])
#
#    #def shap_plot(sample): 
#    #    shap_values = explainer.shap_values(sample)
#    #    p = shap.force_plot(explainer.expected_value, shap_values, sample)
#    #    plt.savefig('shap_force_plot.svg')
#    #    plt.savefig("shap_force_plot.png")
#    #    plt.close()
#    #    return p
#    
#    X_test_preprocessed = model.transform(X_test)
#    sample = X_test_preprocessed[0]
#    shap_values = explainer.shap_values([sample])
#    # shap_value[0][0] is num_likes shap value for the first (and only) sample
#    p = shap.force_plot(explainer.expected_value[0], shap_values[0][0], sample, matplotlib=True, show=False)
#    plt.tight_layout()
#    plt.savefig("reddit_model/shap_force_plot.png")


def upload_model_to_hopsworks(model: Pipeline, X_train, y_train, metrics):
    print("Uploading model to Hopsworks...")
    project = hopsworks.login()

    model_dir="reddit_model"
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)
    joblib.dump(model, model_dir + "/reddit_model.pkl")
    #model.save_model(model_dir + "/reddit_model.json") # xgboost has problems with pickle
    # fig.savefig(model_dir + "/confusion_matrix.png") # TODO?

    # Specify the schema of the model's input/output (names, data types, ...)
    input_schema = Schema(X_train)
    output_schema = Schema(y_train)
    model_schema = ModelSchema(input_schema, output_schema)

    # Register the model in Hopsworks and upload it
    mr = project.get_model_registry()
    reddit_model = mr.python.create_model(
        name="reddit_predict",
        metrics=metrics,
        model_schema=model_schema,
        description="Reddit like and controversiality prediction model."
    )

    # Upload the model to the model registry, including all files in 'model_dir'
    reddit_model.save(model_dir)

# create a function that maps predictions into categories of "no success" = 0-10 likes, "mild success" = 11-100 likes, "great success" = 101-1000 likes, "huge success" = 1000+ likes
def map_predictions(predictions):
    success = []
    for i in range(len(predictions)):
        if predictions[i] < 11:
            success.append(0)
        elif predictions[i] < 101:
            success.append(1)
        elif predictions[i] < 1001:
            success.append(2)
        else:
            success.append(3)
    return success

# create a confusion matrix for map_predictions function
def generate_confusion_matrix(y_test, y_pred):
    print("Creating confusion matrix...")
    if not os.path.isdir("reddit_model"):
        os.mkdir("reddit_model")
    y_test_success = map_predictions(y_test["num_likes"])
    y_pred_success = map_predictions(y_pred[:,0])
    cm = confusion_matrix(y_test_success, y_pred_success)
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=["no success", "mild success", "great success", "huge success"], yticklabels=["no success", "mild success", "great success", "huge success"])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    #fig.savefig("reddit_model/confusion_matrix.png")
    plt.show()