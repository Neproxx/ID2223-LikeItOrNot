import os
import pandas as pd
import numpy as np
import xgboost
import hopsworks
import joblib
import shap
import seaborn as sns
import matplotlib.pyplot as plt
from hsml.schema import Schema
from hsml.model_schema import ModelSchema
from bayes_opt import BayesianOptimization, UtilityFunction
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error, mean_squared_log_error, confusion_matrix
from sklearn.pipeline import Pipeline

# Huggingface UI must import from submodule "main_repo"
#try:
from utils.feature_processing import get_model_pipeline
from utils.feature_validation import validate_preprocessor
#except:
#    from main_repo.utils.feature_processing import get_model_pipeline
#    from main_repo.utils.feature_validation import validate_preprocessor

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

    feature_view = fs.create_feature_view(name="reddit_features",
                                        version=os.getenv("FEATURE_VIEW_VERSION", default=1),
                                        description="Features and labels of the reddit dataset, excluding data that is not relevant for training.",
                                        labels=["num_likes", "upvote_ratio"],
                                        #transformation_functions=transformation_functions,
                                        query=join_query)
    return feature_view

def get_optimal_hyperparameters(n_iterations,use_gpu=False):
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
            get_regressor(
                        n_estimators=int(n_estimators),
                        max_depth=int(max_depth),
                        eta=eta,
                        subsample=subsample,
                        colsample_bytree=colsample_bytree,
                        use_gpu=use_gpu
                        ),
            "tree"
        )

        weights_train = get_weights(y_train)
        model.fit(X_train, y_train, model__sample_weight=weights_train)
        y_pred = model.predict(X_val)
        y_pred = post_process_predictions(y_pred)

        # Weigh num_likes and upvote_ratio 4:1
        weights_val = get_weights(y_val)
        mse_num_likes = mean_squared_error(y_val, y_pred, squared=True, multioutput=[0.8, 0.2], sample_weight=weights_val)
        return -mse_num_likes
    
    # Define parameter space to explore
    pbounds = {'n_estimators': (100, 3000), 'max_depth': (3, 25), 'eta': (0.001, 0.2), 'subsample': (0.5, 1), 'colsample_bytree': (0.5, 1)}
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


def get_weights(y_true):
    """
    Returns the sample weights which correspond to the log of the values.
    """
    y_true = np.abs(y_true.copy().values[:,0])
    return 1 / np.power(y_true + 1, 17/16) # reciprocal of the 17/16 root y^17/16
    #return 1 / np.power(y_true + 1, 9/8) # reciprocal of the 9/8 root y^9/8
    #return 1 / np.power(y_true + 1, 5/4) # reciprocal of the 5/4 root y^5/4
    #return 1 / np.power(y_true + 1, 3/2) # reciprocal of the cube root y^3/2
    #return 1 / (y_true + 1)**2 # reciprocal squared
    # return 1 / (y_true + 1) # reciprocal


def get_regressor(n_estimators, max_depth, eta, subsample, colsample_bytree, use_gpu=False):
    """
    Returns a XGBoost regressor with the given hyperparameters.
    """
    return xgboost.XGBRegressor(
                        n_estimators=n_estimators, 
                        max_depth=max_depth,
                        eta=eta,
                        subsample=subsample,
                        colsample_bytree=colsample_bytree,
                        objective="reg:squarederror",
                        tree_method="gpu_hist" if use_gpu else "hist",
            )



def train_model(bayesian_search=True, bayesian_n_iterations=10, use_gpu=False):
    if bayesian_search:
        results = get_optimal_hyperparameters(n_iterations=bayesian_n_iterations, use_gpu=use_gpu)
        n_estimators, max_depth, eta, subsample, colsample_bytree = results["parameters"]
        X_train, X_test, y_train, y_test = results["data"]
        model = get_model_pipeline(
            get_regressor(n_estimators, max_depth, eta, subsample, colsample_bytree, use_gpu),
            "tree"
        )
    else:
        model = get_model_pipeline(
            get_regressor(
                n_estimators=2250,
                max_depth=4,
                eta=0.067,
                colsample_bytree=0.70,
                subsample=0.88,
                use_gpu=use_gpu
            ),
            "tree"
        )
        print("Loading data...")
        X_train, X_test, y_train, y_test = get_model_features(with_validation_set=False, version=None)
        print("Done")
    validate_preprocessor(model.named_steps["preprocessor"], X_train, "tree")
    print("Starting final training run...")
    weights = get_weights(y_train)
    model.fit(X_train, y_train, model__sample_weight=weights)
    return model, X_train, X_test, y_train, y_test


def post_process_predictions(y_pred):
    """
    Post-processes the predictions of the model to ensure that the predictions are within the valid range.
    """
    # Number of likes can be positive or negative, but upvote ratio can only be between 0 and 1
    # single sample
    if len(y_pred.shape) == 1:
        y_pred[1] = y_pred[1].clip(0, 1)
        return y_pred
    # multiple samples
    y_pred[:, 1] = y_pred[:, 1].clip(0, 1)
    return y_pred


def get_metrics(y_test, y_pred):
    weights = get_weights(y_test)
    rmse = mean_squared_error(y_test, y_pred, multioutput="raw_values", squared=False, sample_weight=weights)
    mae = mean_absolute_error(y_test, y_pred, multioutput="raw_values", sample_weight=weights)
    r2 = r2_score(y_test, y_pred, multioutput="raw_values", sample_weight=weights)
    mape = mean_absolute_percentage_error(y_test, y_pred, multioutput="raw_values", sample_weight=weights)

    # Note: RMSLE is not defined for negative values, so we need to clip the values to be positive
    y_test_copy = y_test.copy()
    y_pred_copy = y_pred.copy()
    y_test_copy[y_test_copy <= 0] = 0
    y_pred_copy[y_pred_copy <= 0] = 0
    rmsle = mean_squared_log_error(y_test_copy, y_pred_copy, multioutput="raw_values", sample_weight=weights)
    
    metrics = {
        "rmse_likes": rmse[0],
        "rmsle_likes": rmsle[0],
        "mae_likes": mae[0],
        "mape_likes": mape[0],
        "r2_likes": r2[0],
        "rmse_upvote_ratio": rmse[1],
        "rmsle_upvote_ratio": rmsle[1],
        "mae_upvote_ratio": mae[1],
        "mape_upvote_ratio": mape[1],
        "r2_upvote_ratio": r2[1]
    }

    print("RMSE num_likes: %.2f" % metrics["rmse_likes"])
    print("RMSLE num_likes: %.2f" % metrics["rmsle_likes"])
    print("MAE num_likes: %.2f" % metrics["mae_likes"])
    print("MAPE num_likes: %.2f" % metrics["mape_likes"])
    print("R2 num_likes: %.2f" % metrics["r2_likes"])
    print("RMSE upvote_ratio: %.2f" % metrics["rmse_upvote_ratio"])
    print("RMSLE upvote_ratio: %.2f" % metrics["rmsle_upvote_ratio"])
    print("MAE upvote_ratio: %.2f" % metrics["mae_upvote_ratio"])
    print("MAPE upvote_ratio: %.2f" % metrics["mape_upvote_ratio"])
    print("R2 upvote_ratio: %.2f" % metrics["r2_upvote_ratio"])
    return metrics


def generate_prediction_plots(y_test, y_pred, output_dir):
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

    fig.savefig(os.path.join(output_dir, "prediction_error.png"))
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    fig.savefig(os.path.join(output_dir, "prediction_error_logscale.png"))
    plt.clf()


def generate_shap_forceplot(model: Pipeline, df_features, output_dir, clear_figure=True):
    """
    :param df_features: DataFrame containing exactly one sample with all the feature values.
    """
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    assert df_features.shape[0] == 1, "df_features must contain exactly one sample"

    explainer = shap.TreeExplainer(model.named_steps["model"])
    feature_names = model.named_steps["preprocessor"].get_feature_names_out()

    X_prep = model[:-1].transform(df_features)
    sample = X_prep[0]

    # explainer.shap_values() outputs an array with dimensions (n_target, n_samples, n_features)
    # Convert from list of arrays to single array
    shap_values = explainer.shap_values(X_prep)
    shap_values = np.concatenate(shap_values, axis=0)
    idx_likes = 0
    idx_upvote_ratio = 1

    ### Raw force plots
    #shap.force_plot(explainer.expected_value[idx_likes], shap_values[idx_likes], sample, feature_names, matplotlib=True, show=False)
    #plt.tight_layout()
    #plt.savefig(os.path.join(output_dir, "shap_force_plot_num_likes.png"))
    #plt.clf()

    #shap.force_plot(explainer.expected_value[idx_upvote_ratio], shap_values[idx_upvote_ratio], sample, feature_names, matplotlib=True, show=False)
    #plt.tight_layout()
    #plt.savefig(os.path.join(output_dir, "shap_force_plot_upvote_ratio.png"))
    #plt.clf()

    ### Compact force plots
    #   Aggregate shap values of embeddings (as sum of values) and plot them as a more compact force plot
    shap_values = shap_values[idx_likes]

    # Get the indices of the features that represent the embedding features
    idx_embedding_text = np.where([feature_name.startswith("embedding_text") for feature_name in feature_names])[0]
    idx_embedding_title = np.where([feature_name.startswith("embedding_title") for feature_name in feature_names])[0]
    idx_embedding_description = np.where([feature_name.startswith("embedding_description") for feature_name in feature_names])[0]

    # Sum the SHAP and feature values of the embeddings
    shap_values_sum_embedding_text = np.sum(shap_values[idx_embedding_text])
    shap_values_sum_embedding_title = np.sum(shap_values[idx_embedding_title])
    shap_values_sum_embedding_description = np.sum(shap_values[idx_embedding_description])
    sample_sum_embedding_text = np.sum(sample[idx_embedding_text])
    sample_sum_embedding_title = np.sum(sample[idx_embedding_title])
    sample_sum_embedding_description = np.sum(sample[idx_embedding_description])

    # Get indices that do not contain the string "embedding"
    idx_not_embedding = np.where([not feature_name.startswith("embedding") for feature_name in feature_names])[0]

    # Create new array with all features where embedding features are aggregated
    feature_names_compact = np.concatenate(
                                [feature_names[idx_not_embedding],
                                ["embedding_text", "embedding_title", "embedding_description"]
                            ])
    shap_values_compact = np.concatenate(
                                [shap_values[idx_not_embedding],
                                    [shap_values_sum_embedding_text, shap_values_sum_embedding_title, shap_values_sum_embedding_description]
                            ])
    sample_compact = np.concatenate(
                                [sample[idx_not_embedding],
                                    [sample_sum_embedding_text, sample_sum_embedding_title, sample_sum_embedding_description]
                            ])
    
    shap.force_plot(explainer.expected_value[idx_likes], shap_values_compact, sample_compact, feature_names_compact, matplotlib=True, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shap_force_plot_num_likes_compact.png"))
    if clear_figure:
        plt.clf()


def generate_shap_summary_plots(model: Pipeline, X, output_dir):
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    explainer = shap.TreeExplainer(model.named_steps["model"])
    feature_names = model.named_steps["preprocessor"].get_feature_names_out()
    X_preprocessed = model[:-1].transform(X)
    shap_values = explainer.shap_values(X_preprocessed)

    ##### Create summary plots with all of the 1000+ features (show only the top 20 features)
    idx_likes = 0
    shap.summary_plot(shap_values[idx_likes], X_preprocessed, feature_names, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shap_summary_plot_num_likes.png"))
    plt.clf()

    idx_upvote_ratio = 1
    shap.summary_plot(shap_values[idx_upvote_ratio], X_preprocessed, feature_names, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shap_summary_plot_upvote_ratio.png"))
    plt.clf()


    ##### Create more compact plots with absolute shap values where embedding features are aggregated (only for num_likes)
    # Average the absolute SHAP values for each feature
    shap_values_abs = np.abs(shap_values[idx_likes])
    shap_values_abs = np.mean(shap_values_abs, axis=0)

    # Get the indices of the features that contain the string "embedding_text", "embedding_title", "embedding_description"
    feature_names = model.named_steps["preprocessor"].get_feature_names_out()
    feature_names = np.array(feature_names)
    idx_embedding_text = np.where([feature_name.startswith("embedding_text") for feature_name in feature_names])[0]
    idx_embedding_title = np.where([feature_name.startswith("embedding_title") for feature_name in feature_names])[0]
    idx_embedding_description = np.where([feature_name.startswith("embedding_description") for feature_name in feature_names])[0]

    # Aggregate the SHAP values for the features that contain the string "embedding_text", "embedding_title", "embedding_description"
    shap_values_abs_sum_embedding_text = np.sum(shap_values_abs[idx_embedding_text])
    shap_values_abs_sum_embedding_title = np.sum(shap_values_abs[idx_embedding_title])
    shap_values_abs_sum_embedding_description = np.sum(shap_values_abs[idx_embedding_description])

    # Get indices that do not contain the string "embedding"
    idx_not_embedding = np.where([not feature_name.startswith("embedding") for feature_name in feature_names])[0]

    # Create new array with all features where embedding features are aggregated
    feature_names_compact = np.concatenate(
                                [feature_names[idx_not_embedding],
                                ["embedding_text", "embedding_title", "embedding_description"]
                            ])
    shap_values_compact = np.concatenate(
                                [shap_values_abs[idx_not_embedding],
                                    [shap_values_abs_sum_embedding_text, shap_values_abs_sum_embedding_title, shap_values_abs_sum_embedding_description]
                            ])
    sns.set()
    plt.barh(feature_names_compact, shap_values_compact)
    plt.xlabel("Average impact of feature\n(embeddings are summed)")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shap_summary_plot_compact.png"))
    plt.xscale("log")
    plt.savefig(os.path.join(output_dir, "shap_summary_plot_compact_log.png"))
    plt.clf()


def map_predictions_to_success(predictions):
    """
    Maps predictions into categories of
    - "no success" = 0-10 likes,
    - "mild success" = 11-100 likes,
    - "great success" = 101-1000 likes,
    - "huge success" = 1000+ likes
    that correspond to the integers 0-3
    """
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


def generate_confusion_matrix(y_test, y_pred, output_dir):
    print("Creating confusion matrix...")
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    y_test_success = map_predictions_to_success(y_test["num_likes"].values)
    y_pred_success = map_predictions_to_success(y_pred[:,0])
    cm = confusion_matrix(y_test_success, y_pred_success)
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=["no success (0-10)", "mild success (11-100)", "great success (101-1000)", "huge success (1000+)"], yticklabels=["no success", "mild success", "great success", "huge success"])
    plt.ylabel('Actual Success')
    plt.xlabel('Predicted Success')
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.clf()


def upload_model_to_hopsworks(model: Pipeline, X_train, y_train, metrics, model_dir="reddit_model"):
    print("Uploading model to Hopsworks...")
    project = hopsworks.login()

    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)
    if os.path.isfile(model_dir + "/reddit_model.pkl"):
        print("Removing old model...")
        os.remove(model_dir + "/reddit_model.pkl")
        print(f"Is removed: {not os.path.isfile(model_dir + '/reddit_model.pkl')}")
    joblib.dump(model, model_dir + "/reddit_model.pkl")
    print(f"Model saved to disk: {os.path.isfile(model_dir + '/reddit_model.pkl')}")

    ### Upload dashboard data to model directory
    # Huggingface has problems reading from Hopsworks due to port issues
    # Therefore, we upload the required data to the model directory here already
    try:
        fs = project.get_feature_store()
        posts_fg = fs.get_feature_group("reddit_posts", version=os.getenv("POSTS_FG_VERSION", default=1))
        users_fg = fs.get_feature_group("reddit_users", version=os.getenv("USERS_FG_VERSION", default=1))
        subreddits_fg = fs.get_feature_group("reddit_subreddits", version=os.getenv("SUBREDDITS_FG_VERSION", default=1))
        full_join = posts_fg.select(features=["post_id", "snapshot_time", "num_likes", "upvote_ratio"]).join(
                            users_fg.select(features=["user_id", "snapshot_time"]), on=["user_id", "snapshot_time"]).join(
                                subreddits_fg.select(features=["subreddit_id", "snapshot_time"]), on=["subreddit_id", "snapshot_time"])
        df_dashboard = full_join.read()
        df_dashboard.to_pickle(os.path.join(model_dir, "df_dashboard.pkl"))
    except Exception as e:
        print(f"Could not upload dashboard data to model directory: {e}")
    ### 

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
