# TODO:
# - refactor training pipeline into components that can be tested
# - implement tests here

# Things to test:
# If things that are supposed to have value are not None
# Check that dataset has the right number of rows and columns and is not empty


# tests in the feature pipeline
# uses a test post from reddit
# checks specific values of features sometimes and just checks that they are not None other times

# functions in training are:
# - get_model_features
# - create_feature_view
# - get_opimal_hyperparameters
# - train_model
# - post_process_predictions
# - get_metrics


# in order to import from utils.training we need to add the root directory to the path
import sys
sys.path.append(".")

from utils.training import train_model, upload_model_to_hopsworks, generate_prediction_plots, get_metrics, post_process_predictions

import pytest
import hopsworks
import os
import pandas as pd
import numpy as np


local=True
if local:
    from dotenv import load_dotenv
    load_dotenv()

# test get_optimal_hyperparameters
# TODO test on cloud
@pytest.mark.parametrize("n_iterations", [2, 10, 15])
def test_get_optimal_hyperparameters(n_iterations):
    optimal_hyperparameters = get_optimal_hyperparameters(n_iterations=n_iterations)
    assert optimal_hyperparameters is not None
    assert optimal_hyperparameters["max_depth"] is not None
    assert optimal_hyperparameters["n_estimators"] is not None
    assert optimal_hyperparameters["eta"] is not None
    assert optimal_hyperparameters["subsample"] is not None

# test get_full_dataset
@pytest.mark.parametrize("with_validation", [False, True])
def test_get_model_features(with_validation):
    from dotenv import load_dotenv
    load_dotenv()
    model_features = get_model_features(with_validation_set=with_validation)
    print(model_features)
    assert model_features is not None
    if with_validation:
        assert len(model_features) == 6, f"Expected 6, got {len(model_features)}"
    else:
        assert len(model_features) == 4, f"Expected 4, got {len(model_features)}"


@pytest.mark.parametrize("bayesian_search, bayesian_n_iterations", [(True, 1)])
def test_full_training():
    model, X_train, X_test, y_train, y_test = train_model(bayesian_search=True, bayesian_n_iterations=1)

    assert model is not None
    assert X_train is not None
    assert X_test is not None
    assert y_train is not None
    assert y_test is not None

    y_pred = model.predict(X_test)
    y_pred = post_process_predictions(y_pred)
    assert y_pred is not None

    # Upvote ratio can only be in interval [0,1]
    upvote_ratio_idx = y_test.columns.get_loc("upvote_ratio")

    assert y_pred[upvote_ratio_idx].min() >= 0