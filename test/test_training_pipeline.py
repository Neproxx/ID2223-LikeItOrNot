from utils.training import train_model, post_process_predictions
import pytest

local=False
if local:
    from dotenv import load_dotenv
    load_dotenv()

@pytest.mark.parametrize("use_bayesian_search, bayesian_n_iterations", [(True, 1)])
def test_full_training(use_bayesian_search, bayesian_n_iterations):
    model, X_train, X_test, y_train, y_test = train_model(use_bayesian_search, bayesian_n_iterations, use_gpu=False)

    assert model is not None, "model must not be None"
    assert X_train is not None, "X_train must not be None"
    assert X_test is not None, "X_test must not be None"
    assert y_train is not None, "y_train must not be None"
    assert y_test is not None, "y_test must not be None"

    y_pred = model.predict(X_test)
    y_pred = post_process_predictions(y_pred)
    assert y_pred is not None, "y_pred must not be None"

    # Upvote ratio can only be in interval [0,1]
    upvote_ratio_idx = 1
    assert y_pred[:, upvote_ratio_idx].min() >= 0, "upvote ratio must be >= 0"
    assert y_pred[:, upvote_ratio_idx].max() <= 1, "upvote ratio must be <= 1"
