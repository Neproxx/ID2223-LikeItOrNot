RUN_ON_MODAL=False

def g():
    from utils.training import train_model, upload_model_to_hopsworks, generate_prediction_plots, get_metrics, post_process_predictions
    
    # TODO: Try to bin the number of likes into ranges and try to predict the range instead of the exact number

    model, X_train, X_test, y_train, y_test = train_model(bayesian_search=True)

    # Note: the model automatically calls transform on all the preprocessing steps and then calls predict on the model
    y_pred = model.predict(X_test)
    y_pred = post_process_predictions(y_pred)

    # Upvote ratio can only be in interval [0,1]
    upvote_ratio_idx = y_test.columns.get_loc("upvote_ratio")
    y_pred[upvote_ratio_idx] = y_pred[upvote_ratio_idx].clip(0, 1)

    generate_prediction_plots(y_test, y_pred)
    
    metrics = get_metrics(y_test, y_pred)

    # TODO: General shap evaluation

    upload_model_to_hopsworks(model, X_train, y_train, metrics)

    # TODO: how does bayesian handle multi output? Better to just give it the output of the num_likes prediction?


import modal
stub = modal.Stub()
image = modal.Image.debian_slim().pip_install(["hopsworks","joblib","pandas","xgboost","scikit-learn","seaborn","praw","bayesian-optimization","shap"])
@stub.function(image=image,
               schedule=modal.Period(days=1),
               secret=modal.Secret.from_name("reddit-predict"),
               mounts=[modal.Mount(remote_dir="/root/utils", local_dir="./utils")],
               timeout=60*60, # 1h timeout
               retries=1
               )
def f():
    g()

if __name__ == "__main__":
    if RUN_ON_MODAL:
        with stub.run():
            f.call()
    else:
        # NOTE: Create an .env file in the root directory if you want to run this locally
        from dotenv import load_dotenv
        load_dotenv()
        g()

