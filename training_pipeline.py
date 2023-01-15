RUN_ON_MODAL=True
USE_GPU=True
BAYESIAN_SEARCH=True
BAYESIAN_ITERATIONS=7

def g():
    from utils.training import train_model, upload_model_to_hopsworks, generate_prediction_plots, get_metrics, post_process_predictions, generate_shap_summary_plots, generate_confusion_matrix

    model, X_train, X_test, y_train, y_test = train_model(bayesian_search=BAYESIAN_SEARCH, bayesian_n_iterations=BAYESIAN_ITERATIONS, use_gpu=USE_GPU)

    # Note: the model automatically calls transform on all the preprocessing steps and then calls predict on the model
    y_pred = model.predict(X_test)
    y_pred = post_process_predictions(y_pred)

    # Upvote ratio can only be in interval [0,1]
    upvote_ratio_idx = y_test.columns.get_loc("upvote_ratio")
    y_pred[upvote_ratio_idx] = y_pred[upvote_ratio_idx].clip(0, 1)

    output_dir = "reddit_model"

    generate_prediction_plots(y_test, y_pred, output_dir)
    
    metrics = get_metrics(y_test, y_pred)

    generate_shap_summary_plots(model, X_test, output_dir)

    generate_confusion_matrix(y_test, y_pred, output_dir)

    try:
        upload_model_to_hopsworks(model, X_train, y_train, metrics, output_dir)
    except Exception as e:
        import traceback
        import pickle
        print("Could not upload model to hopsworks")
        print(e)
        traceback.print_exc()
        with open("rescue_save.pkl", "wb") as f:
            rescue = [model, X_train, X_test, y_train, y_test, metrics]
            pickle.dump(rescue, f)


import modal
stub = modal.Stub()
image = modal.Image.debian_slim().pip_install(["hopsworks==3.0.5","joblib==1.2.0","pandas==1.3.5","xgboost==1.6.2","scikit-learn==1.2.0","seaborn==0.12.2","praw==7.6.1","bayesian-optimization==1.4.2","shap==0.41.0"])
@stub.function(image=image,
               schedule=modal.Period(days=7),
               secret=modal.Secret.from_name("reddit-predict"),
               mounts=[modal.Mount(remote_dir="/root/utils", local_dir="./utils")],
               timeout=60*60, # 60min timeout
               gpu="any",
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

