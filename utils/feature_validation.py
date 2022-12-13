import pandas as pd

def validate_samples(df_users, df_posts, df_subreddits):
    tables = {
        'users': df_users,
        'posts': df_posts,
        'subreddits': df_subreddits
    }
    inconsistencies = dict((k, {}) for k in tables.keys())
    for table_name in tables.keys():
        df = tables[table_name]

        # Check if any column contains NaN values
        if pd.isna(df).any().any():
            nan_columns = df.columns[pd.isna(df).any()].tolist()
            inconsistencies[table_name]["nan_cols"] = nan_columns
    return inconsistencies

def problems_found(inconsistencies):
    for table_name in inconsistencies.keys():
        if len(inconsistencies[table_name].keys()) > 0:
            return True
    return False
