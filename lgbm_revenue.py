import lightgbm as lgb
import mlflow
import pandas as pd
from feast import FeatureStore
import shap

def train_revenue_model(entity_df: pd.DataFrame) -> lgb.Booster:
    store = FeatureStore(repo_path='feature_repo/')
    training_df = store.get_historical_features(
        entity_df=entity_df,
        features=[
            'revenue_features:lag_1w',
            'revenue_features:lag_4w',
            'revenue_features:seasonality_idx',
            'macro_features:pmi_index'
        ]
    ).to_df()

    X = training_df.drop(['revenue_actuals', 'event_timestamp'], axis=1)
    y = training_df['revenue_actuals']

    params = {
        'objective': 'regression',
        'metric': 'mape',
        'num_leaves': 63,
        'learning_rate': 0.05
    }

    with mlflow.start_run(run_name='lgbm_revenue_weekly'):
        model = lgb.train(params, lgb.Dataset(X, y), num_boost_round=500)
        mlflow.lightgbm.log_model(model, 'revenue_model')
    return model
