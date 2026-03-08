import pandas as pd
from src.data_processing import clean_and_feature_engineer, get_preprocessing_pipeline
from src.model_stacking import StackingModel
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import AdaBoostClassifier

# Configuration
ESTIMATORS_CONFIG = {
    'XGBoost': (
        XGBClassifier(),
        {
            'n_estimators': {'type': 'int', 'low': 100, 'high': 500},
            'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3},
            'max_depth': {'type': 'int', 'low': 3, 'high': 10},
            'subsample': {'type': 'float', 'low': 0.6, 'high': 1.0},
            'colsample_bytree': {'type': 'float', 'low': 0.6, 'high': 1.0}
        }
    ),
    'LightGBM': (
        LGBMClassifier(),
        {
            'n_estimators': {'type': 'int', 'low': 100, 'high': 500},
            'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3},
            'max_depth': {'type': 'int', 'low': 3, 'high': 10},
            'num_leaves': {'type': 'int', 'low': 20, 'high': 100},
            'subsample': {'type': 'float', 'low': 0.6, 'high': 1.0}
        }
    ),
    'AdaBoost': (
        AdaBoostClassifier(),
        {
            'n_estimators': {'type': 'int', 'low': 50, 'high': 500},
            'learning_rate': {'type': 'float', 'low': 0.01, 'high': 1.0},
        }
    )
}

def run_pipeline():
    # Load
    df_train = pd.read_csv('train.csv', index_col='id')
    
    # Process
    df = clean_and_feature_engineer(df_train)
    pipe = get_preprocessing_pipeline()
    X_processed = pipe.fit_transform(df.drop('y', axis=1))
    y = df['y']
    
    # Train
    model = StackingModel(device='cuda', meta_learner='logistic_regression')
    model.add_base_estimators(ESTIMATORS_CONFIG)
    
    for name in ESTIMATORS_CONFIG.keys():
        model.optuna_search(X_processed, y, name, n_trials=20)
        
    model.fit(X_processed, y)
    model.save_model('models/stacking_model.pkl')

if __name__ == "__main__":
    run_pipeline()