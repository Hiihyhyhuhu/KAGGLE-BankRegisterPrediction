import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def clean_and_feature_engineer(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    # 1. Age Clustering
    kmeans = KMeans(n_clusters=4, random_state=42, n_init='auto')
    df['age'] = kmeans.fit_predict(df[['age']])
    
    # 2. Binary Mapping
    df['contact'] = df['contact'].map({'cellular': 1, 'telephone': 1, 'unknown': 0})
    df['poutcome'] = df['poutcome'].map({'success': 1, 'other': 0, 'failure': -1, 'unknown': 0})
    
    for col in ['default', 'housing', 'loan']:
        df[col] = df[col].map({'yes': 1, 'no': -1, 'unknown': 0})
    
    # 3. Loan consolidation & Duration
    df['loan'] = ((df['housing'] == 1) | (df['loan'] == 1) | (df['default'] == 1)).astype(int)
    df['duration_per_contact'] = (df['duration'] / (df['campaign'] + df['previous'])).replace([np.inf, -np.inf], 0)
    
    # 4. Education & Time Features
    df['education'] = df['education'].map({'primary': 1, 'secondary': 2, 'tertiary': 3, 'unknown': 0})
    df['day_category'] = pd.qcut(df['day'], q=3, labels=['early', 'mid', 'late'])
    df['day_month'] = df['day_category'].astype(str) + '_' + df['month']
    
    return df.drop(columns=['default', 'housing', 'campaign', 'previous', 'duration', 'day', 'month', 'day_category'])

def get_preprocessing_pipeline():
    categorical_cols = ['marital', 'job', 'day_month']
    numerical_cols = ['balance', 'pdays', 'duration_per_contact']
    
    return ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
            ('num', StandardScaler(), numerical_cols),
        ],
        remainder='passthrough'
    )