import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def _impute_mode_and_convert_bool(x: pd.Series) -> pd.Series:
    return x.fillna('most_frequent') == 'Yes'

def _impute_with_mean_or_median(x: pd.Series) -> pd.Series:
    if np.abs(x.skew()) > 1:
        return x.fillna(x.median())
    return x.fillna(x.mean())

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:

    df_processed = df.copy()
    
    to_bool_cols = ['Stage_fear', 'Drained_after_socializing', 'Personality']
    for col in to_bool_cols:
        df_processed[col] = _impute_mode_and_convert_bool(df_processed[col])
    
    num_cols = [
        'Time_spent_Alone',
        'Social_event_attendance',
        'Going_outside', 
        'Friends_circle_size',
        'Post_frequency'
    ]
    for col in num_cols:
        df_processed[col] = _impute_with_mean_or_median(df_processed[col])
    
    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(df_processed[num_cols])
    
    df_processed[num_cols] = scaled_array

    return df_processed.drop('id', axis=1)