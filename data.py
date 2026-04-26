import numpy as np
import pandas as pd
import warnings
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings('ignore')

def load_and_clean_data(filepath):
    # the DEFRA CSV has 4 metadata rows before the actual header
    df_raw = pd.read_csv(filepath, skiprows=4)
    
    # each pollutant takes 3 columns: [value, status, unit]
    key_cols = {
        'Date'  : df_raw.columns[0],
        'time'  : df_raw.columns[1],
        'CO'    : df_raw.columns[2],   # Carbon monoxide
        'PM10'  : df_raw.columns[5],   # PM10 particulate matter
        'NO'    : df_raw.columns[8],   # Nitric oxide
        'NO2'   : df_raw.columns[11],  # Nitrogen dioxide (target)
        'NOx'   : df_raw.columns[14],  # Nitrogen oxides (as NO2)
        'O3'    : df_raw.columns[17],  # Ozone
        'PM2.5' : df_raw.columns[20],  # PM2.5 particulate matter
        'SO2'   : df_raw.columns[23],  # Sulphur dioxide
    }
    
    df_raw = df_raw[[v for v in key_cols.values()]].copy()
    df_raw.columns = list(key_cols.keys())
    
    # convert all pollutant columns to numeric
    for col in ['CO','PM10','NO','NO2','NOx','O3','PM2.5','SO2']:
        df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')
        
    df = df_raw.copy()
    
    # fix DEFRA-specific issue: '24:00' means end-of-day midnight
    mask = df['time'] == '24:00'
    df.loc[mask, 'time'] = '00:00'
    df.loc[mask, 'Date'] = (
        pd.to_datetime(df.loc[mask, 'Date'], dayfirst=True) + pd.Timedelta(days=1)
    ).dt.strftime('%d-%m-%Y')
    
    # 1 — build datetime index
    df['datetime'] = pd.to_datetime(
        df['Date'].astype(str) + ' ' + df['time'].astype(str),
        dayfirst=True
    )
    df.set_index('datetime', inplace=True)
    df.drop(columns=['Date', 'time'], inplace=True)
    
    # 2 — replace negatives with NaN (instrument artefacts)
    num_cols = df.columns.tolist()
    for col in num_cols:
        df.loc[df[col] < 0, col] = np.nan
        
    # 3 — clip extreme outliers at 99th percentile per column
    for col in num_cols:
        upper = df[col].quantile(0.99)
        df[col] = df[col].clip(upper=upper)
        
    # 4 — linear interpolation for short gaps
    df.interpolate(method='linear', limit=6, limit_direction='both', inplace=True)
    df.dropna(inplace=True)
    
    return df

def make_sequences(data, target_idx, seq_len=24):
    """
    sliding window — creates (X, y) pairs for sequence modelling
    each sample uses seq_len past hours to predict the next hour's target
    """
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i : i + seq_len, :])
        y.append(data[i + seq_len, target_idx])
    return np.array(X), np.array(y)

def prepare_data(df, target_col='NO2', seq_len=24):
    """
    chronological 70 / 15 / 15 split — no shuffling for time series
    fits scaler on training data only to avoid data leakage
    """
    n        = len(df)
    n_train  = int(n * 0.70)
    n_val    = int(n * 0.15)
    
    train_df = df.iloc[:n_train]
    val_df   = df.iloc[n_train : n_train + n_val]
    test_df  = df.iloc[n_train + n_val:]
    
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_df)
    val_scaled   = scaler.transform(val_df)
    test_scaled  = scaler.transform(test_df)
    
    target_idx = df.columns.get_loc(target_col)
    
    X_train, y_train = make_sequences(train_scaled, target_idx, seq_len)
    X_val,   y_val   = make_sequences(val_scaled, target_idx, seq_len)
    X_test,  y_test  = make_sequences(test_scaled, target_idx, seq_len)
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler, target_idx
