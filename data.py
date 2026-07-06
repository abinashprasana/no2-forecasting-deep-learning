import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

POLLUTANTS = ["CO", "PM10", "NO", "NO2", "NOx", "O3", "PM2.5", "SO2"]


def load_and_clean_data(filepath):
    raw = pd.read_csv(filepath, skiprows=4)
    cols = {
        "Date": raw.columns[0],
        "time": raw.columns[1],
        "CO": raw.columns[2],
        "PM10": raw.columns[5],
        "NO": raw.columns[8],
        "NO2": raw.columns[11],
        "NOx": raw.columns[14],
        "O3": raw.columns[17],
        "PM2.5": raw.columns[20],
        "SO2": raw.columns[23],
    }

    df = raw[list(cols.values())].copy()
    df.columns = list(cols)
    df[POLLUTANTS] = df[POLLUTANTS].apply(pd.to_numeric, errors="coerce")

    midnight = df["time"] == "24:00"
    df.loc[midnight, "time"] = "00:00"
    df.loc[midnight, "Date"] = (
        pd.to_datetime(df.loc[midnight, "Date"], dayfirst=True) + pd.Timedelta(days=1)
    ).dt.strftime("%d-%m-%Y")

    df["datetime"] = pd.to_datetime(
        df["Date"].astype(str) + " " + df["time"].astype(str),
        dayfirst=True,
    )
    df = df.set_index("datetime").drop(columns=["Date", "time"])

    df = df.mask(df < 0)
    df = df.clip(upper=df.quantile(0.99), axis=1)
    df = df.interpolate(method="linear", limit=6, limit_direction="both").dropna()
    return df


def make_sequences(data, target_idx, seq_len=24):
    end = len(data) - seq_len
    X = [data[i : i + seq_len] for i in range(end)]
    y = [data[i + seq_len, target_idx] for i in range(end)]
    return np.array(X), np.array(y)


def prepare_data(df, target_col="NO2", seq_len=24):
    n_train = int(len(df) * 0.70)
    n_val = int(len(df) * 0.15)

    train_df = df.iloc[:n_train]
    val_df = df.iloc[n_train : n_train + n_val]
    test_df = df.iloc[n_train + n_val :]

    scaler = MinMaxScaler()
    train = scaler.fit_transform(train_df)
    val = scaler.transform(val_df)
    test = scaler.transform(test_df)
    target_idx = df.columns.get_loc(target_col)

    return (
        make_sequences(train, target_idx, seq_len),
        make_sequences(val, target_idx, seq_len),
        make_sequences(test, target_idx, seq_len),
        scaler,
        target_idx,
    )
