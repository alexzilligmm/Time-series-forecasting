import numpy as np
from sklearn.pipeline import FunctionTransformer
import torch
import pandas as pd

from enefit.utils.utils import get_lag_columns


def get_portion(df, i, j):
    """
    Extracts a portion of the DataFrame based on `data_block_id` boundaries.

    Args:
        df (pd.DataFrame): Input DataFrame containing a 'data_block_id' column.
        i (int): Starting data_block_id (inclusive).
        j (int): Ending data_block_id (inclusive).

    Returns:
        pd.DataFrame: A filtered DataFrame containing rows with data_block_id in [i, j],
                      or -1 if i or j are out of range.
    """
    last_datablock = max(df["data_block_id"])
    if i < 0 or j > last_datablock:
        print(
            "the i-j portion is invalid! try a larger i or smaller j. You have i: ",
            i,
            "j: ",
            j,
            ", and last datablock: ",
            last_datablock,
        )
        return -1

    return df.loc[(i <= df.data_block_id) & (df.data_block_id <= j)]


def get_history_and_target(df, i, j):
    """
    Splits a DataFrame portion into history and target parts based on data_block_id.

    The history contains data from block i to j-2, and the target contains blocks j-1 and j.
    Requires at least 4 blocks (i.e., j - i >= 3).

    Args:
        df (pd.DataFrame): Input DataFrame containing a 'data_block_id' column.
        i (int): Starting data_block_id (inclusive).
        j (int): Ending data_block_id (inclusive).

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (history_portion, target_portion),
                                           or -1 if the range is too narrow or invalid.
    """
    if j - i < 3:
        print("i and j are too close!")
        return -1

    whole_portion = get_portion(df, i, j)
    history_portion = get_portion(whole_portion, i, j - 2)
    target_portion = get_portion(whole_portion, j - 1, j)
    return history_portion, target_portion


def get_gt_tensor(target_portion):
    """
    Converts a 2-day target DataFrame portion into a PyTorch tensor of shape (N, 24, 2).

    Assumes that the 'target' column in the DataFrame represents hourly data over 2 days
    (i.e., 48 values per id), and reshapes it into a tensor of (ids, 24 hours, 2 days).

    Args:
        target_portion (pd.DataFrame): A DataFrame with exactly 2 data_block_id values
                                       and a 'target' column.

    Returns:
        torch.Tensor: A tensor of shape (N, 24, 2), where N is the number of unique ids.
    """
    targets = torch.from_numpy(target_portion.target.to_numpy())
    ids = targets.shape[0] // 48
    targets = targets.unflatten(0, (ids, 24, 2))
    return targets


def get_time_columns(df):
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['Month'] = df['datetime'].dt.month
    df['Hour'] = df['datetime'].dt.hour
    df['Weekday'] = df['datetime'].dt.weekday
    

def refill_nan(df): # for the dataframe for each unit, we replace missing values with the mean value of the same dataframe
    df['production'].fillna(df["production"].mean(), inplace=True)
    df['consumption'].fillna(df["consumption"].mean(), inplace=True)
    df["production_lag_24"].fillna(df["production"].mean(), inplace=True)       
    df["consumption_lag_24"].fillna(df["consumption"].mean(), inplace=True)   
    df["production_lag_25"].fillna(df["production"].mean(), inplace=True)       
    df["consumption_lag_25"].fillna(df["consumption"].mean(), inplace=True)     
    df["production_lag_48"].fillna(df["production"].mean(), inplace=True)       
    df["consumption_lag_48"].fillna(df["consumption"].mean(), inplace=True)     
    df["production_lag_49"].fillna(df["production"].mean(), inplace=True)       
    df["consumption_lag_49"].fillna(df["consumption"].mean(), inplace=True)  
    
def format_history(df, lags, production_means, production_stds, consumption_means, consumption_stds):
    pred_df = df.loc[df.is_consumption == 0]
    cons_column = df.loc[df.is_consumption == 1, 'target']
    pred_df['consumption'] = cons_column.values
    df = pred_df.rename(columns={'target': 'production'})
    
    df.production = ((df.production/(df.installed_capacity +100))**.5 - df.product_type.map(lambda x: production_means[x]))/df.product_type.map(lambda x: production_stds[x])
    df.consumption = ((df.consumption/(df.installed_capacity +100))**.5 - df.product_type.map(lambda x: consumption_means[x]))/df.product_type.map(lambda x: consumption_stds[x])
    
    df = df[["prediction_unit_id","production","consumption","county", "is_business","product_type","eic_count", "installed_capacity","datetime"]]
    get_time_columns(df)
    df = get_lag_columns(df, lags)
    refill_nan(df)
    
    return df

def extract_day_feature(df, col: str):
    # function date unwraps the datetime object in different columns in a dataframe
    datetime = df[col]
    df['month'] = datetime.dt.month
    df['day_of_week'] = datetime.dt.dayofweek 
    df['hour'] = datetime.dt.hour
    return df

def move_columns_to_the_back(df, columns_to_move):
    # function that takes the columns listed and put them in the back of a dataframe    
    other_columns = [col for col in df.columns if col not in columns_to_move]
    desired_column_order = other_columns + columns_to_move
    df = df[desired_column_order]
    return df

def row_to_tensor(row):
    return torch.tensor(row.values, dtype=torch.float32)

def mean_and_std(df, columns=['target', 'target_consumption']):
    df[columns] = np.sqrt(df[columns]).div((df['installed_capacity'] +100) ** 0.5, axis=0)
    grouped = df.groupby(['is_business', 'product_type'])
    mean_variance = grouped[columns].agg(['mean', 'std'])

    return mean_variance

def get_one_hot_encoding(df, column: str):
    one_hot_encoded = pd.get_dummies(df[column], prefix=column).astype(int)
    df = pd.concat([df, one_hot_encoded], axis=1)
    return df


def mean_and_std(df, columns=['target', 'target_consumption']):
    df[columns] = np.sqrt(df[columns]).div((df['installed_capacity'] +100) ** 0.5, axis=0)
    grouped = df.groupby(['is_business', 'product_type'])
    mean_variance = grouped[columns].agg(['mean', 'std'])

    return mean_variance


def sin_transformer(period):
	return FunctionTransformer(lambda x: np.sin(x / period * 2 * np.pi))

def cos_transformer(period):
	return FunctionTransformer(lambda x: np.cos(x / period * 2 * np.pi))

def cyclical_encoding(df, sin_transformer, cos_transformer, column, period):
    df[column + "_sin"] = sin_transformer(period).fit_transform(df[column])
    df[column + "_cos"] = cos_transformer(period).fit_transform(df[column])
    return df