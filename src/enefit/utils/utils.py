# src/enefit/utils/utils.py
import numpy as np
import torch
import pandas as pd


def norm_temp(data):
    return (data - 5.74) / 7.84


def norm_dew(data):
    return (data - 2.41) / 7.12


def norm_precipitation(data):
    data = 112.8 * ((np.maximum(data, 0)) ** 0.5)
    return data


def norm_snowfall(data):
    data = 198.8 * ((np.maximum(data, 0)) ** 0.5)
    return data


def norm_windspeed(data):
    return (data**0.5 - 2.21) / 0.61


def norm_rad(data):
    return 0.5 * np.log(np.maximum(data, 0) + 1)


def unnormalize_tensor(inputs, tensor, production_stds, production_means, consumption_stds, consumption_means):
    unnormalized_prod = []
    unnormalized_cons = []
      
    units_num, seq_len, target = tensor.size()

    for unit in range(units_num):
        county = inputs[unit,0,2].long().item()
        is_business = inputs[unit,0,3].long().item()
        product_type = inputs[unit,0,4].long().item()
        installed_capacity = inputs[unit,0,6].long()
        prod_tensor = tensor[unit,:,0]
        cons_tensor = tensor[unit,:,1]
        
        unnormalized_prod_tensor = ((prod_tensor * production_stds[int(product_type)] + production_means[int(product_type)]).unsqueeze(0)) ** 2 * (installed_capacity+100)
        unnormalized_cons_tensor = ((cons_tensor * consumption_stds[int(product_type)] + consumption_means[int(product_type)]).unsqueeze(0)) ** 2 * (installed_capacity+100)
        unnormalized_prod.append(unnormalized_prod_tensor)
        unnormalized_cons.append(unnormalized_cons_tensor)
        
    unnormalized_prod = torch.cat(unnormalized_prod, dim=0)
    unnormalized_cons = torch.cat(unnormalized_cons, dim=0)
    return unnormalized_prod, unnormalized_cons


def get_lag_features(df, hours_lag=48):
    lag_df = df.copy()
    lag_df.datetime = lag_df.datetime + pd.Timedelta(hours=hours_lag)

    df = df.merge(
        lag_df,
        how="left",
        on=["prediction_unit_id", "datetime", "is_consumption"],
        suffixes=("", "_lag"),
        validate="1:1",
    )

    df.target.fillna(0, inplace=True)
    df.dropna()

    to_keep = {
        "target_lag",
        "county",
        "contract",
        "datetime",
        "prediction_unit_id",
        "is_consumption",
        "mean_lag",
        "std_lag",
    }

    to_remove = set(df.columns).difference(to_keep)

    df.drop(to_remove, axis=1)

    return df

# slightly different from the one above
def get_lag_columns(df, lags):
    
    to_keep = set(df.columns)
    
    for hours in lags:
        lag_df = df.copy()
        lag_df.datetime = lag_df.datetime + pd.Timedelta(hours=hours)
        
        df = df.merge(lag_df, how="left", on=["prediction_unit_id", "datetime"], suffixes=("", "_lag_" + str(hours)), validate="1:1")

        to_keep.add("production_lag_" + str(hours))
        to_keep.add("consumption_lag_" + str(hours))

        to_remove = set(df.columns).difference(to_keep)

        df.drop(to_remove, axis=1, inplace=True)
    
    return df

def get_lag_feature(df, lags):

    to_keep = {"target_consumption", "target", "contract", "datetime", "prediction_unit_id", "is_business", "product_type", "data_block_id", "installed_capacity"}
    
    for hours in lags:
        lag_df = df.copy()
        lag_df.datetime = lag_df.datetime + pd.Timedelta(hours=hours)
        
        df = df.merge(lag_df, how="left", on=["prediction_unit_id", "datetime"], suffixes=("", "_" + str(hours) + "_lag"), validate="1:1")
        
        df.fillna(0, inplace=True)

        to_keep.add("target_" + str(hours) + "_lag")
        to_keep.add("target_consumption_" + str(hours) + "_lag")

        to_remove = set(df.columns).difference(to_keep)

        df.drop(to_remove, axis=1, inplace=True)
    
    return df


def reproject(projected, scaling):
    projected[..., 0] = projected[..., 0] * scaling[..., 0, 1] + scaling[..., 0, 0]
    projected[..., 1] = projected[..., 1] * scaling[..., 1, 1] + scaling[..., 1, 0]
    projected = projected ** 2
    return torch.relu(projected)

def projection(unscaled, scaling):
    unscaled = unscaled ** .5
    unscaled[..., 0] = (unscaled[..., 0] - scaling[..., 0, 0]) / scaling[..., 0, 1]
    unscaled[..., 1] = (unscaled[..., 1] - scaling[..., 1, 0]) / scaling[..., 1, 1]
    return unscaled


def de_proj(data, proj):
    mean, std, scaling = proj
    return torch.mul(torch.add(torch.mul(data, std.unsqueeze(1)), mean.unsqueeze(1)), scaling.unsqueeze(1)) **2   