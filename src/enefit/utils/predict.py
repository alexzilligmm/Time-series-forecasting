import torch
import pandas as pd

from enefit import DEVICE
from enefit.data.days_dataset import DaysDataset
from enefit.utils.data_utils import cos_transformer, cyclical_encoding, extract_day_feature, format_history, move_columns_to_the_back, sin_transformer
from enefit.utils.utils import de_proj, get_lag_feature, reproject, unnormalize_tensor

def transformer_predict(model, test, revealed_targets, client, forecast_weather, running_dataset):
    
    test = test.copy()
    revealed_targets = revealed_targets.copy()
    client = client.copy()
    forecast_weather = forecast_weather.copy()
    
    data_block_id = test.data_block_id.unique()[0]
    revealed_targets = revealed_targets.loc[revealed_targets.data_block_id == revealed_targets.data_block_id.max()]
    forecast_weather = forecast_weather.loc[forecast_weather.data_block_id == data_block_id]
    
    data, proj, row_ids = running_dataset(test, revealed_targets, client, forecast_weather)

    x = data[0].to(device=DEVICE, dtype=torch.float32)
    cond = data[1].to(device=DEVICE)
    weather = data[2].to(device=DEVICE, dtype=torch.float32)
    county_mask = data[3].to(device=DEVICE)
    contract_mask = data[4].to(device=DEVICE)
    proj = proj.to(device=DEVICE, dtype=torch.float32)

    pred = model(x, cond, weather, county_mask, contract_mask)
    pred = reproject(pred, proj).detach()[0]
    #pred = torch.nan_to_num(pred)
        
    # Stuff will be out-of-order!
    #pred = dict(zip(row_ids[0].tolist(), pred.tolist())) # row_ids might be out of order
    
    return pred, proj[0]



def tsft_predict(model, test, revealed_targets, client, forecast_weather):
        
    test = test.copy()
    revealed_targets = revealed_targets.copy()
    client = client.copy()
    
    client_targets = pd.merge(revealed_targets, client, on=["county","is_business","product_type"], how="left")
    del client_targets["date"]
    del client_targets["row_id"]
    client_targets['is_business'] = client_targets['is_business'].astype(int)
    client_targets['is_consumption'] = client_targets['is_consumption'].astype(int)
    
    client_targets = format_history(client_targets)
    
    client_targets.sort_values(by=["prediction_unit_id", "datetime"], axis=0, inplace=True)
    
    client_targets.drop(["prediction_unit_id", "datetime"], axis=1, inplace=True)
    
    test_tensor = torch.from_numpy(client_targets.to_numpy()).float()
    ids = test_tensor.shape[0] // 168
    test_tensor = test_tensor.unflatten(0, (ids, 168))
    
    with torch.no_grad():
        outputs = model(test_tensor.to(DEVICE))
        outputs = outputs[:,-24:,:]

    unnormalized_pred_prod, unnormalized_pred_cons = unnormalize_tensor(test_tensor,outputs)
    unnormalized_pred_prod, unnormalized_pred_cons = torch.relu(unnormalized_pred_prod), torch.relu(unnormalized_pred_cons)
    unnormalized_pred = torch.stack((unnormalized_pred_prod, unnormalized_pred_cons), dim=-1)
    return unnormalized_pred


def lstm_predict(model, test, revealed_targets, client, forecast_weather):
    
    test = test.copy()
    revealed_targets = revealed_targets.copy()
    df_client = client.copy()
    
    df_train = pd.concat([revealed_targets, test])
    
    duplicated_rows = df_train.loc[df_train.is_consumption == 1, "target"]
    df_train = df_train.loc[df_train.is_consumption == 0]
    df_train["target_consumption"] = duplicated_rows.values
    
    df_client.drop("data_block_id", axis=1, inplace=True)
    
    merged_df = pd.merge(df_train, df_client, on=['product_type', 'county', 'is_business'], how='left')

    del merged_df['row_id']
    del merged_df['is_consumption']
    del merged_df['date']
    del merged_df['eic_count']


    pred_unit_id = merged_df.groupby('prediction_unit_id')
    for name, group in pred_unit_id:
        means = group['installed_capacity'].mean()
        merged_df.loc[merged_df['prediction_unit_id'] == name, 'installed_capacity'].fillna(means, inplace=True)


    lags = [24, 48, 72, 96, 120]
    
    merged_df = get_lag_feature(merged_df, lags=lags)
        
    merged_df = extract_day_feature(merged_df, 'datetime')
    
    merged_df = cyclical_encoding(merged_df.copy(), sin_transformer, cos_transformer, 'month', 12)
    merged_df = cyclical_encoding(merged_df.copy(), sin_transformer, cos_transformer, 'day_of_week', 7)
    merged_df = cyclical_encoding(merged_df.copy(), sin_transformer, cos_transformer, 'hour', 24)
    
    del merged_df['datetime']

    lags.reverse()
    for lag in lags:
        merged_df = move_columns_to_the_back(merged_df, ["target" + "_" + str(lag) + "_lag", "target_consumption" + "_" + str(lag) + "_lag"])

    merged_df = move_columns_to_the_back(merged_df, ['target', 'target_consumption'])
    
    del merged_df['data_block_id']
    del merged_df['month']
    del merged_df['hour']
    del merged_df['day_of_week']
    
    merged_df = merged_df.loc[:, ['is_business', 'product_type', 'prediction_unit_id',
       'installed_capacity', 'month_sin', 'month_cos', 'day_of_week_sin',
       'day_of_week_cos', 'hour_sin', 'hour_cos', 'target_120_lag',
       'target_consumption_120_lag', 'target_96_lag',
       'target_consumption_96_lag', 'target_72_lag',
       'target_consumption_72_lag', 'target_48_lag',
       'target_consumption_48_lag', 'target_24_lag',
       'target_consumption_24_lag', 'target', 'target_consumption']]
    
    data = DaysDataset(merged_df, lags, 168)
    
    inputs = []
    scalings = []

    for input, scaling in data:
        inputs.append(input)
        scalings.append(scaling)

    input = torch.stack(inputs).to(DEVICE)
    mean = torch.stack([scaling[0] for scaling in scalings], 0)
    std = torch.stack([scaling[1] for scaling in scalings], 0)
    scaling = torch.stack([scaling[2] for scaling in scalings], 0)
    scaling = [sl.to(DEVICE) for sl in [mean, std, scaling]]
    
    input = input[:, :168, :]

    output = model(input, 24)
    output = de_proj(output, scaling)
    
    return output