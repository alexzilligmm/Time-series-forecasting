import itertools

import pandas as pd


def pad_n_cut(to_pass, revealed_targets, client, data_block_id):
    to_pass = to_pass.loc[to_pass.data_block_id == data_block_id]
    to_pass.target.fillna(0, inplace=True)
    
    groups = list(to_pass.groupby(['prediction_unit_id', "county", "is_business", "product_type"]).groups.keys())
    groups = list(zip(*groups))
    dates = revealed_targets.datetime.unique()

    data_block_ids = []
    for dbi in range(revealed_targets.data_block_id.min(), revealed_targets.data_block_id.max()+1):
        data_block_ids += [dbi] * 24
    
    n_client_df = pd.DataFrame(list(zip(groups[1], groups[2], groups[3], [data_block_id] * len(groups[1]))), columns=['county', 'is_business', 'product_type', "data_block_id"])
    client_date = client.date.unique()[-1]
    client = pd.merge(n_client_df, client,on=['county', 'is_business', 'product_type', "data_block_id"], how='left')
    client.date.fillna(client_date, inplace=True)
    client.fillna(0, inplace=True)
    
    
    new_df = pd.DataFrame(itertools.product(dates, list(range(len(groups[0])))), columns =['datetime', 'coso'])
    new_df["prediction_unit_id"] = new_df.coso.map(lambda x: groups[0][x])
    new_df["county"] = new_df.coso.map(lambda x: groups[1][x])
    new_df["is_business"] = new_df.coso.map(lambda x: groups[2][x])
    new_df["product_type"] = new_df.coso.map(lambda x: groups[3][x])
    new_df["data_block_id"] = 0
    new_df.drop("coso", axis=1, inplace=True)

    new_df.datetime = pd.to_datetime(new_df.datetime)
    
    for unit in new_df.prediction_unit_id.unique():
        new_df.loc[new_df.prediction_unit_id == unit, "data_block_id"] = data_block_ids 

    new_df["is_consumption"] = False
    other_df = new_df.copy()
    other_df["is_consumption"] = True

    new_df = pd.concat([new_df, other_df])
    
    revealed_targets = pd.merge(new_df,revealed_targets,on=['datetime','prediction_unit_id', "county", "is_business", "product_type", "is_consumption", "data_block_id"], how='left')
    revealed_targets.fillna(0, inplace=True)
    
    to_pass.sort_values(by=["prediction_unit_id", "datetime", "is_consumption"], axis=0, inplace=True)
    revealed_targets.sort_values(by=["prediction_unit_id", "datetime", "is_consumption"], axis=0, inplace=True)

    return to_pass, revealed_targets, client