import torch
import tqdm
import pandas as pd

from enefit import DEVICE as device
from enefit.models.enemble import Ensemble
from enefit.models.lstm import lstmEncAttDec
from enefit.models.transformer import SelectiveTransformer
from enefit.models.tsft_transformer import MyTSFTransformer
from enefit.utils.data_utils import get_gt_tensor, get_history_and_target, get_portion
from enefit.utils.predict import lstm_predict, transformer_predict, tsft_predict
from enefit.utils.train_utils import pad_n_cut
from enefit.utils.utils import projection, reproject


def test():
    train_df = pd.read_csv("/kaggle/input/predict-energy-behavior-of-prosumers/train.csv")
    client_df = pd.read_csv("/kaggle/input/predict-energy-behavior-of-prosumers/client.csv")
    forecast_weather_df = pd.read_csv("/kaggle/input/predict-energy-behavior-of-prosumers/forecast_weather.csv")

    # these two business are T/F format in the API, so let's convert them into that format
    client_df['is_business'] = client_df['is_business'].astype(bool)
    train_df['is_business'] = train_df['is_business'].astype(bool)
    train_df['is_consumption'] = train_df['is_consumption'].astype(bool)
    train_df.datetime = pd.to_datetime(train_df.datetime)
    client_df.date = pd.to_datetime(client_df.date)
    forecast_weather_df.forecast_datetime = pd.to_datetime(forecast_weather_df.forecast_datetime)
    forecast_weather_df.origin_datetime = pd.to_datetime(forecast_weather_df.origin_datetime)

    last_data_block_id = max(train_df['data_block_id'])
    testing_df = get_portion(train_df, last_data_block_id-34, last_data_block_id)
    testing_clients = get_portion(client_df, last_data_block_id-34, last_data_block_id)
    testing_forecast_weather = get_portion(forecast_weather_df, last_data_block_id-34, last_data_block_id)


    luca_model = MyTSFTransformer(embd_size=32, num_heads=2, num_enc_blocks=2, num_dec_layers=2, dropout_p = 0.2, output_size=48, modality="relative").to(device)
    luca_model.load_state_dict(torch.load("/kaggle/input/davides-transformer-stored/luca_model_3.pth", map_location=device))
    luca_model.eval()

    davide_model = SelectiveTransformer(hid_size=128, cond_size=128, heads=4, n_blocks=5).to(device) # instantiate and load state_dict
    davide_model.load_state_dict(torch.load("/kaggle/input/davides-transformer-stored/davide_L_model.pth", map_location=device), strict=False)
    davide_model.eval()

    ale_model = lstmEncAttDec(18, 256, 168).to(device) # instantiate and load state_dict
    ale_model.load_state_dict(torch.load("/kaggle/input/davides-transformer-stored/alessandro_model.pth", map_location=device))
    ale_model.eval()

    test_ids = testing_df.data_block_id.unique().tolist()

    test_ids = test_ids[:-8]

    ensemble_model = Ensemble(mlp_hidden_size=128).to(device)
    ensemble_model.load_state_dict(torch.load("/kaggle/input/davides-transformer-stored/best_model.pth", map_location=device))

    ensemble_model.eval()
        
    with torch.no_grad():
        diffs = []
        stuff = {}

        for i in tqdm.tqdm(test_ids):
            past_train, target_train = get_history_and_target(testing_df, i, i+8)
            _, past_client = get_history_and_target(testing_clients, i, i+8)
            _,past_forecast_weather = get_history_and_target(testing_forecast_weather, i, i+8)

            # For nomenclature's sake, we hereby rename the variables to match their names in the API iterator
            to_pass = target_train.copy()
            revealed_targets = past_train
            client = past_client
            forecast_weather = past_forecast_weather

            to_pass, revealed_targets, client = pad_n_cut(to_pass, revealed_targets, client, to_pass.data_block_id.max())
            

            davide_pred, proj = transformer_predict(davide_model, to_pass, revealed_targets, client, forecast_weather)
            luca_pred = tsft_predict(luca_model, to_pass, revealed_targets, client, forecast_weather)
            ale_pred = lstm_predict(ale_model, to_pass, revealed_targets, client, forecast_weather)
            
            avg_pred = .5 * davide_pred + .5 * ale_pred
            
            target = get_gt_tensor(to_pass).to(device)

            davide_pred_pr = projection(davide_pred, proj)
            #luca_pred = projection(luca_pred, proj)
            ale_pred_pr = projection(ale_pred, proj)
            #target = projection(target, proj)

            model_preds = torch.stack((davide_pred_pr, ale_pred_pr), dim=-2) # N x 24 x 3 x 2
            model_preds = model_preds.flatten(2)
            
            ensemble_pred = ensemble_model(model_preds)
            
            ensemble_pred = reproject(ensemble_pred, proj)
            #target = reproject(target, proj).cpu().detach()

            diffs.append((ensemble_pred - target).abs().cpu())
            
            ids = list(to_pass.prediction_unit_id.unique())

            for curr_e, currd, curr_a, curr_l, avg_curr, curr_target, pred_unit_id in zip(ensemble_pred, davide_pred, ale_pred, luca_pred, avg_pred, target, ids):
                if stuff.get(pred_unit_id, None) is None:
                    stuff[pred_unit_id] = [(curr_e, currd, curr_a, curr_l, avg_curr, curr_target)]
                else:
                    stuff[pred_unit_id].append((curr_e, currd, curr_a, curr_l,  avg_curr, curr_target))

        torch_diffs = torch.stack(diffs)
        prod_diff = torch_diffs[..., 0].mean().item()
        cons_diff = torch_diffs[..., 1].mean().item()
        mean_diff = torch_diffs.mean().item()
        print(f"Test: MAE={mean_diff:.3f}, prod_mae={prod_diff:.3f}, cons_mae={cons_diff:.3f}")