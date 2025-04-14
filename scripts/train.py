
import random
import numpy as np
import pandas as pd
import torch
import tqdm 

from enefit import DEVICE as device
from enefit.models.enemble import Ensemble
from enefit.models.lstm import lstmEncAttDec
from enefit.models.transformer import SelectiveTransformer
from enefit.utils.data_utils import get_gt_tensor, get_history_and_target, get_portion
from enefit.utils.predict import lstm_predict, transformer_predict
from enefit.utils.train_utils import pad_n_cut
from enefit.utils.utils import projection, reproject


def train():
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
    # The actual train data, which is essentially the complement of the test data
    training_df = get_portion(train_df, 0, last_data_block_id-35)
    training_clients = get_portion(client_df, 0, last_data_block_id-35)
    training_forecast_weather = get_portion(forecast_weather_df, 0, last_data_block_id-35)
    
    #luca_model = MyTSFTransformer(embd_size=32,num_heads=2,num_enc_blocks=2,num_dec_layers=2, dropout_p = 0.2, output_size=48, modality="relative").to(device)  # instantiate and load state_dict
    #luca_model.load_state_dict(torch.load("/kaggle/input/davides-transformer-stored/luca_model.pth", map_location=device))

    davide_model = SelectiveTransformer(hid_size=128, cond_size=128, heads=4, n_blocks=5).to(device) # instantiate and load state_dict
    davide_model.load_state_dict(torch.load("/kaggle/input/davides-transformer-stored/davide_L_model.pth", map_location=device), strict=False)

    ale_model = lstmEncAttDec(18, 256, 168).to(device) # instantiate and load state_dict
    ale_model.load_state_dict(torch.load("/kaggle/input/davides-transformer-stored/alessandro_model.pth", map_location=device))


    LR = 1e-2
    WD = 1e-5
    GRAD_ACCU = 8

    ensemble_model = Ensemble(mlp_hidden_size=128).to(device)

    params_to_train = []
    for param in ensemble_model.parameters():
        if param.requires_grad:
            params_to_train.append(param)

    optimizer = torch.optim.AdamW(params_to_train, lr=LR, weight_decay=WD)

    train_ids = training_df.data_block_id.unique().tolist()

    train_ids = train_ids[:-8]

    valid_ids = train_ids[:28]
    train_ids = train_ids[28:]

    torch.random.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    random.shuffle(train_ids)

    EPOCHS = 5

    pbar = tqdm.tqdm()

    train_losses = []
    train_diffs = []

    valid_losses = []
    valid_diffs = []

    best_diff = torch.inf

    for epoch in range(EPOCHS):
        losses = []
        diffs = []

        ensemble_model.train()
        random.shuffle(train_ids)
        
        pbar.set_description_str(f"Train {epoch + 1:02d}", False)
        pbar.reset(len(train_ids))
        
        optimizer.zero_grad()
        steps_accu = 0
        
        for i in train_ids:
            past_train, target_train = get_history_and_target(training_df, i, i+8)
            _, past_client = get_history_and_target(training_clients, i, i+8)
            _,past_forecast_weather = get_history_and_target(training_forecast_weather, i, i+8)
            
            # For nomenclature's sake, we hereby rename the variables to match their names in the API iterator
            to_pass = target_train.copy()
            revealed_targets = past_train
            client = past_client
            forecast_weather = past_forecast_weather
            
            to_pass, revealed_targets, client = pad_n_cut(to_pass, revealed_targets, client, i+8)
            
            target = get_gt_tensor(to_pass).to(device)
            
            with torch.no_grad():
                davide_pred, proj = transformer_predict(davide_model, to_pass, revealed_targets, client, forecast_weather)
                #luca_pred = luca_predict(luca_model, to_pass, revealed_targets, client, forecast_weather)
                ale_pred = lstm_predict(ale_model, to_pass, revealed_targets, client, forecast_weather)
                
                davide_pred = projection(davide_pred, proj)
                #luca_pred = projection(luca_pred, proj)
                ale_pred = projection(ale_pred, proj)
                #target = projection(target, proj)
                
                model_preds = torch.stack((davide_pred, ale_pred), dim=-2).detach() # N x 24 x 3 x 2
                model_preds = model_preds.flatten(2)
                
            
            # TODO normalize before!
            pred = ensemble_model(model_preds)
            pred = reproject(pred, proj)
            
            
            
            loss = (pred - target).abs() # Replace ... with the ground truth tensor
            (loss.mean() / GRAD_ACCU).backward()
            steps_accu += 1
            if steps_accu >= GRAD_ACCU:
                steps_accu = 0
                optimizer.step()
                optimizer.zero_grad()
            
            losses.append(loss.cpu().detach())
            diffs.append((pred- target).abs().cpu().detach())
            
            if len(losses) >= len(train_ids) * .1:
                mean_loss = torch.cat(losses, 0).mean().item()
                torch_diffs = torch.cat(diffs, 0)
                prod_diff = torch_diffs[..., 0].mean().item()
                cons_diff = torch_diffs[..., 1].mean().item()
                mean_diff = torch_diffs.mean().item()
                pbar.set_postfix_str(f"loss={mean_loss:.3f} MAE={mean_diff:.3f}, prod_mae={prod_diff:.3f}, cons_mae={cons_diff:.3f}", False)
                losses.clear()
                diffs.clear()

                train_losses.append(mean_loss)
                train_diffs.append(mean_diff)
            
            pbar.update()
        
        ensemble_model.eval()
        
        with torch.no_grad():
            losses = []
            diffs = []

            for i in valid_ids:
                past_train, target_train = get_history_and_target(training_df, i, i+8)
                _, past_client = get_history_and_target(training_clients, i, i+8)
                _,past_forecast_weather = get_history_and_target(training_forecast_weather, i, i+8)

                # For nomenclature's sake, we hereby rename the variables to match their names in the API iterator
                to_pass = target_train.copy()
                revealed_targets = past_train
                client = past_client
                forecast_weather = past_forecast_weather

                to_pass, revealed_targets, client = pad_n_cut(to_pass, revealed_targets, client, to_pass.data_block_id.max())
                
                target = get_gt_tensor(to_pass).to(device)
                

                davide_pred, proj = transformer_predict(davide_model, to_pass, revealed_targets, client, forecast_weather)
                #luca_pred = luca_predict(luca_model, to_pass, revealed_targets, client, forecast_weather)
                ale_pred = lstm_predict(ale_model, to_pass, revealed_targets, client, forecast_weather)

                davide_pred = projection(davide_pred, proj)
                #luca_pred = projection(luca_pred, proj)
                ale_pred = projection(ale_pred, proj)
                target = projection(target, proj)

                model_preds = torch.stack((davide_pred, ale_pred), dim=-2) # N x 24 x 3 x 2
                model_preds = model_preds.flatten(2)

                pred = ensemble_model(model_preds)

                loss = (pred - target).abs() # Replace ... with the ground truth tensor

                losses.append(loss.cpu())
                diffs.append((reproject(pred, proj) - reproject(target, proj)).abs().cpu())

            mean_loss = torch.cat(losses, 0).mean().item()
            torch_diffs = torch.cat(diffs, 0)
            prod_diff = torch_diffs[..., 0].mean().item()
            cons_diff = torch_diffs[..., 1].mean().item()
            mean_diff = torch_diffs.mean().item()
            print(f"Valid {epoch + 1:02d}: loss={mean_loss:.3f} MAE={mean_diff:.3f}, prod_mae={prod_diff:.3f}, cons_mae={cons_diff:.3f}")
            losses.clear()
            diffs.clear()

            valid_losses.append(mean_loss)
            valid_diffs.append(mean_diff)
            
            if best_diff > mean_diff:
                best_diff = mean_diff
                torch.save(ensemble_model.state_dict(), "/kaggle/working/best_model.pth")