import torch
import pandas as pd
import numpy as np
from src.enefit.utils.utils import (
    norm_temp,
    norm_dew,
    norm_precipitation,
    norm_snowfall,
    norm_windspeed,
    norm_rad,
    get_lag_features,
)


class RunningDataset:
    def __init__(self):
        weights = pd.read_csv(
            "/kaggle/input/davides-transformer-stored/county_pixel_weights.csv"
        )
        weights = weights.sort_values(by=["county", "longitude", "latitude"], axis=0)
        w = weights.weight.to_numpy()
        self.w = w.reshape((16, 112))

        self.means = [
            0.1277779782376821,
            0.26107474891445476,
            0.07862313143698685,
            0.24906541229289986,
            0.15429031000507026,
            0.26429071124835923,
            0.09041434149383237,
            0.6492504504013048,
            0.11064407989990507,
            0.45457957923078396,
            0.08409352756308007,
            0.4617043748429825,
            0.11872232959259765,
            0.628484155104194,
        ]
        self.stds = [
            0.19453651424387805,
            0.10260584476563492,
            0.134744362977315,
            0.09993685767144098,
            0.22255086340003832,
            0.09620643916073567,
            0.16159597866935038,
            0.23916194723685066,
            0.1845046375508081,
            0.17503611656780277,
            0.1486883772037978,
            0.25257191170364235,
            0.18246006143611357,
            0.2447319686556321,
        ]

    def weatherpoints_to_counties(self, f_weather, date_min, date_max):

        data = f_weather.loc[
            (f_weather.forecast_datetime >= date_min),
            [
                "cloudcover_total",
                "temperature",
                "snowfall",
                "direct_solar_radiation",
                "total_precipitation",
                "windspeed",
                "dewpoint",
            ],
        ].iloc[:2688]

        data = data.to_numpy()  # 112 x k
        data = data.reshape((24, 112, 7))
        data = np.einsum("hsv, cs -> hcv", data, self.w).transpose(
            1, 0, 2
        )  # 16 x 24 x k

        data[..., 1] = norm_temp(data[..., 1])
        data[..., 2] = norm_snowfall(data[..., 2])
        data[..., 3] = norm_rad(data[..., 3])
        data[..., 4] = norm_precipitation(data[..., 4])
        data[..., 5] = norm_windspeed(data[..., 5])
        data[..., 6] = norm_dew(data[..., 6])

        return torch.from_numpy(data)

    def __call__(self, to_test, revealed_targets, clients, f_weather):

        avg_temp = 6.4
        if f_weather.temperature.isnull().sum() < len(f_weather):
            avg_temp = f_weather.temperature.mean()

        f_weather.temperature.fillna(avg_temp, inplace=True)
        f_weather.fillna(0, inplace=True)

        f_weather["windspeed"] = (
            f_weather["10_metre_u_wind_component"] ** 2
            + f_weather["10_metre_v_wind_component"] ** 2
        ) ** 0.5
        f_weather = f_weather.sort_values(
            by=["forecast_datetime", "longitude", "latitude"], axis=0
        )
        f_weather.forecast_datetime = pd.to_datetime(f_weather.forecast_datetime)

        to_test.target = 0

        min_date = to_test.datetime.min()

        df = pd.concat([revealed_targets, to_test])

        # revealed_targets.set_index(["product_type", "county", "is_business", "data_block_id", "is_consumption", "datetime"])
        # to_test.set_index(["product_type", "county", "is_business", "data_block_id", "is_consumption", "datetime"])
        # clients.set_index(["product_type", "county", "is_business", "data_block_id", "date"])

        df = df.merge(
            clients,
            on=["product_type", "county", "is_business"],
            how="left",
            suffixes=("", "_client"),
            validate="m:1",
        )

        df["contract"] = df.product_type + df.is_business * 4 - 1

        df["distrib_select"] = df.contract * 2 + df.is_consumption

        df["scaling"] = (df.installed_capacity + 100) ** 0.5
        # 0:(0.12, 0.19), 1:(0.42, 0.24)

        df["mean"] = df.distrib_select.map(lambda x: self.means[x]) * df.scaling
        df["std"] = df.distrib_select.map(lambda x: self.stds[x]) * df.scaling

        df = get_lag_features(df, hours_lag=48)

        rows = df[df.datetime >= min_date].sort_values(
            by=["prediction_unit_id", "datetime", "is_consumption"], axis=0
        )

        max_date = rows.datetime.max()

        forecast_weather = self.weatherpoints_to_counties(f_weather, min_date, max_date)

        ids = rows.shape[0] // 48

        scaling = torch.from_numpy(rows[["mean_lag", "std_lag"]].to_numpy())
        lag_features = torch.from_numpy(rows["target_lag"].to_numpy())
        conditions = rows.loc[
            rows.is_consumption == 0, ["county", "contract", "datetime"]
        ]

        counties = conditions["county"].to_numpy()
        contracts = conditions["contract"].to_numpy()

        months = (conditions.datetime.dt.month - 1).to_numpy()
        dayofweek = conditions.datetime.dt.dayofweek.to_numpy()

        conditions = torch.from_numpy(
            np.stack([counties, contracts, months, dayofweek], -1)
        )
        counties = torch.from_numpy(counties).long()
        contracts = torch.from_numpy(contracts).long()

        scaling = scaling.unflatten(0, (ids, 24, 2))

        lag_features = lag_features.unflatten(0, (ids, 24, 2))

        lag_features = lag_features**0.5

        lag_features[..., 0] = (lag_features[..., 0] - scaling[..., 0, 0]) / scaling[
            ..., 0, 1
        ]
        lag_features[..., 1] = (lag_features[..., 1] - scaling[..., 1, 0]) / scaling[
            ..., 1, 1
        ]

        lag_features = lag_features.flatten(2)

        conditions = conditions.unflatten(0, (ids, 24))[:, 0]
        counties = counties.unflatten(0, (ids, 24))[:, 0]
        forecast_weather = forecast_weather[counties]
        counties_mask = (counties[:, None] - counties[None]).abs() > 0

        contracts = contracts.unflatten(0, (ids, 24))[:, 0]
        contracts_mask = (contracts[:, None] - contracts[None]).abs() > 0

        return (
            (
                lag_features[None],
                conditions[None],
                forecast_weather[None],
                counties_mask[None],
                contracts_mask[None],
            ),
            scaling[None],
            rows.row_id.to_numpy()[None],
        )
