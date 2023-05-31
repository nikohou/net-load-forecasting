#utils.py

import numpy as np
import pvlib
from pvlib.pvsystem import PVSystem
from pvlib.location import Location
from pvlib.modelchain import ModelChain
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS
import pandas as pd
import requests
import plotly.express as px
import copy
from darts.metrics import rmse

'''Utility functions for the project'''

# Data collection


def get_weather_data(lat, lng, start_date, end_date,variables:list):
    
    df_weather = pd.DataFrame()
    for variable in variables:
        response = requests.get('https://archive-api.open-meteo.com/v1/archive?latitude={}&longitude={}&start_date={}&end_date={}&hourly={}'.format(lat, lng, start_date, end_date, variable))
        df = pd.DataFrame(response.json()['hourly'])
        df = df.set_index('time')
        df_weather = pd.concat([df_weather, df], axis=1)

    return df_weather



# Data Prep

def drop_duplicate_index(df, axis=0):
    """
    Removes all duplicate columns or index items of a pd.DataFrame. (Keeps first)
    """
    if axis == 0:
        df = df.loc[~df.index.duplicated(keep="last"),:]
    elif axis == 1:
        df = df.loc[:,~df.columns.duplicated(keep="last")]
    else:
        raise ValueError("Make sure axis is either 0 (index) or 1 (column)")

    return df


def infer_frequency(df):
    '''Infers the frequency of a timeseries dataframe and returns the value in minutes'''
    freq = df.index.to_series().diff().mode()[0].seconds / 60
    return freq



def reindex_df(df):
    timesteplen = int(infer_frequency(df))
    past_dates = pd.date_range(df.index[0], df.index[-1], freq = str(timesteplen) + "T")
    df = df.reindex(past_dates) # keep nan drop it in the end
    return df



def drop_days_with_nans(df, threshold):

    """
    
    A function that drops days with more than a certain percentage of NaNs

    Parameters

    df: pd.DataFrame
        The dataframe to be cleaned
    threshold: float
        The percentage of NaNs that a day can have before it is dropped

    Returns

    df_load_cleaned: pd.DataFrame
        The cleaned dataframe

    """
    
    days_to_drop = []
    for day, data in df.groupby(df.index.date):
        entries = data.shape[0] * data.shape[1]
        nans = data.isna().sum().sum()

        nan_ratio = nans/entries

        if nan_ratio > threshold:
            days_to_drop.append(day)

    mask = np.in1d(df.index.date, days_to_drop)
    df_cleaned = df.loc[~mask].sort_index()
    df_cleaned = drop_duplicate_index(df_cleaned, axis=0)
    
    print(f"Dropped {len(days_to_drop)} days with more than {100*threshold}% NaNs")

    return df_cleaned, days_to_drop


def drop_days_with_nans_advances(df, threshold):

    '''

    A function that drops days with more than a certain percentage of NaNs per row

    '''

    days_to_drop = []
    for day, data in df.groupby(df.index.date):

        reduced = data.dropna(axis = 0, thresh=threshold*data.shape[1])

        keep_ratio = reduced.shape[0]/data.shape[0]

        if keep_ratio < (1-threshold): # heuristic
            days_to_drop.append(day)


    mask = np.in1d(df.index.date, days_to_drop)
    df_cleaned = df.loc[~mask].sort_index()
    df_cleaned = drop_duplicate_index(df_cleaned, axis=0)

    print(f"Dropped {len(days_to_drop)} days with more than {100* (1-threshold)}% of the rows had {threshold*100}% NaNs")

    return df_cleaned, days_to_drop


def dropped_days_plotted(df, days_to_drop):

    '''A helper function to 'drop_days_with_nans' that plots the days that were dropped'''

    df_days_to_drop = pd.DataFrame(index = days_to_drop, data = [1]*len(days_to_drop), columns=['drop'])

    df_days_to_drop_full_index = df_days_to_drop.reindex(set(df.index.date)).fillna(0)

    fig = px.line(df_days_to_drop_full_index.sort_index())
    fig.show()

    return df_days_to_drop_full_index




def review_subseries(ts, min_length, ts_cov=None):
    """
    Reviews a list of timeseries, by checking if it is long enough for the model.
    If covariate timeseries are provided, it slices them to the same length as the target timeseries.
    
    Parameters

    ts: list of dart timeseries
        The list of timeseries to be reviewed
    min_length: int
        The minimum length of the timeseries
    ts_cov: list of dart timeseries
        The list of covariate timeseries to be reviewed

    Returns

    ts_reviewed: list of dart timeseries

    """
    ts_reviewed = [] 
    ts_cov_reviewed = []
    for ts in ts:
        if len(ts) > min_length:
            ts_reviewed.append(ts)
            if ts_cov is not None:
                ts_cov_reviewed.append(ts_cov.slice_intersect(ts))
    return ts_reviewed, ts_cov_reviewed




# Meta data scenarios

def physical_profile(row, df_irr):

    """

    This function generates the PV generation profile for a single PV system.

    Parameters

    row: pd.Series
        A row of the meta data dataframe
    df_irr: pd.DataFrame
        The irradiance data with ghi, dni and dhi and temperature

    Returns

    df_results: pd.Series
        The PV generation profile

    """



    idx, latitude, longitude, tilt, azimuth, capacity = row

    temperature_model_parameters = TEMPERATURE_MODEL_PARAMETERS["sapm"][
        "open_rack_glass_glass"
    ]

    location = Location(latitude=latitude, longitude=longitude)

    pvwatts_system = PVSystem(
        surface_tilt=tilt,
        surface_azimuth=azimuth,
        module_parameters={"pdc0": capacity, "gamma_pdc": -0.004},
        inverter_parameters={"pdc0": capacity},
        temperature_model_parameters=temperature_model_parameters,
    )

    mc = ModelChain(
        pvwatts_system, location, aoi_model="physical", spectral_model="no_loss" #these are my model chain assumptions
    )
    mc.run_model(df_irr)
    results = mc.results.ac

    df_results = pd.Series(results)
    df_results.index = df_results.index.tz_localize(None)
    df_results.index.name = "timestamp"
    df_results.name = idx

    return df_results



def rounder(value:int, interval:int):

    """
    Rounding the tilt to interval'th degrees.
    """
    rounded_value = round(value / interval) * interval
    
    return rounded_value


def prepare_metadata_per_scenario(df_netload: pd.DataFrame, df_meta:pd.DataFrame, df_irr:pd.DataFrame, meta_data_scenario:str):

    assert meta_data_scenario in ['META-1', 'META-2', 'META-3', 'META-4', 'META-5'], "meta_data_scenario must be one of ['META-1', 'META-2', 'META-3', 'META-4', 'META-5']"

    list_metas = []
    if meta_data_scenario == 'META-1':
        # omniscient case
        df_meta_ = copy.deepcopy(df_meta)
        list_metas.append(df_meta_)
    
    elif meta_data_scenario == 'META-2':
        # realistic case, the aggregator knows the exact location of the PV system, the tilt and azimuth angle were recorded by the installer (never exact)
        df_meta_ = copy.deepcopy(df_meta)
        df_meta_["tilt"] = df_meta_["tilt"].apply(lambda x: rounder(x, 10))
        df_meta_["azimuth"] = df_meta_["azimuth"].apply(lambda x: rounder(x, 45)) #modifying the azimuth angle based on estimation
        list_metas.append(df_meta_)

    elif meta_data_scenario == 'META-3':
        # the aggregator does not know the exact location of the PV system, so he makes a guess for the general area of the energy community
        df_meta_ = copy.deepcopy(df_meta)
        df_meta_[['latitude', 'longitude']] = df_meta_[['latitude', 'longitude']].mean()
        list_metas.append(df_meta_)

    elif meta_data_scenario == 'META-4':
        # the aggregator does not know the installed dc capacity of the PV systems, so it estimates it from the netload data
        df_meta_ = copy.deepcopy(df_meta)
        estimated_ac_capacity = abs(df_netload.min().values[0]) / df_meta.shape[0]  # equally distributed capacity estimated from the netload data
        df_meta_['estimated_dc_capacity'] = estimated_ac_capacity
        list_metas.append(df_meta_)

    elif meta_data_scenario == 'META-5':
        # the tilt and azimuth were not recorded by the installer, so the aggregator makes a guess, same for the location and the installed capacity
        # will sample random tilts and azimuths from a uniform distribution between 0 and 60 degrees and 0 and 360 degrees respectively, 100 times
        for i in range(10):
            df_meta_ = copy.deepcopy(df_meta)
            random_tilts = np.random.randint(0, 60, (df_meta_.shape[0]))
            random_azimuths = np.random.randint(0, 360, (df_meta_.shape[0]))
            df_meta_["tilt"] = random_tilts
            df_meta_["azimuth"] = random_azimuths
            df_meta_[['latitude', 'longitude']] = df_meta_[['latitude', 'longitude']].mean()
            estimated_ac_capacity = abs(df_netload.min().values[0]) / df_meta.shape[0] # equally distributed capacity estimated from the netload data
            df_meta_['estimated_dc_capacity'] = estimated_ac_capacity
            list_metas.append(df_meta_)

    return list_metas


def pv_generation_forecasts(list_meta_per_scenario, scenarios:list, df_irr):

    """

    This function generates the PV generation forecasts for the different meta data scenarios.

    Parameters

    list_meta_per_scenario: list of pd.DataFrames
        The list of meta data per scenario
    scenarios: list of str
        The list of scenarios to generate the forecasts for
    df_irr: pd.DataFrame
        The irradiance data

    Returns

    dict_pv_forecasts: dict
        A dictionary with the PV generation forecasts per scenario

    """
    
    dict_pv_forecasts = {}
    for meta_scen in scenarios:
        print(f'Generating PV forecasts for scenario: {meta_scen}')
        list_meta = list_meta_per_scenario[meta_scen]
        pv_forecast_per_scenario = []
        for df_meta in list_meta:
            df_pv_forecast = df_meta.apply(lambda x: physical_profile(x, df_irr), axis=1).T.sum(axis=1).to_frame('pv_forecast')
            pv_forecast_per_scenario.append(df_pv_forecast)
        
        pv_forecast_per_scenario = pd.concat(pv_forecast_per_scenario, axis=1).mean(axis=1).to_frame(f'pv_forecast_{meta_scen}')
        dict_pv_forecasts[meta_scen] = pv_forecast_per_scenario

    return dict_pv_forecasts



# Model training


class Config:

    '''
    
    Class to store config parameters, to circumvent the wandb.config when combining multiple models.
    
    '''

    def __init__(self):
        self.data = {}

    def __getattr__(self, key):
        if key in self.data:
            return self.data[key]
        else:
            raise AttributeError(f"'Config' object has no attribute '{key}'")

    def __setattr__(self, key, value):
        if key == 'data':
            # Allow normal assignment for the 'data' attribute
            super().__setattr__(key, value)
        else:
            self.data[key] = value

    def __delattr__(self, key):
        if key in self.data:
            del self.data[key]
        else:
            raise AttributeError(f"'Config' object has no attribute '{key}'")

    def __len__(self):
        return len(self.data)

    def keys(self):
        return self.data.keys()

    def values(self):
        return self.data.values()

    def items(self):
        return self.data.items()

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    @classmethod
    def from_dict(cls, data):
        config = cls()
        for key, value in data.items():
            if isinstance(value, dict):
                value = cls.from_dict(value)  # Recursively convert nested dictionaries
            setattr(config, key, value)
        return config



def build_config(config_dataset):

    '''
    
    Takes a config_dataset dictionary and builds a config object from it, deriving the rest of the parameters from the config_dataset.

    '''

    config = Config().from_dict(config_dataset)
    config.temp_resolution = 15 # in minutes
    config.horizon_in_hours = 24 + 36 if config.METER == '2' else 36 # in hours, 24 for the data gap in METER-2 and 36 for the day-ahead forecast horizon

    config.timestep_encoding = ["hour", "minute"] if config.temp_resolution == 1 else ['quarter']
    config.datetime_encoding =  {
                        "cyclic": {"future": config.timestep_encoding}, 
                        "position": {"future": ["relative",]},
                        "datetime_attribute": {"future": ["dayofweek", "week"]},
                        'position': {'future': ['relative']},
                } if config.use_datetime_encoding else None

    config.timesteps_per_hour = int(60 / config.temp_resolution)
    config.n_lags = config.lookback_in_hours * config.timesteps_per_hour
    config.n_ahead = config.horizon_in_hours * config.timesteps_per_hour
    config.eval_stride = int(np.sqrt(config.n_ahead)) # evaluation stride, how often to evaluate the model, in this case we evaluate every n_ahead steps
    
    return config


def predict_testset(config, model, ts, ts_covs, pipeline):

    historics = model.historical_forecasts(ts, 
                                        future_covariates= ts_covs,
                                        start=ts.get_index_at_point(config.n_lags),
                                        verbose=False,
                                        stride=config.eval_stride, 
                                        forecast_horizon=config.n_ahead, 
                                        retrain=False, 
                                        last_points_only=False, # leave this as False unless you want the output to be one series, the rest will not work with this however
                                        )

    if config['METER'] == '2':
        historics = [historic[(config.timesteps_per_hour*24):] for historic in historics] # since in METER 2 we always have to wait 24 hours for the data

    historics_gt = [ts.slice_intersect(historic) for historic in historics]
    
    scores = {}
    for metric in config.eval_metrics:
        score = np.array(metric(historics_gt, historics)).mean()
        scores[metric.__name__] = score

    ts_predictions = ts_list_concat(historics, config.eval_stride) # concatenating the batches into a single time series for plot 1, this keeps the n_ahead
    ts_predictions_inverse = pipeline.inverse_transform(ts_predictions) # inverse transform the predictions, we need the original values for the evaluation
    
    return ts_predictions_inverse.pd_series().to_frame('prediction'), scores




# ML Eval

def get_longest_subseries_idx(ts_list):
    """
    Returns the longest subseries from a list of darts TimeSeries objects and its index
    """
    longest_subseries_length = 0
    longest_subseries_idx = 0
    for idx, ts in enumerate(ts_list):
        if len(ts) > longest_subseries_length:
            longest_subseries_length = len(ts)
            longest_subseries_idx = idx
    return longest_subseries_idx


def ts_list_concat(ts_list, eval_stride):
    '''
    This function concatenates a list of time series into one time series.
    The result is a time series that concatenates the subseries so that n_ahead is preserved.
    
    '''
    ts = ts_list[0]
    n_ahead = len(ts)
    skip = n_ahead // eval_stride
    for i in range(skip, len(ts_list)-skip, skip):
        ts_1 = ts_list[i][ts.end_time():]
        ts = ts[:-1].append(ts_1)
    return ts


