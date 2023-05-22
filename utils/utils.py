#utils.py

import pvlib
from pvlib.pvsystem import PVSystem
from pvlib.location import Location
from pvlib.modelchain import ModelChain
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS
import pandas as pd
import requests


'''Utility functions for the project'''


def physical_profile(row, df_irr):
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


def get_weather_data(lat, lng, start_date, end_date,variables:list):
    
    df_weather = pd.DataFrame()
    for variable in variables:
        response = requests.get('https://archive-api.open-meteo.com/v1/archive?latitude={}&longitude={}&start_date={}&end_date={}&hourly={}'.format(lat, lng, start_date, end_date, variable))
        df = pd.DataFrame(response.json()['hourly'])
        df = df.set_index('time')
        df_weather = pd.concat([df_weather, df], axis=1)

    return df_weather