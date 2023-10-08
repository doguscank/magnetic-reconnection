import json
import pandas as pd
import numpy as np
import os
import re

# Extracted fill values for each column from your data description
fill_values = {
    "Bartels rotation number": 9999,
    "Scalar B, nT": 999.9,
    "Vector B Magnitude,nT": 999.9,
    "Lat. Angle of B (GSE)": 999.9,
    "Long. Angle of B (GSE)": 999.9,
    "BX, nT (GSE, GSM)": 999.9,
    "BY, nT (GSE)": 999.9,
    "BZ, nT (GSE)": 999.9,
    "BY, nT (GSM)": 999.9,
    "BZ, nT (GSM)": 999.9,
    "RMS_magnitude, nT": 999.9,
    "RMS_field_vector, nT": 999.9,
    "RMS_BX_GSE, nT": 999.9,
    "RMS_BY_GSE, nT": 999.9,
    "RMS_BZ_GSE, nT": 999.9,
    "SW Plasma Temperature, K": 9999999.0,
    "SW Proton Density, Ncm^3": 999.9,
    "SW Plasma Speed, kms": 9999.0,
    "SW Plasma flow long. angle": 999.9,
    "SW Plasma flow lat. angle": 999.9,
    "AlphaProt. ratio": 9.999,
    "Flow pressure": 99.99,
    "sigma-T,K": 9999999.0,
    "sigma-n, Ncm^3": 999.9,
    "sigma-V, kms": 9999.0,
    "sigma-phi V, degrees": 999.9,
    "sigma-theta V, degrees": 999.9,
    "sigma-ratio": 9.999,
    "E elecrtric field": 999.99,
    "Plasma Beta": 999.99,
    "Alfen mach number": 999.9,
    "Kp index": 99,
    "R (Sunspot No.)": 999,
    "DST Index": 99999,
    "AE-index, nT": 9999,
    "Proton flux (>1 Mev)": 999999.99,
    "Proton flux (>2 Mev)": 99999.99,
    "Proton flux (>4 Mev)": 99999.99,
    "Proton flux (>10 Mev)": 99999.99,
    "Proton flux (>30 Mev)": 99999.99,
    "Proton flux (>60 Mev)": 99999.99,
    "Flux FLAG": 0,
    "ap-index": 999,
    "f10.7_index": 999.9,
    "pc-index": 999.9,
    "AL-index, nT": 99999,
    "AU-index, nT": 99999,
    "MAC": 99.9,
    "Lyman_alpha": 0.999999,
    "Proton Quasy-Invariant (QI)": 9.9999,
}


def set_fillers_nan(df):
    for column, fill_value in fill_values.items():
        if column in df.columns:  # Check if the column exists in your DataFrame
            df[column] = df[column].replace(fill_value, np.nan)

    return df


def drop_na(df):
    df.dropna(axis=0, how="any", inplace=True)


def read_data(
    year, replace_fillers=False, drop_na_values=False, keep_columns=[], drop_columns=[]
):
    with open(f"data/{year}.json", "r") as f:
        df = pd.read_json(f)

    if keep_columns:
        df = df[keep_columns]
    elif drop_columns:
        df.drop(drop_columns, axis="columns", inplace=True)

    if replace_fillers:
        df = set_fillers_nan(df)

    if drop_na_values:
        drop_na(df)

    return df


def read_multi_data(
    years, replace_fillers=False, drop_na_values=False, keep_columns=[], drop_columns=[]
):
    df = None

    for year in years:
        sub_df = read_data(
            year,
            replace_fillers=replace_fillers,
            drop_na_values=drop_na_values,
            keep_columns=keep_columns,
            drop_columns=drop_columns,
        )

        if df is None:
            df = sub_df
        else:
            df = pd.concat([df, sub_df], ignore_index=True)

    return df
