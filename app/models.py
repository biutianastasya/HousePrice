import joblib
import pandas as pd
import numpy as np
import glob
from pathlib import Path

def load_model(path):
    return joblib.load(path)

def load_features(path):
    return joblib.load(path)

def load_poi(pattern):
    files = glob.glob(pattern)
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        dtype = Path(f).stem.replace('poi_', '')
        df['facility_type'] = dtype
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame(columns=['lat','long','facility_type'])

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))

def compute_nearest_distance(lat, lon, poi_df):
    if poi_df.empty:
        return np.nan
    dists = haversine(lat, lon, poi_df['lat'].values, poi_df['long'].values)
    return np.min(dists)