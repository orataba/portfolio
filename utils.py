import pandas as pd
import joblib
import os
import logging

def load_csv(path: str, index_col=0, parse_dates=True) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=index_col, parse_dates=parse_dates)
    df.index = pd.to_datetime(df.index, format="%Y%m").to_period("M")
    return df

def ensure_numeric(df: pd.DataFrame) -> pd.DataFrame:
    return df.apply(pd.to_numeric, errors='coerce')

def save_pickle(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(obj, path)

def load_pickle(path: str):
    return joblib.load(path)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def init_logger(logfile_path: str = "output/pipeline.log"):
    ensure_dir(os.path.dirname(logfile_path))
    logger = logging.getLogger("PMQuant")
    logger.setLevel(logging.DEBUG)
    # Avoid duplicate handlers
    if not logger.handlers:
        fh = logging.FileHandler(logfile_path, mode='w')
        fh.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        fmt = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
        fh.setFormatter(fmt)
        ch.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger