import kaggle
from pathlib import Path
import yaml
import gzip 
import pickle
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup as soup

import dask
dask.config.set({'dataframe.query-planning': True})
import dask.dataframe as dd

import dask_ml

config = yaml.safe_load(open("config.yaml"))
data_path_str = config["data_path"]
cleaned_data_path_str = config["cleaned_data_path"]
redownload = config["redownload"]
news_mhtml_path_str = config["news_mhtml_path"]
sampling_rate = config["sampling_rate"]
news_csv_path_str = config["news_csv_path"]
timedelta = config["timedelta"]

cleaned_data_path = Path(cleaned_data_path_str)
cleaned_data_path.mkdir(parents=True, exist_ok=True)