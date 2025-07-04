### This script generates a summary statistics report for the training and prediction times of different models.
# It groups the data by model name, number of cores, and platform, and calculates various statistics such as count, min, max, mean, median, std, and percentiles.

import pandas as pd
import numpy as np

maindf = pd.read_csv("master_csv.csv")

stats = maindf.groupby(["Model Name",  "Cores", "Platform"])[["Train Time", "Prediction Time"]].agg([
    "count", 
    "min", 
    "max",
    "mean",
    "median",
    "std", 
    ("1st", lambda x: np.percentile(x, 1)),
    ("5th", lambda x: np.percentile(x, 5)),
    ("95th", lambda x: np.percentile(x, 95)),
    ("99th", lambda x: np.percentile(x, 99))
    ])


stats.to_excel("output.xlsx", engine="xlsxwriter") #Save the file to excel spreadsheet. 