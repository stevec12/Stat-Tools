#!/usr/bin/env python
""" CSV Column Statistics

For each column of a CSV, prints:
- (min, lower quantile, median, upper quantile, max)
- (mean, sd, skewness, kurtosis)
- Likely distribution from (binomial, geometric, Poisson, exponential, Gaussian, multinomial)

Accepts .csv files.
"""
import sys
import pandas as pd

def summarize_csv(file_name : str) -> None:
    data = pd.read_csv(file_name)
    for i in range(0,2):#len(data.columns.values)):
        print(data.columns.values[i])
        col = data.iloc[:,i]
        if col.dtype == 'int64' or col.dtype == 'float64':
            print(data.iloc[:,i].describe())


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: Please specify an Excel File to summarize.")
        exit(1)
    summarize_csv(sys.argv[1])