import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def filter_1x5(df: pd.DataFrame, or_ = True) -> pd.DataFrame:
    filter_1 = df.loc[:, "label"] == 1
    filter_5 = df.loc[:, "label"] == 5
    
    if or_:
        return df[(filter_1) | (filter_5)]
    
    return df[filter_1], df[filter_5]

def plot1x5(df: pd.DataFrame) -> None:
    filter_1, filter_5 = filter_1x5(df, False)

    plt.scatter(filter_1['intensidade'], filter_1['simetria'], color="blue", label="Um")
    plt.scatter(filter_5['intensidade'], filter_5['simetria'], color="red", label="Cinco")
    plt.ylabel("Simetria")
    plt.xlabel("Intensidade")
    plt.title("Intensidade x Simetria")
    plt.legend()
    plt.show()
    