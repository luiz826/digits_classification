import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .constants import ORDER

def filter_all(df: pd.DataFrame, class_: int, or_ = True) -> pd.DataFrame:
    filter_c_1 = df.loc[:, "label"] == class_
    filter_c_rest = df.loc[:, "label"] != class_
 
    if or_:
        return df[(filter_c_1) | (filter_c_rest)]
    
    return df[filter_c_1], df[filter_c_rest]

def remove_class(df: pd.DataFrame, class_: int) -> pd.DataFrame:
    return df[df.loc[:, "label"] != class_]
    
    
def plot_all(df: pd.DataFrame, wei = []) -> None:
    for k,w in enumerate(wei): 
        x1 = np.array([20, 150]) 
        x2 = -(w[0] + w[1]*x1) / w[2]
        plt.plot(x1, x2, label=f"{ORDER[k]} x {ORDER[k+1]}")
        

    for i in ORDER:
        plt.scatter(test.loc[test["label"] == i]['intensidade'], test.loc[test["label"] == i]['simetria'], label=f"{i}")

    plt.ylabel("Simetria")
    plt.xlabel("Intensidade")
    plt.title("Intensidade x Simetria")
    plt.ylim(50, 200)
    plt.legend()
    plt.show()
