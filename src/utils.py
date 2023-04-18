import numpy as np
import pandas as pd
from .constants import *

# Intensidade

def intensity(pixels: np.array) -> np.array:
    sum_I = np.sum(pixels, axis=1)
    
    return sum_I / MAX_TON    

# Simetria

def vertical_simetry(pixels: np.array)-> np.array:
    '''
      pixels -> Is all the dataset
    '''
    p = pixels.reshape(-1, N_PIXELS, N_PIXELS)

    A = p[:, :, :N_PIXELS//2]
    B = p[:, :, N_PIXELS//2:]

    B_ = np.flip(B, 2)

    return np.sum(np.sum(np.abs(A - B_), axis=1), axis=1) / MAX_TON 

def horizontal_simetry(pixels: np.array) -> np.array:
    '''
    pixels -> Is all the dataset
    '''
    p = pixels.reshape(-1, N_PIXELS, N_PIXELS)

    A = p[:, :N_PIXELS//2, :]
    B = p[:, N_PIXELS//2:, :]

    B_ = np.flip(B, 1)

    return np.sum(np.sum(np.abs(A - B_), axis=2), axis=1) / MAX_TON

def simetry(pixels: np.array) -> np.array:
    ver = vertical_simetry(pixels)
    hor = horizontal_simetry(pixels)
    
    return ver + hor


def filter_0_x_5(df: pd.DataFrame) -> pd.DataFrame:
    filter_0 = df.loc[:, "label"] == 0
    filter_5 = df.loc[:, "label"] == 5
    
    return df[(filter_0) | (filter_5)]