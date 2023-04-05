import numpy as np
import pandas as pd
from .constants import *

# Intensidade

def intensity(pixels: np.array) -> np.array:
    sum_I = np.sum(pixels, axis=1)
    
    return sum_I / MAX_TON    

# Simetria

def vertical_simetry(pixels):
    '''
      pixels -> Is all the dataset
    '''
    p = pixels.reshape(-1, 28, 28)

    A = p[:, :, :14]
    B = p[:, :, 14:]

    B_ = np.flip(B, 2)

    return np.sum(np.abs(A - B_), axis=1) / 255

def horizontal_simetry(pixels):
    '''
    pixels -> Is all the dataset
    '''
    p = pixels.reshape(-1, 28, 28)

    A = p[:, :14, :]
    B = p[:, 14:, :]

    B_ = np.flip(B, 1)

    return np.sum(np.abs(A - B_), axis=1) / 255

def simetry(pixels):
    ver = vertical_simetry, pixels)
    hor = horizontal_simetry, pixels)
    
    return ver + hor
