import numpy as np
import pandas as pd
from .constants import *

# Intensidade

def intensity(pixels: np.array) -> np.array:
    sum_I = np.sum(pixels, axis=1)
    
    return sum_I / MAX_TON    

# Simetria

def vertical_simetry(img):
    p = img.reshape(
        N_PIXELS, 
        N_PIXELS
    )

    sum_diffs = 0
    for k in range(N_PIXELS):
        sum_diffs_row = 0 
        for i in range((N_PIXELS-2)//2): # 13
            j = (N_PIXELS-1)-i
            sum_diffs_row += abs(p[k][i] - p[k][j])

        sum_diffs += sum_diffs_row

    return sum_diffs / MAX_TON

def horizontal_simetry(img):
    p = img.reshape(
        N_PIXELS, 
        N_PIXELS
    )
    
    sum_diffs = 0
    for k in range((N_PIXELS-2)//2):
        sum_diffs_col = 0 
        for i in range(N_PIXELS):
            j = 27-k
            sum_diffs_col += abs(p[k][i] - p[j][i])

        sum_diffs += sum_diffs_col
    
    return sum_diffs / MAX_TON

def simetry(pixels):
    ver = np.array(list(map(vertical_simetry, pixels))) 
    hor = np.array(list(map(horizontal_simetry, pixels)))
    
    return ver + hor