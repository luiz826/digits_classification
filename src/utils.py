import numpy as np
import pandas as pd
from . import constants

def intensity(pixels: np.array) -> np.array:
    sum_I = np.sum(pixels, axis=1)
    
    return sum_I / constants.MAX_TON    
