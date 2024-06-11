import pandas as pd
import numpy as np

np_data = np.load("./data_total/original_full_1999_2021.npy")

print(np_data.dtype)
print(np_data.shape)