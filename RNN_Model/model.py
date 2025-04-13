import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
import random
import os
import warnings
from torch.utils.data import DataLoader, TensorDataset

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

df = pd.read_csv("../Data/cleaned_dataset.csv")
print(df)


train_df, test_df = train_test_split(df, test_size=0.2, shuffle=False)
train_df, val_df = train_test_split(train_df, test_size=0.25, shuffle=False)

train_df_x = train_df.drop(columns=["Mean Temperature"])
train_df_y = train_df["Mean Temperature"]
val_df_x = val_df.drop(columns=["Mean Temperature"])
val_df_y = val_df["Mean Temperature"]
test_df_x = test_df.drop(columns=["Mean Temperature"])
test_df_y = test_df["Mean Temperature"]