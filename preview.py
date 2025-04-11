import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np

# relocated the dataset to the same directory as the script
os.chdir(os.path.dirname(__file__))

df = pd.read_csv('../cleaned_dataset.csv')

plt.style.use('ggplot')

# creating subplots for the original and log-transformed distributions
fig, axes = plt.subplots(2, 2, figsize=(14, 6))

# original distribution of mean temperature
axes[0, 0].hist(df['Mean Temperature'], bins=30, color='blue', alpha=0.7)
axes[0, 0].set_title('Distribution of Mean Temperature')
axes[0, 0].set_xlabel('Mean Temperature')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].grid(axis='y', alpha=0.75)

# time series plot of mean temperature
axes[0, 1].plot(df['Mean Temperature'], color='blue', alpha=0.7)
axes[0, 1].set_title('Time Series of Mean Temperature')
axes[0, 1].set_xlabel('Time')
axes[0, 1].set_ylabel('Mean Temperature')
axes[0, 1].grid(axis='y', alpha=0.75)

# original distribution of rainfall
axes[1, 0].hist(df['Rainfall'], bins=30, color='green', alpha=0.7)
axes[1, 0].set_title('Distribution of Rainfall')
axes[1, 0].set_xlabel('Rainfall')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].grid(axis='y', alpha=0.75)

# time series plot of rainfall
axes[1, 1].plot(df['Rainfall'], color='green', alpha=0.7)
axes[1, 1].set_title('Time Series of Rainfall')
axes[1, 1].set_xlabel('Time')
axes[1, 1].set_ylabel('Rainfall')
axes[1, 1].grid(axis='y', alpha=0.75)

plt.tight_layout()
plt.show()

