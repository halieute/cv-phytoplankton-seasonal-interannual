# -*- coding: utf-8 -*-
"""
Created on Fri Oct 17 12:00:13 2025
What this script does
PCA - Analysis
Removes seasonality by subtracting the monthly climatology.

Applies a Butterworth low-pass filter to isolate interannual/decadal signals.

Standardizes variables so they contribute equally.

Runs PCA and prints loadings + explained variance.
@author: nilto
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# === USER SETTINGS ===
file_path = "PCA_deseaonalized/cv1_selected_variables.csv"   # path to your CSV
time_col = "time"                # time column name
sep = ","                        # CSV separator
n_components = 3                 # number of principal components
window_months = 12               # low-pass filter window (12 = annual mean)

# === 1. Load the data ===
df = pd.read_csv(file_path, sep=sep)
df[time_col] = pd.to_datetime(df[time_col])
df = df.set_index(time_col)
variables = [c for c in df.columns if c != time_col]

# === 2. Remove seasonality ===
monthly_clim = df.groupby(df.index.month).mean()
anomalies = df.copy()
for t in df.index:
    anomalies.loc[t] = df.loc[t] - monthly_clim.loc[t.month]

# === 3. Apply low-pass filter (rolling mean) ===
# Rolling window centered on each month
lowpass = anomalies.rolling(window=window_months, center=True, min_periods=1).mean()

# === 4. Standardize anomalies ===
scaler = StandardScaler()
scaled_data = scaler.fit_transform(lowpass.dropna())

# Keep aligned time index (after removing NaNs)
valid_index = lowpass.dropna().index

# === 5. Run PCA ===
pca = PCA(n_components=n_components)
pcs = pca.fit_transform(scaled_data)
explained_var = pca.explained_variance_ratio_

# === 6. Results ===
print("\nExplained variance ratio:")
for i, var in enumerate(explained_var):
    print(f"PC{i+1}: {var*100:.2f}%")

loadings = pd.DataFrame(
    pca.components_.T,
    columns=[f"PC{i+1}" for i in range(n_components)],
    index=variables
)
print("\nPCA Loadings (EOFs):")
print(loadings)

# === 7. Plot principal components ===
plt.figure(figsize=(10, 6))
for i in range(n_components):
    plt.plot(valid_index, pcs[:, i], label=f"PC{i+1}")
plt.title(f"PCA after Removing Seasonality + {window_months}-month Low-pass Filter")
plt.xlabel("Time")
plt.ylabel("Principal Component Value")
plt.legend()
plt.tight_layout()
plt.show()

# === 8. Save results ===
np.save("pcs_lowpass.npy", pcs)
loadings.to_csv("pca_loadings_lowpass.csv")
pd.DataFrame(pcs, index=valid_index,
             columns=[f"PC{i+1}" for i in range(n_components)]).to_csv("pcs_timeseries_lowpass.csv")

print("\nResults saved: pcs_lowpass.npy, pca_loadings_lowpass.csv, pcs_timeseries_lowpass.csv")
