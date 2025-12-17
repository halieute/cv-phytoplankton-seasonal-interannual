# -*- coding: utf-8 -*-
"""
PCA Analysis with Seasonality Removal, Low-Pass Filter, 
and Statistical Independence Testing (Distance Correlation + P-Value)
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import dcor 

# === USER SETTINGS ===
file_path = "cv_box_selected_variables.csv"   
time_col = "time"                
sep = ","                        
n_components = 3                 
window_months = 12               

# === 1. Load and Pre-process ===
df = pd.read_csv(file_path, sep=sep)
df[time_col] = pd.to_datetime(df[time_col])
df = df.set_index(time_col)
variables = [c for c in df.columns if c != time_col]

# === 2. Remove Seasonality & Apply Filter ===
monthly_clim = df.groupby(df.index.month).mean()
anomalies = df.copy()
for t in df.index:
    anomalies.loc[t] = df.loc[t] - monthly_clim.loc[t.month]

lowpass = anomalies.rolling(window=window_months, center=True, min_periods=1).mean()
valid_data = lowpass.dropna()
valid_index = valid_data.index

# === 3. Standardize and Run PCA ===
scaler = StandardScaler()
scaled_data = scaler.fit_transform(valid_data)

pca = PCA(n_components=n_components)
pcs = pca.fit_transform(scaled_data)

# === 4. STATISTICAL INDEPENDENCE TEST (dCor + P-VALUE) ===
print("\n--- Statistical Independence Test (Distance Correlation) ---")

for i in range(n_components):
    for j in range(i + 1, n_components):
        pc_i = pcs[:, i]
        pc_j = pcs[:, j]

        # 1. Calculate the dCor value
        dcor_val = dcor.distance_correlation(pc_i, pc_j)

        # 2. Calculate the p-value (Statistical Significance)
        # num_resamples=500 is a good balance between accuracy and speed
        test_result = dcor.independence.distance_covariance_test(pc_i, pc_j, num_resamples=500)
        p_value = test_result.p_value

        print(f"Testing PC{i+1} vs PC{j+1}:")
        print(f"  - Distance Correlation: {dcor_val:.4f}")
        print(f"  - P-Value: {p_value:.4f}")

        if p_value < 0.05:
            print("  - RESULT: Significant dependence (Reject Independence)")
        else:
            print("  - RESULT: No significant dependence (Accept Independence)")
        print("-" * 40)

# === 5. Results and Plotting ===
explained_var = pca.explained_variance_ratio_
print("\nExplained variance ratio:")
for i, var in enumerate(explained_var):
    print(f"PC{i+1}: {var*100:.2f}%")

plt.figure(figsize=(10, 6))
for i in range(n_components):
    plt.plot(valid_index, pcs[:, i], label=f"PC{i+1}")
plt.title(f"PCA Time Series (Window: {window_months} months)")
plt.legend()
plt.savefig("pcs_timeseries_with_test.png")
plt.show()