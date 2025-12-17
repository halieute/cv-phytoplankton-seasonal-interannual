# -*- coding: utf-8 -*-
"""
‘PCA for a selected month with removal of monthly climatology (anomalies).
Robust version: does not use asfreq, coercion to numeric, and NaN checks.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import dcor

# === USER SETTINGS ===
csv_file = "souley_thesis_data_aod_chl_sst_ws_dd_wd_cvbox.csv"
sep = ";"                 # adjust if your CSV uses a different separator
time_col = "time"
n_components = 3
output_prefix = "climate_pca_anoms"
selected_month = 12        # 1=jan ... 12=dec

# === 1. LOAD DATA ===
df = pd.read_csv(csv_file, sep=sep)
df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
df = df.set_index(time_col)

# Remove rows where date was not parsed
df = df[~df.index.isna()]
if df.empty:
    raise ValueError("Time index empty after parsing. Check the 'time' column in the CSV.")

# === 2. ENSURE COLUMNS ARE NUMERIC ===
# Identify data columns (all except 'time' which is now the index)
# try to convert all columns to numeric (coerce -> turns invalid strings into NaN)
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Now keep only truly numeric columns (those that have some number)
num_cols = df.columns[df.dtypes.apply(lambda dt: np.issubdtype(dt, np.number))]
if len(num_cols) == 0:
    raise ValueError("No numeric columns found after coercion. Check the CSV and separator/formats.")

df = df[num_cols]  # reduce to numeric columns

# === 3. CALCULATE MONTHLY CLIMATOLOGY (ignoring NaNs) ===
months = df.index.month
clim = df.groupby(months).mean()  # index 1..12 -> mean per month
# Diagnostic: how many entries per month and if climatology exists
count_by_month = df.groupby(months).count()
print("\nSample count per month (per variable):")
print(count_by_month.head(12))

print("\nMonthly climatology (shows NaNs if any month lacks data):")
print(clim.round(4))

# === 4. RECONSTRUCT CLIMATOLOGY DATAFRAME ALIGNED WITH ORIGINAL INDEX ===
# For each timestamp in df, get the climatology row for the corresponding month
clim_rows = []
for m in df.index.month:
    if m in clim.index:
        clim_rows.append(clim.loc[m].values)  # vector of variable means
    else:
        # if any month does not exist in clim (no data), put NaNs
        clim_rows.append(np.full(len(num_cols), np.nan))

clim_reindexed = pd.DataFrame(clim_rows, index=df.index, columns=num_cols)

# === 5. CALCULATE ANOMALIES (value - monthly climatology) ===
anoms = df - clim_reindexed

# Show mean of anomalies per month — if removal is OK, should be close to 0 (where data exists)
print("\nMean of anomalies per month (where data exists):")
print(anoms.groupby(anoms.index.month).mean().round(4))

# === 6. FILTER SELECTED MONTH ===
mask_month = anoms.index.month == selected_month
anoms_month = anoms.loc[mask_month].copy()
df_month = df.loc[mask_month].copy()

if anoms_month.empty:
    raise ValueError(f"No data available for month {selected_month} (filter resulted in empty).")

# Remove rows with NaNs (StandardScaler and PCA do not accept NaNs)
before = len(anoms_month)
anoms_month = anoms_month.dropna(axis=0, how='any')
df_month = df_month.loc[anoms_month.index]
after = len(anoms_month)
print(f"\nSelected month: {selected_month} -> {before} records before, {after} after dropna().")

if anoms_month.empty:
    raise ValueError(f"After removing rows with NaNs, no data remains for month {selected_month}.")

# === 7. STANDARDIZE AND RUN PCA ===
scaler = StandardScaler()
X = scaler.fit_transform(anoms_month.values)

pca = PCA(n_components=n_components)
PCs = pca.fit_transform(X)
EOFs = pca.components_
explained = pca.explained_variance_ratio_

# === 8. SAVE AND PRINT RESULTS ===
np.save(f"{output_prefix}_month{selected_month}_PCs.npy", PCs)
np.save(f"{output_prefix}_month{selected_month}_EOFs.npy", EOFs)
np.save(f"{output_prefix}_month{selected_month}_explained.npy", explained)

print("\nExplained variance ratio (%):")
for i, var in enumerate(explained):
    print(f"  PC{i+1}: {var*100:.2f}%")

loadings = pd.DataFrame(EOFs.T, columns=[f"PC{i+1}" for i in range(n_components)], index=num_cols)
print("\nLoadings (EOFs):")
print(loadings.round(4))


# === START OF NEW SECTION: TEST FOR INDEPENDENCE ===
# The 'PCs' array has shape (n_samples, n_components)
# We test the columns (the scores for each PC) against each other.
# PCs[:, 0] is PC1, PCs[:, 1] is PC2, PCs[:, 2] is PC3

print(f"\n--- Independence Tests for Month {selected_month} ---")

if n_components >= 2:
    # Test PC1 vs PC2
    dcor_1_2 = dcor.distance_correlation(PCs[:, 0], PCs[:, 1])
    print(f"  Distance Correlation (PC1 vs PC2): {dcor_1_2:.4f}")

if n_components >= 3:
    # Test PC1 vs PC3
    dcor_1_3 = dcor.distance_correlation(PCs[:, 0], PCs[:, 2])
    print(f"  Distance Correlation (PC1 vs PC3): {dcor_1_3:.4f}")

    # Test PC2 vs PC3
    dcor_2_3 = dcor.distance_correlation(PCs[:, 1], PCs[:, 2])
    print(f"  Distance Correlation (PC2 vs PC3): {dcor_2_3:.4f}")

print("--- End of Tests ---")
# === END OF NEW SECTION ===


# === 9. SIMPLE PLOTS ===
plt.figure(figsize=(6,4))
plt.bar(np.arange(1, n_components+1), explained*100)
plt.xlabel("PC")
plt.ylabel("Explained variance (%)")
plt.title(f"Explained variance - month {selected_month}")
plt.tight_layout()
plt.savefig(f"{output_prefix}_month{selected_month}_explained_variance.png", dpi=150)

# Time series of PCs (one point per year for the selected month)
fig, axs = plt.subplots(n_components, 1, figsize=(10, 2.5*n_components), sharex=True)
for i in range(n_components):
    axs[i].plot(anoms_month.index, PCs[:, i], marker='o')
    axs[i].set_ylabel(f"PC{i+1}")
axs[-1].set_xlabel("Time")
plt.tight_layout()
plt.savefig(f"{output_prefix}_month{selected_month}_PC_timeseries.png", dpi=150)

print(f"\nFiles saved with prefix: {output_prefix}_month{selected_month}_*")