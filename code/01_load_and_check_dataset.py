import pandas as pd

# Load one charging station
file_path = "data/raw/20220901-20230228_station-raw/charge_5min/1001.csv"

df = pd.read_csv(file_path)

# Display first 5 rows
print("First 5 rows:")
print(df.head())

# Dataset size
print("\nShape:")
print(df.shape)

# Column names
print("\nColumns:")
print(df.columns)

# Data types
print("\nData types:")
print(df.dtypes)

# Missing values
print("\nMissing values:")
print(df.isnull().sum())

# Duplicate rows
print("\nDuplicate rows:")
print(df.duplicated().sum())

# Convert time column to datetime
df["time"] = pd.to_datetime(df["time"])

# Check time range
print("\nTime range:")
print(df["time"].min(), "to", df["time"].max())

# Basic statistics
print("\nBasic statistics:")
print(df.describe())

# Rows where station data is missing
missing_rows = df[df["fast_idle"].isnull()]

print("\nFirst missing rows:")
print(missing_rows.head(20))

print("\nLast missing rows:")
print(missing_rows.tail(20))

# Count missing rows by month
print("\nMissing values by month:")
print(
    missing_rows["time"]
    .dt.to_period("M")
    .value_counts()
    .sort_index()
)

# Find continuous blocks of missing data
df["is_missing"] = df["fast_idle"].isnull()

# Give each change from missing to non-missing a group number
df["missing_group"] = (df["is_missing"] != df["is_missing"].shift()).cumsum()

missing_blocks = (
    df[df["is_missing"]]
    .groupby("missing_group")
    .agg(
        start_time=("time", "min"),
        end_time=("time", "max"),
        number_of_rows=("time", "count")
    )
)

print("\nContinuous missing blocks:")
print(missing_blocks)