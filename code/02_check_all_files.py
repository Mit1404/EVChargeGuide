import pandas as pd
import glob
import os

# Path containing all station CSV files
data_path = "data/raw/20220901-20230228_station-raw/charge_5min/*.csv"
# Get all station CSV files
station_files = glob.glob(data_path)
# Store information for every station
results = []
for file in station_files:
    # Load only the columns needed for checking missing availability data
    df = pd.read_csv(file,usecols=["time", "fast_idle", "slow_idle"])
    # Convert time column to datetime
    df["time"] = pd.to_datetime(df["time"])
    # Get station ID from file name
    station_id = os.path.splitext(os.path.basename(file))[0]
    # A row is considered missing if fast or slow availability is missing
    df["availability_missing"] = (df["fast_idle"].isnull() |df["slow_idle"].isnull())
    # September
    september = df[(df["time"] >= "2022-09-01") &(df["time"] < "2022-10-01")]
    # October
    october = df[(df["time"] >= "2022-10-01") &(df["time"] < "2022-11-01")]
    # November
    november = df[(df["time"] >= "2022-11-01") &(df["time"] < "2022-12-01")]
    # December
    december = df[(df["time"] >= "2022-12-01") &(df["time"] < "2023-01-01")]
    # January
    january = df[(df["time"] >= "2023-01-01") &(df["time"] < "2023-02-01")]
    # February
    february = df[(df["time"] >= "2023-02-01") &(df["time"] < "2023-03-01")]
    # Calculate percentage of missing rows for every month
    sep_missing = september["availability_missing"].mean() * 100
    oct_missing = october["availability_missing"].mean() * 100
    nov_missing = november["availability_missing"].mean() * 100
    dec_missing = december["availability_missing"].mean() * 100
    jan_missing = january["availability_missing"].mean() * 100
    feb_missing = february["availability_missing"].mean() * 100
    # Overall missing percentage
    overall_missing = df["availability_missing"].mean() * 100
    # Store results for this station
    results.append({
        "station_id": station_id,
        "sep_missing_percent": sep_missing,
        "oct_missing_percent": oct_missing,
        "nov_missing_percent": nov_missing,
        "dec_missing_percent": dec_missing,
        "jan_missing_percent": jan_missing,
        "feb_missing_percent": feb_missing,
        "overall_missing_percent": overall_missing
    })
# Convert results into a DataFrame
station_summary = pd.DataFrame(results)
# Round percentages to 2 decimal places
station_summary = station_summary.round(2)
# Sort from least missing data to most missing data
station_summary = station_summary.sort_values(by="overall_missing_percent")
print("Number of stations:")
print(len(station_summary))
print("\nBest 20 stations:")
print(station_summary.head(20))
print("\nWorst 20 stations:")
print(station_summary.tail(20))
print("\nSummary:")
print(station_summary.describe())

# Save results to a CSV file
station_summary.to_csv("outputs/results/station_missing_summary.csv",index=False)
# Check how many stations have low missing data in every month
month_columns = [
    "sep_missing_percent",
    "oct_missing_percent",
    "nov_missing_percent",
    "dec_missing_percent",
    "jan_missing_percent",
    "feb_missing_percent"
]

stations_under_5 = station_summary[(station_summary[month_columns] <= 5).all(axis=1)]
stations_under_10 = station_summary[(station_summary[month_columns] <= 10).all(axis=1)]
stations_under_20 = station_summary[(station_summary[month_columns] <= 20).all(axis=1)]

print("\nStations with 5% or less missing in every month:")
print(len(stations_under_5))

print("\nStations with 10% or less missing in every month:")
print(len(stations_under_10))

print("\nStations with 20% or less missing in every month:")
print(len(stations_under_20))