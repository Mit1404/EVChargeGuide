import pandas as pd

# Load missing-data summary
station_summary = pd.read_csv("outputs/results/station_missing_summary.csv")
# Load station metadata
station_info = pd.read_csv("data/raw/20220901-20230228_station-raw/station_information.csv")

# Monthly missing-data columns
month_columns = [
    "sep_missing_percent",
    "oct_missing_percent",
    "nov_missing_percent",
    "dec_missing_percent",
    "jan_missing_percent",
    "feb_missing_percent"
]

# Keep stations with 10% or less missing data in every month
usable_summary = station_summary[(station_summary[month_columns] <= 10).all(axis=1)]

print("Usable stations based on missing-data rule:")
print(len(usable_summary))
# Combine usable stations with their metadata
usable_stations = pd.merge(usable_summary,station_info,on="station_id")
# Combine missing-data information with station metadata
# Merge all stations with metadata so we can compare
# different missing-data limits specifically for fast stations
all_stations = pd.merge(
    station_summary,
    station_info,
    on="station_id"
)
# Total number of stations that actually have fast chargers
fast_stations = all_stations[
    all_stations["fast_count"] > 0
]

print("\nTotal stations with fast chargers:")
print(len(fast_stations))


# Fast stations with <= 10% missing in every month
fast_under_10 = fast_stations[
    (fast_stations[month_columns] <= 10).all(axis=1)
]

print("\nFast stations with 10% or less missing in every month:")
print(len(fast_under_10))


# Fast stations with <= 20% missing in every month
fast_under_20 = fast_stations[
    (fast_stations[month_columns] <= 20).all(axis=1)
]

print("\nFast stations with 20% or less missing in every month:")
print(len(fast_under_20))


# Fast stations with <= 30% missing in every month
fast_under_30 = fast_stations[
    (fast_stations[month_columns] <= 30).all(axis=1)
]

print("\nFast stations with 30% or less missing in every month:")
print(len(fast_under_30))

print("\nFirst 10 usable stations within 10% pool:")
print(usable_stations.head(10))

print("\nStations with slow chargers within 10% pool:")
print((usable_stations["slow_count"] > 0).sum())

print("\nStations with both fast and slow chargers within 10% pool:")
print( ((usable_stations["fast_count"] > 0) &(usable_stations["slow_count"] > 0)).sum())

print("\nBasic metadata statistics:")
print(usable_stations[["slow_count", "fast_count", "charge_count"]].describe())

# Save usable station information
usable_stations.to_csv(
    "outputs/results/stations_under_10_percent.csv",
    index=False
)