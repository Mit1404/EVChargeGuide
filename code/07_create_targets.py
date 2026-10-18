import os
import pandas as pd

# Load final station lists from File 04
final_fast = pd.read_csv("outputs/results/final_fast_stations.csv")
final_slow = pd.read_csv("outputs/results/final_slow_stations.csv")

# Create output folders if they do not exist yet
os.makedirs("data/processed/target_data/fast",exist_ok=True)
os.makedirs("data/processed/target_data/slow",exist_ok=True)

summary_rows = []


def create_targets(station_list,charger_type):

    for station_id in station_list["station_id"].tolist():

        station = station_list[station_list["station_id"] == station_id]

        # Load cleaned station file from File 05
        file_path = "data/processed/cleaned_stations/" + str(station_id) + ".csv"
        df = pd.read_csv(file_path)

        df["time"] = pd.to_datetime(df["time"])
        df = df.sort_values("time")
        df = df.reset_index(drop=True)

        # Keep only what we need, with the same column names for fast and slow
        target_df = pd.DataFrame()
        target_df["station_id"] = [station_id] * len(df)
        target_df["time"] = df["time"]
        target_df["current_idle"] = df[charger_type + "_idle"]
        target_df["current_busy"] = df[charger_type + "_busy"]
        target_df["capacity"] = int(station[charger_type + "_count"].values[0])
        target_df["longitude"] = station["longitude"].values[0]
        target_df["latitude"] = station["latitude"].values[0]

        # Simple time features (all known at the current time)
        target_df["hour"] = target_df["time"].dt.hour
        target_df["minute"] = target_df["time"].dt.minute
        target_df["day_of_week"] = target_df["time"].dt.dayofweek
        target_df["month"] = target_df["time"].dt.month

        # Future values: shift(-6) takes the value from 6 rows below
        # 6 rows x 5 minutes = 30 minutes, 12 rows = 60 minutes
        # If the future value is missing (bad hour or gap), the target stays missing
        target_df["target_30"] = target_df["current_idle"].shift(-6)
        target_df["target_time_30"] = target_df["time"].shift(-6)

        target_df["target_60"] = target_df["current_idle"].shift(-12)
        target_df["target_time_60"] = target_df["time"].shift(-12)

        # Safety check: target times must be exactly 30 and 60 minutes later
        has_30 = target_df["target_time_30"].notna()
        has_60 = target_df["target_time_60"].notna()

        wrong_30 = ((target_df["target_time_30"] - target_df["time"])[has_30] != pd.Timedelta(minutes=30)).sum()
        wrong_60 = ((target_df["target_time_60"] - target_df["time"])[has_60] != pd.Timedelta(minutes=60)).sum()

        if wrong_30 > 0 or wrong_60 > 0:
            print("WARNING: station", station_id, "has target times that are not 30/60 minutes ahead")

        output_path = "data/processed/target_data/" + charger_type + "/" + str(station_id) + ".csv"
        target_df.to_csv(output_path,index=False)

        summary_rows.append({
            "station_id": station_id,
            "charger_type": charger_type,
            "rows": len(target_df),
            "current_available": target_df["current_idle"].notnull().sum(),
            "target_30_available": target_df["target_30"].notnull().sum(),
            "target_60_available": target_df["target_60"].notnull().sum(),
            "wrong_target_times": wrong_30 + wrong_60
        })


print("Creating fast targets...")
create_targets(final_fast,"fast")

print("Creating slow targets...")
create_targets(final_slow,"slow")


target_summary = pd.DataFrame(summary_rows)
target_summary.to_csv("outputs/results/target_summary.csv",index=False)

print("\nFiles created:")
print(target_summary["charger_type"].value_counts())

print("\nTotals by charger type:")
print(target_summary.groupby("charger_type")[["rows","current_available","target_30_available","target_60_available"]].sum())

print("\nStations with wrong target times (should be 0):")
print((target_summary["wrong_target_times"] > 0).sum())
