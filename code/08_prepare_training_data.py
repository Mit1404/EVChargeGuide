import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Load final station lists from File 04
final_fast = pd.read_csv("outputs/results/final_fast_stations.csv")
final_slow = pd.read_csv("outputs/results/final_slow_stations.csv")

# Input features: only information known at the current time
# station_id is kept in the files only to know which station a row belongs to
# month is NOT used: training has Sep-Dec only, so the model never sees month 1 or 2
feature_columns = [
    "current_idle",
    "current_busy",
    "capacity",
    "hour",
    "minute",
    "day_of_week",
    "longitude",
    "latitude"
]

# Chronological split: [start, end)
splits = {
    "train": ["2022-09-01","2023-01-01"],
    "validation": ["2023-01-01","2023-02-01"],
    "test": ["2023-02-01","2023-03-01"]
}

horizons = [30,60]


# SECONDARY EXPERIMENT: hold out 20% of complete station IDs
# Random choice is used ONLY for station IDs, never for time rows
holdout_rows = []
holdout_ids = {}

for charger_type in ["fast","slow"]:

    if charger_type == "fast":
        station_ids = final_fast["station_id"].tolist()
    else:
        station_ids = final_slow["station_id"].tolist()

    known_ids,held_out_ids = train_test_split(station_ids,test_size=0.2,random_state=42)

    holdout_ids[charger_type] = held_out_ids

    for station_id in station_ids:

        if station_id in held_out_ids:
            group = "holdout"
        else:
            group = "known"

        holdout_rows.append({
            "station_id": station_id,
            "charger_type": charger_type,
            "group": group
        })

    print(charger_type, "stations:", len(station_ids), "known:", len(known_ids), "holdout:", len(held_out_ids))

holdout_split = pd.DataFrame(holdout_rows)
holdout_split.to_csv("outputs/results/station_holdout_split.csv",index=False)

# Save the feature list so the next teammate uses exactly the same inputs
pd.DataFrame({"feature": feature_columns}).to_csv("outputs/results/model_features.csv",index=False)


# MAIN EXPERIMENT: build train / validation / test files station by station
summary = {}

for charger_type in ["fast","slow"]:

    output_folder = "data/processed/model_data/" + charger_type + "/"
    os.makedirs(output_folder,exist_ok=True)

    # Remove old files first, because rows are appended below
    for horizon in horizons:
        for split_name in splits:
            path = output_folder + split_name + "_" + str(horizon) + ".csv"
            if os.path.exists(path):
                os.remove(path)

    if charger_type == "fast":
        station_ids = final_fast["station_id"].tolist()
    else:
        station_ids = final_slow["station_id"].tolist()

    print("\nPreparing", charger_type, "data...")

    for station_id in station_ids:

        file_path = "data/processed/target_data/" + charger_type + "/" + str(station_id) + ".csv"
        df = pd.read_csv(file_path)

        df["time"] = pd.to_datetime(df["time"])
        df["target_time_30"] = pd.to_datetime(df["target_time_30"])
        df["target_time_60"] = pd.to_datetime(df["target_time_60"])

        if station_id in holdout_ids[charger_type]:
            df["holdout_station"] = True
        else:
            df["holdout_station"] = False

        for horizon in horizons:

            target_column = "target_" + str(horizon)
            target_time_column = "target_time_" + str(horizon)

            for split_name in splits:

                start = splits[split_name][0]
                end = splits[split_name][1]

                # BOTH the current time and the target time must be inside the split
                # Example: Dec 31 23:50 + 30 min is in January, so it is not a training row
                in_split = (df["time"] >= start) & (df["time"] < end) & (df[target_time_column] >= start) & (df[target_time_column] < end)

                rows = df[in_split]

                # Keep only rows where all features and the target exist (nothing is filled)
                rows = rows.dropna(subset=feature_columns + [target_column])

                # Keep identifier columns, features and the target with simple names
                rows = rows[["station_id","holdout_station","time",target_time_column] + feature_columns + [target_column]]
                rows = rows.rename(columns={target_time_column: "target_time",target_column: "target"})

                # Add this station's rows to the end of the file (header only once)
                path = output_folder + split_name + "_" + str(horizon) + ".csv"
                rows.to_csv(path,mode="a",header=not os.path.exists(path),index=False)

                # Keep running totals for the summary and the leakage check
                key = charger_type + "_" + str(horizon) + "_" + split_name

                if key not in summary:
                    summary[key] = {
                        "charger_type": charger_type,
                        "horizon_minutes": horizon,
                        "split": split_name,
                        "rows": 0,
                        "holdout_station_rows": 0,
                        "stations": 0,
                        "first_time": None,
                        "last_target_time": None
                    }

                if len(rows) > 0:

                    summary[key]["rows"] = summary[key]["rows"] + len(rows)
                    summary[key]["holdout_station_rows"] = summary[key]["holdout_station_rows"] + rows["holdout_station"].sum()
                    summary[key]["stations"] = summary[key]["stations"] + 1

                    first_time = rows["time"].min()
                    last_target_time = rows["target_time"].max()

                    if summary[key]["first_time"] is None or first_time < summary[key]["first_time"]:
                        summary[key]["first_time"] = first_time

                    if summary[key]["last_target_time"] is None or last_target_time > summary[key]["last_target_time"]:
                        summary[key]["last_target_time"] = last_target_time


data_summary = pd.DataFrame(list(summary.values()))

# Leakage check: every target time must be before the end of its split
leak_free = []
for i in range(len(data_summary)):

    split_name = data_summary["split"][i]
    end = pd.Timestamp(splits[split_name][1])

    last_target_time = data_summary["last_target_time"][i]
    first_time = data_summary["first_time"][i]
    start = pd.Timestamp(splits[split_name][0])

    if last_target_time is None:
        leak_free.append(True)
    elif first_time >= start and last_target_time < end:
        leak_free.append(True)
    else:
        leak_free.append(False)

data_summary["leak_free"] = leak_free

data_summary.to_csv("outputs/results/training_data_summary.csv",index=False)

print("\nTraining data summary:")
print(data_summary.to_string(index=False))

print("\nAll splits leak free:")
print(data_summary["leak_free"].all())
