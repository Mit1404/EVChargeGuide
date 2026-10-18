import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load final station lists from File 04
final_fast = pd.read_csv("outputs/results/final_fast_stations.csv")
final_slow = pd.read_csv("outputs/results/final_slow_stations.csv")

results = []


# Calculate MAE and RMSE and save one result row
def add_result(charger_type,horizon,split_name,station_group,baseline,actual,predicted):

    mae = mean_absolute_error(actual,predicted)
    rmse = mean_squared_error(actual,predicted) ** 0.5

    results.append({
        "charger_type": charger_type,
        "horizon_minutes": horizon,
        "split": split_name,
        "stations": station_group,
        "baseline": baseline,
        "rows": len(actual),
        "MAE": round(mae,4),
        "RMSE": round(rmse,4)
    })


for charger_type in ["fast","slow"]:

    if charger_type == "fast":
        station_ids = final_fast["station_id"].tolist()
    else:
        station_ids = final_slow["station_id"].tolist()

    # Idle values from Dec 24 onward, used to look up "same time last week"
    # (validation starts Jan 1, and one week before that is Dec 25)
    lookup_parts = []

    for station_id in station_ids:

        file_path = "data/processed/target_data/" + charger_type + "/" + str(station_id) + ".csv"
        df = pd.read_csv(file_path,usecols=["station_id","time","current_idle"])

        df["time"] = pd.to_datetime(df["time"])
        df = df[df["time"] >= "2022-12-24"]

        lookup_parts.append(df)

    # pd.concat stacks the small per-station tables into one table
    lookup = pd.concat(lookup_parts)
    lookup = lookup.rename(columns={"time": "week_ago_time","current_idle": "same_week_prediction"})

    for horizon in [30,60]:

        for split_name in ["validation","test"]:

            file_path = "data/processed/model_data/" + charger_type + "/" + split_name + "_" + str(horizon) + ".csv"
            data = pd.read_csv(file_path)

            data["target_time"] = pd.to_datetime(data["target_time"])

            # Same clock time one week before the TARGET time
            data["week_ago_time"] = data["target_time"] - pd.Timedelta(days=7)

            # Bring in the idle value from one week earlier (missing if that reading was missing)
            data = pd.merge(data,lookup,on=["station_id","week_ago_time"],how="left")

            # Evaluate on all stations, and on test also split by known / holdout stations
            groups = {"all": data}

            if split_name == "test":
                groups["known"] = data[data["holdout_station"] == False]
                groups["holdout"] = data[data["holdout_station"] == True]

            for station_group in groups:

                group_data = groups[station_group]

                if len(group_data) == 0:
                    continue

                # Persistence: future idle = current idle
                add_result(charger_type,horizon,split_name,station_group,"persistence",group_data["target"],group_data["current_idle"])

                # Same time last week: only rows where that value exists (nothing invented)
                week_rows = group_data[group_data["same_week_prediction"].notna()]

                if len(week_rows) > 0:
                    add_result(charger_type,horizon,split_name,station_group,"same_time_last_week",week_rows["target"],week_rows["same_week_prediction"])

                    # Persistence on exactly the same rows, for a fair comparison
                    add_result(charger_type,horizon,split_name,station_group,"persistence_same_rows",week_rows["target"],week_rows["current_idle"])


baseline_metrics = pd.DataFrame(results)
baseline_metrics.to_csv("outputs/results/baseline_metrics.csv",index=False)

print("Baseline results:")
print(baseline_metrics.to_string(index=False))
