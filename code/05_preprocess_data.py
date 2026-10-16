import pandas as pd

# Load final station lists from Step 04
final_fast = pd.read_csv("outputs/results/final_fast_stations.csv")
final_slow = pd.read_csv("outputs/results/final_slow_stations.csv")

# Create one list of unique final station IDs
final_station_ids = final_slow["station_id"].tolist()

for station_id in final_fast["station_id"].tolist():
    if station_id not in final_station_ids:
        final_station_ids.append(station_id)

print("Total unique final stations:")
print(len(final_station_ids))


# Store comparisons and changes for every hour
hour_comparisons = {}
hour_changes = {}

# Also store daily results so Nov 3 and Nov 4 can be checked
day_comparisons = {}
day_changes = {}


# Check every final station
for station_id in final_station_ids:

    file_path = "data/raw/20220901-20230228_station-raw/charge_5min/" + str(station_id) + ".csv"

    df = pd.read_csv(file_path)

    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time")
    df = df.reset_index(drop=True)

    time_values = df["time"].tolist()
    busy_values = df["busy"].tolist()


    # Compare every reading with the previous 5-minute reading
    for i in range(1,len(df)):

        previous_time = time_values[i - 1]
        current_time = time_values[i]

        previous_busy = busy_values[i - 1]
        current_busy = busy_values[i]


        # Only compare rows that are exactly 5 minutes apart
        if current_time - previous_time == pd.Timedelta(minutes=5):

            # Only compare when both busy values are available
            if pd.notna(previous_busy) and pd.notna(current_busy):

                hour = current_time.replace(minute=0,second=0)
                day = str(current_time.date())


                # Create dictionary entries when the hour is first seen
                if hour not in hour_comparisons:
                    hour_comparisons[hour] = 0
                    hour_changes[hour] = 0


                # Create dictionary entries when the day is first seen
                if day not in day_comparisons:
                    day_comparisons[day] = 0
                    day_changes[day] = 0


                hour_comparisons[hour] = hour_comparisons[hour] + 1
                day_comparisons[day] = day_comparisons[day] + 1


                # Count a change when busy is different from previous reading
                if current_busy != previous_busy:

                    hour_changes[hour] = hour_changes[hour] + 1
                    day_changes[day] = day_changes[day] + 1


# Calculate change percentage for every hour
hour_results = []

for hour in sorted(hour_comparisons.keys()):

    change_percent = (hour_changes[hour] / hour_comparisons[hour]) * 100

    if change_percent > 30:
        bad_hour = True
    else:
        bad_hour = False

    hour_results.append({
        "hour": hour,
        "comparisons": hour_comparisons[hour],
        "changes": hour_changes[hour],
        "change_percent": round(change_percent,2),
        "bad_hour": bad_hour
    })


hourly_summary = pd.DataFrame(hour_results)


# Calculate change percentage for every day
day_results = []

for day in sorted(day_comparisons.keys()):

    change_percent = (day_changes[day] / day_comparisons[day]) * 100

    day_results.append({
        "date": day,
        "change_percent": round(change_percent,2)
    })


daily_summary = pd.DataFrame(day_results)


# Count how many hours fall into different ranges
hours_under_10 = 0
hours_above_30 = 0
hours_above_40 = 0

for result in hour_results:

    if result["change_percent"] < 10:
        hours_under_10 = hours_under_10 + 1

    if result["change_percent"] > 30:
        hours_above_30 = hours_above_30 + 1

    if result["change_percent"] > 40:
        hours_above_40 = hours_above_40 + 1


print("\nTotal hours checked:")
print(len(hourly_summary))

print("\nHours below 10% change:")
print(hours_under_10)

print("\nHours above 30% change:")
print(hours_above_30)

print("\nHours above 40% change:")
print(hours_above_40)


# Check November 3 and November 4
print("\nNovember 3:")
print(daily_summary[daily_summary["date"] == "2022-11-03"])

print("\nNovember 4:")
print(daily_summary[daily_summary["date"] == "2022-11-04"])


# Calculate average change percentage for other November days
normal_november = []

for result in day_results:

    day = result["date"]

    if day.startswith("2022-11-") and day != "2022-11-03" and day != "2022-11-04":
        normal_november.append(result["change_percent"])


if len(normal_november) > 0:
    normal_november_average = sum(normal_november) / len(normal_november)
else:
    normal_november_average = 0


print("\nAverage change percentage for other November days:")
print(round(normal_november_average,2))


# Show the hours with the highest change percentages
highest_change_hours = hourly_summary.sort_values(by="change_percent",ascending=False)

print("\nHours with highest change percentages:")
print(highest_change_hours.head(20))


# Save the full hourly check
hourly_summary.to_csv("outputs/results/hourly_anomaly_check.csv",index=False)

# Save only hours detected as abnormal
bad_hours = hourly_summary[hourly_summary["bad_hour"] == True]
bad_hours.to_csv("outputs/results/bad_hours.csv",index=False)

# Get the 204 bad hours detected above
bad_hour_values = bad_hours["hour"].tolist()

# Columns affected by the charger-state anomaly
columns_to_blank = [
    "busy",
    "idle",
    "fast_busy",
    "fast_idle",
    "slow_busy",
    "slow_idle"
]

cleaning_results = []

print("\nCleaning final station files...")


for station_id in final_station_ids:

    file_path = "data/raw/20220901-20230228_station-raw/charge_5min/" + str(station_id) + ".csv"
    df = pd.read_csv(file_path)

    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time")
    df = df.reset_index(drop=True)


    # Check whether the station has a complete 5-minute timeline
    complete_timeline = True

    time_values = df["time"].tolist()

    for i in range(1,len(time_values)):

        if time_values[i] - time_values[i - 1] != pd.Timedelta(minutes=5):
            complete_timeline = False
            break


    # Create the hour belonging to every timestamp
    df["hour"] = df["time"].dt.floor("h")


    # Find rows that belong to one of the detected bad hours
    bad_rows = df["hour"].isin(bad_hour_values)

    number_of_bad_rows = bad_rows.sum()


    # Blank charger-state values during bad hours
    df.loc[bad_rows,columns_to_blank] = pd.NA


    # Remove temporary hour column
    df = df.drop(columns=["hour"])


    # Save cleaned station file
    output_path = "data/processed/cleaned_stations/" + str(station_id) + ".csv"
    df.to_csv(output_path,index=False)


    cleaning_results.append({
        "station_id": station_id,
        "rows": len(df),
        "complete_timeline": complete_timeline,
        "blanked_rows": number_of_bad_rows
    })


# Save cleaning information for every station
cleaning_summary = pd.DataFrame(cleaning_results)

cleaning_summary.to_csv(
    "outputs/results/cleaning_summary.csv",
    index=False
)


print("\nStations processed:")
print(len(cleaning_summary))

print("\nStations with complete 5-minute timeline:")
print(cleaning_summary["complete_timeline"].sum())

print("\nStations with incomplete timeline:")
print((cleaning_summary["complete_timeline"] == False).sum())

print("\nBlanked rows per station:")
print(cleaning_summary["blanked_rows"].describe())