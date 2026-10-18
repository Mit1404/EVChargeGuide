import pandas as pd

# Load missing-data summary and station metadata
station_summary = pd.read_csv("outputs/results/station_missing_summary.csv")
station_info = pd.read_csv("data/raw/20220901-20230228_station-raw/station_information.csv")

# Combine missing-data information with station metadata
all_stations = pd.merge(station_summary,station_info,on="station_id")

# Select slow station candidates with 10% or less missing data in every month
slow_candidates = all_stations[(all_stations["slow_count"] > 0) & (all_stations["sep_missing_percent"] <= 10) & (all_stations["oct_missing_percent"] <= 10) & (all_stations["nov_missing_percent"] <= 10) & (all_stations["dec_missing_percent"] <= 10) & (all_stations["jan_missing_percent"] <= 10) & (all_stations["feb_missing_percent"] <= 10)]

# Select fast station candidates with 20% or less missing data in every month
fast_candidates = all_stations[(all_stations["fast_count"] > 0) & (all_stations["sep_missing_percent"] <= 20) & (all_stations["oct_missing_percent"] <= 20) & (all_stations["nov_missing_percent"] <= 20) & (all_stations["dec_missing_percent"] <= 20) & (all_stations["jan_missing_percent"] <= 20) & (all_stations["feb_missing_percent"] <= 20)]

print("Slow station candidates:")
print(len(slow_candidates))

print("\nFast station candidates:")
print(len(fast_candidates))


# Apply the same checks to fast and slow stations
def apply_rules(candidate_stations,charger_type):

    final_station_ids = []
    check_results = []

    if charger_type == "fast":
        target_column = "fast_idle"
    else:
        target_column = "slow_idle"

    # Go through every candidate station
    for station_id in candidate_stations["station_id"]:

        station_id = int(station_id)

        # Get metadata for this station
        station = candidate_stations[candidate_stations["station_id"] == station_id]

        expected_fast = int(station["fast_count"].values[0])
        expected_slow = int(station["slow_count"].values[0])

        # Load station time-series data
        file_path = "data/raw/20220901-20230228_station-raw/charge_5min/" + str(station_id) + ".csv"
        df = pd.read_csv(file_path)

        df["time"] = pd.to_datetime(df["time"])
        df = df.sort_values("time")
        df = df.reset_index(drop=True)


        # Check if charger counts match the station metadata
        type_match = True

        fast_busy_values = df["fast_busy"].tolist()
        fast_idle_values = df["fast_idle"].tolist()
        slow_busy_values = df["slow_busy"].tolist()
        slow_idle_values = df["slow_idle"].tolist()

        for i in range(len(df)):

            fast_busy = fast_busy_values[i]
            fast_idle = fast_idle_values[i]
            slow_busy = slow_busy_values[i]
            slow_idle = slow_idle_values[i]

            if pd.notna(fast_busy) and pd.notna(fast_idle):

                if fast_busy + fast_idle != expected_fast:
                    type_match = False
                    break

            if pd.notna(slow_busy) and pd.notna(slow_idle):

                if slow_busy + slow_idle != expected_slow:
                    type_match = False
                    break


        # Check for long periods where availability never changes
        availability_values = df[target_column].tolist()

        previous_value = None
        current_run = 0
        stuck_rows = 0

        for value in availability_values:

            # Missing value ends the current continuous run
            if pd.isna(value):

                if current_run >= 288:
                    stuck_rows = stuck_rows + current_run

                current_run = 0
                previous_value = None

            else:

                # Same value as previous 5-minute reading
                if previous_value is not None and value == previous_value:
                    current_run = current_run + 1

                else:

                    # Save the previous run if it lasted at least 24 hours
                    if current_run >= 288:
                        stuck_rows = stuck_rows + current_run

                    current_run = 1

                previous_value = value


        # Check the final run
        if current_run >= 288:
            stuck_rows = stuck_rows + current_run

        # Count availability readings that are actually present
        valid_rows = df[target_column].notnull().sum()

        # Percentage of available readings that belong to long unchanged periods
        if valid_rows > 0:
            stuck_percent = (stuck_rows / valid_rows) * 100
        else:
            stuck_percent = 100

        if stuck_percent > 50:
            stuck_feed = True
        else:
            stuck_feed = False


        # Check whether the same hour is repeatedly missing in January or February
        worst_missing_hour_percent = 0
        worst_month = ""
        worst_hour = -1

        for month in [1,2]:

            month_data = df[(df["time"].dt.year == 2023) & (df["time"].dt.month == month)]

            total_days = len(month_data["time"].dt.date.unique())

            # Store which days are missing for each hour
            missing_days_by_hour = {}

            for hour in range(24):
                missing_days_by_hour[hour] = []


            # Look only at rows where availability is missing
            missing_rows = month_data[month_data[target_column].isnull()]


            # Save the date on which each hour has missing data
            for timestamp in missing_rows["time"]:

                hour = timestamp.hour
                day = timestamp.date()

                if day not in missing_days_by_hour[hour]:
                    missing_days_by_hour[hour].append(day)


            # Calculate missing percentage for every hour
            for hour in range(24):

                missing_days = len(missing_days_by_hour[hour])

                if total_days > 0:
                    missing_percent = (missing_days / total_days) * 100
                else:
                    missing_percent = 0


                if missing_percent > worst_missing_hour_percent:

                    worst_missing_hour_percent = missing_percent
                    worst_hour = hour

                    if month == 1:
                        worst_month = "January"
                    else:
                        worst_month = "February"


        if worst_missing_hour_percent > 50:
            repeated_missing_hour = True
        else:
            repeated_missing_hour = False


        # Store reasons if a station fails any check
        reasons = []

        if type_match == False:
            reasons.append("charger type/count mismatch")

        if stuck_feed == True:
            reasons.append("stuck feed")

        if repeated_missing_hour == True:
            reasons.append("same hour repeatedly missing in Jan/Feb")


        # Station passes only when there are no failure reasons
        if len(reasons) == 0:
            passed = True
            final_station_ids.append(station_id)

        else:
            passed = False
            print(station_id,"removed:",", ".join(reasons))


        # Save results of the checks
        check_results.append({
            "station_id": station_id,
            "charger_type": charger_type,
            "type_match": type_match,
            "stuck_percent": round(stuck_percent,2),
            "worst_missing_hour_percent": round(worst_missing_hour_percent,2),
            "worst_month": worst_month,
            "worst_hour": worst_hour,
            "passed": passed,
            "reason": ", ".join(reasons)
        })


    # Keep only stations that passed all checks
    final_stations = candidate_stations[candidate_stations["station_id"].isin(final_station_ids)].copy()

    return final_stations,check_results


print("\nChecking fast stations...")

final_fast_stations,fast_checks = apply_rules(fast_candidates,"fast")


print("\nChecking slow stations...")

final_slow_stations,slow_checks = apply_rules(slow_candidates,"slow")


# Combine results from fast and slow stations
all_checks = fast_checks + slow_checks
station_quality_checks = pd.DataFrame(all_checks)

# Sort final stations by station ID
final_fast_stations = final_fast_stations.sort_values(by="station_id")
final_slow_stations = final_slow_stations.sort_values(by="station_id")

print("\nFinal fast stations:")
print(len(final_fast_stations))

print("Total fast chargers:")
print(final_fast_stations["fast_count"].sum())

print("\nFinal slow stations:")
print(len(final_slow_stations))

print("Total slow chargers:")
print(final_slow_stations["slow_count"].sum())


# Save results
final_fast_stations.to_csv("outputs/results/final_fast_stations.csv",index=False)
final_slow_stations.to_csv("outputs/results/final_slow_stations.csv",index=False)
station_quality_checks.to_csv("outputs/results/station_quality_checks.csv",index=False)