import math
import pandas as pd

# Load results from earlier files
station_summary = pd.read_csv("outputs/results/station_missing_summary.csv")
station_info = pd.read_csv("data/raw/20220901-20230228_station-raw/station_information.csv")
final_fast = pd.read_csv("outputs/results/final_fast_stations.csv")
final_slow = pd.read_csv("outputs/results/final_slow_stations.csv")
bad_hours = pd.read_csv("outputs/results/bad_hours.csv")

# Combine missing-data information with station metadata for all original stations
all_stations = pd.merge(station_summary,station_info,on="station_id")

core_fast_ids = final_fast["station_id"].tolist()
core_slow_ids = final_slow["station_id"].tolist()

print("All original stations:")
print(len(all_stations))

print("\nCore fast stations:")
print(len(core_fast_ids))

print("\nCore slow stations:")
print(len(core_slow_ids))


# Straight-line distance in km between two points using latitude and longitude
# (haversine formula: distance on the surface of the Earth)
def distance_km(lat1,lon1,lat2,lon2):

    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)

    a = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2

    return 6371 * 2 * math.asin(math.sqrt(a))


# PART A: redirection coverage of the core stations

radius_list = [2,5,10]


def check_coverage(charger_type,core_ids):

    count_column = charger_type + "_count"

    # Every original station that has this charger type
    type_stations = all_stations[all_stations[count_column] > 0]

    station_ids = type_stations["station_id"].tolist()
    latitudes = type_stations["latitude"].tolist()
    longitudes = type_stations["longitude"].tolist()
    capacities = type_stations[count_column].tolist()

    # True if the station is in the core list for this charger type
    is_core = []
    for station_id in station_ids:
        is_core.append(station_id in core_ids)

    coverage_rows = []

    # Distance from every station of this type to its nearest OTHER core station
    nearest_core_km = {}

    for i in range(len(station_ids)):

        core_counts = {}
        all_counts = {}
        for radius in radius_list:
            core_counts[radius] = 0
            all_counts[radius] = 0

        nearest_core = None
        nearest_any = None

        for j in range(len(station_ids)):

            # Do not compare a station with itself
            if i == j:
                continue

            d = distance_km(latitudes[i],longitudes[i],latitudes[j],longitudes[j])

            if nearest_any is None or d < nearest_any:
                nearest_any = d

            if is_core[j] == True:
                if nearest_core is None or d < nearest_core:
                    nearest_core = d

            for radius in radius_list:
                if d <= radius:
                    all_counts[radius] = all_counts[radius] + 1
                    if is_core[j] == True:
                        core_counts[radius] = core_counts[radius] + 1

        nearest_core_km[station_ids[i]] = nearest_core

        # Save neighbour counts only for core stations
        if is_core[i] == True:

            row = {
                "station_id": station_ids[i],
                "charger_type": charger_type,
                "capacity": capacities[i]
            }

            for radius in radius_list:
                row["core_within_" + str(radius) + "km"] = core_counts[radius]
                row["all_within_" + str(radius) + "km"] = all_counts[radius]

            row["nearest_core_km"] = round(nearest_core,2)
            row["nearest_any_km"] = round(nearest_any,2)

            coverage_rows.append(row)

    return coverage_rows,nearest_core_km


print("\nCalculating fast redirection coverage...")
fast_coverage,fast_nearest_core = check_coverage("fast",core_fast_ids)

print("Calculating slow redirection coverage...")
slow_coverage,slow_nearest_core = check_coverage("slow",core_slow_ids)

coverage = pd.DataFrame(fast_coverage + slow_coverage)
coverage.to_csv("outputs/results/core_redirection_coverage.csv",index=False)


# Short summary table that can go into the report
summary_rows = []

for charger_type in ["fast","slow"]:

    type_coverage = coverage[coverage["charger_type"] == charger_type]

    if charger_type == "fast":
        nearest_core = fast_nearest_core
        core_ids = core_fast_ids
    else:
        nearest_core = slow_nearest_core
        core_ids = core_slow_ids

    # Excluded stations of this type and their distance to the nearest core station
    excluded_distances = []
    for station_id in nearest_core:
        if station_id not in core_ids:
            excluded_distances.append(nearest_core[station_id])

    for radius in radius_list:

        core_column = "core_within_" + str(radius) + "km"
        all_column = "all_within_" + str(radius) + "km"

        # How many excluded stations have a core station within this radius
        excluded_covered = 0
        for d in excluded_distances:
            if d is not None and d <= radius:
                excluded_covered = excluded_covered + 1

        if len(excluded_distances) > 0:
            excluded_covered_percent = excluded_covered / len(excluded_distances) * 100
        else:
            excluded_covered_percent = 0

        summary_rows.append({
            "charger_type": charger_type,
            "radius_km": radius,
            "core_stations": len(type_coverage),
            "avg_core_neighbours": round(type_coverage[core_column].mean(),2),
            "avg_all_neighbours": round(type_coverage[all_column].mean(),2),
            "percent_core_with_no_core_neighbour": round((type_coverage[core_column] == 0).mean() * 100,2),
            "percent_core_with_no_neighbour_at_all": round((type_coverage[all_column] == 0).mean() * 100,2),
            "excluded_stations": len(excluded_distances),
            "percent_excluded_with_core_station_nearby": round(excluded_covered_percent,2)
        })

coverage_summary = pd.DataFrame(summary_rows)
coverage_summary.to_csv("outputs/results/core_redirection_summary.csv",index=False)

print("\nRedirection coverage summary:")
print(coverage_summary.to_string(index=False))


# PART B: what do the excluded stations actually look like?

# The 204 network-wide bad hours found in File 05
bad_hour_values = pd.to_datetime(bad_hours["hour"]).tolist()

month_columns = [
    "sep_missing_percent",
    "oct_missing_percent",
    "nov_missing_percent",
    "dec_missing_percent",
    "jan_missing_percent",
    "feb_missing_percent"
]

# 7 days = 2016 five-minute readings
useful_run_rows = 7 * 288


# Same charger count check as File 04
def check_type_match(df,expected_fast,expected_slow):

    fast_busy_values = df["fast_busy"].tolist()
    fast_idle_values = df["fast_idle"].tolist()
    slow_busy_values = df["slow_busy"].tolist()
    slow_idle_values = df["slow_idle"].tolist()

    for i in range(len(df)):

        if pd.notna(fast_busy_values[i]) and pd.notna(fast_idle_values[i]):
            if fast_busy_values[i] + fast_idle_values[i] != expected_fast:
                return False

        if pd.notna(slow_busy_values[i]) and pd.notna(slow_idle_values[i]):
            if slow_busy_values[i] + slow_idle_values[i] != expected_slow:
                return False

    return True


# Same stuck-feed check as File 04 (unchanged for 288 readings = 24 hours)
def check_stuck_percent(availability_values):

    previous_value = None
    current_run = 0
    stuck_rows = 0

    for value in availability_values:

        if pd.isna(value):

            if current_run >= 288:
                stuck_rows = stuck_rows + current_run

            current_run = 0
            previous_value = None

        else:

            if previous_value is not None and value == previous_value:
                current_run = current_run + 1

            else:

                if current_run >= 288:
                    stuck_rows = stuck_rows + current_run

                current_run = 1

            previous_value = value

    if current_run >= 288:
        stuck_rows = stuck_rows + current_run

    valid_rows = 0
    for value in availability_values:
        if pd.notna(value):
            valid_rows = valid_rows + 1

    if valid_rows > 0:
        return (stuck_rows / valid_rows) * 100
    else:
        return 100


# Longest stretch of continuous non-missing readings
def longest_valid_run(availability_values):

    longest = 0
    current = 0

    for value in availability_values:

        if pd.notna(value):
            current = current + 1
            if current > longest:
                longest = current
        else:
            current = 0

    return longest


extended_rows = []

for charger_type in ["fast","slow"]:

    count_column = charger_type + "_count"
    target_column = charger_type + "_idle"

    if charger_type == "fast":
        core_ids = core_fast_ids
        other_core_ids = core_slow_ids
        nearest_core = fast_nearest_core
    else:
        core_ids = core_slow_ids
        other_core_ids = core_fast_ids
        nearest_core = slow_nearest_core

    # Stations with this charger type that are NOT in the core list for this type
    excluded = all_stations[(all_stations[count_column] > 0) & (~all_stations["station_id"].isin(core_ids))]

    print("\nChecking excluded", charger_type, "stations:", len(excluded))

    for station_id in excluded["station_id"].tolist():

        station = excluded[excluded["station_id"] == station_id]

        expected_fast = int(station["fast_count"].values[0])
        expected_slow = int(station["slow_count"].values[0])

        file_path = "data/raw/20220901-20230228_station-raw/charge_5min/" + str(station_id) + ".csv"
        df = pd.read_csv(file_path)

        df["time"] = pd.to_datetime(df["time"])
        df = df.sort_values("time")
        df = df.reset_index(drop=True)

        # Quality checks on the raw data, exactly like File 04
        type_match = check_type_match(df,expected_fast,expected_slow)
        stuck_percent = check_stuck_percent(df[target_column].tolist())

        if stuck_percent > 50:
            stuck_feed = True
        else:
            stuck_feed = False

        # Treat the 204 bad hours as missing, like File 05 does for core stations
        df["hour"] = df["time"].dt.floor("h")
        df.loc[df["hour"].isin(bad_hour_values),target_column] = pd.NA

        availability_values = df[target_column].tolist()

        valid_rows = df[target_column].notnull().sum()
        valid_percent = valid_rows / len(df) * 100
        longest_run = longest_valid_run(availability_values)

        # Put each station into one simple group
        if valid_rows == 0:
            category = "no valid data"
        elif type_match == False or stuck_feed == True:
            category = "clearly bad"
        elif longest_run >= useful_run_rows:
            category = "missing history but useful periods"
        else:
            category = "mostly missing, no long useful period"

        row = {
            "station_id": station_id,
            "charger_type": charger_type,
            "capacity": int(station[count_column].values[0]),
            "core_for_other_type": station_id in other_core_ids
        }

        for column in month_columns:
            row[column] = station[column].values[0]

        row["valid_rows"] = valid_rows
        row["valid_percent"] = round(valid_percent,2)
        row["longest_valid_days"] = round(longest_run / 288,2)
        row["type_match"] = type_match
        row["stuck_percent"] = round(stuck_percent,2)
        row["stuck_feed"] = stuck_feed

        if nearest_core[station_id] is None:
            row["nearest_core_km"] = None
        else:
            row["nearest_core_km"] = round(nearest_core[station_id],2)

        row["category"] = category

        extended_rows.append(row)


extended_summary = pd.DataFrame(extended_rows)
extended_summary.to_csv("outputs/results/extended_station_summary.csv",index=False)


print("\nExcluded stations by category:")
print(pd.crosstab(extended_summary["category"],extended_summary["charger_type"]))

# Useful excluded stations that are far from every core station fill real gaps for redirection
useful = extended_summary[extended_summary["category"] == "missing history but useful periods"]

print("\nUseful excluded stations with no core station of the same type within 5 km:")
print(pd.crosstab(useful["nearest_core_km"] > 5,useful["charger_type"]))

print("\nCapacity of useful excluded stations:")
print(useful.groupby("charger_type")["capacity"].sum())
