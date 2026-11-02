
import pandas as pd
from pathlib import Path
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# Change this variable to run different experiments
experiment = "slow_30"  # fast_30, fast_60, slow_30, slow_60

charger_type, horizon = experiment.split("_")
horizon = int(horizon)

folder = "data/processed/model_data/" + charger_type + "/"

results_folder = Path("outputs/results")
models_folder = Path("models/xgboost")

#if these two folders don't exist then uncomment these lines
#results_folder.mkdir(parents=True, exist_ok=True)
#models_folder.mkdir(parents=True, exist_ok=True)


# Features used in File 10
feature_columns = [
    "current_idle",
    "current_busy",
    "capacity",
    "hour",
    "minute",
    "day_of_week",
    "idle_5_min_ago",
    "idle_15_min_ago",
    "idle_30_min_ago",
    "idle_60_min_ago",
    "change_last_30_min",
    "idle_same_time_yesterday",
    "idle_same_time_last_week"
]


# Historical features
past_features = [
    (5, "idle_5_min_ago"),
    (15, "idle_15_min_ago"),
    (30, "idle_30_min_ago"),
    (60, "idle_60_min_ago"),
    (1440 - horizon, "idle_same_time_yesterday"),
    (10080 - horizon, "idle_same_time_last_week")
]


needed_columns = [
    "station_id",
    "holdout_station",
    "time",
    "current_idle",
    "current_busy",
    "capacity",
    "hour",
    "minute",
    "day_of_week",
    "target"
]

data_types = {
    "station_id": "int32",
    "current_idle": "float32",
    "current_busy": "float32",
    "capacity": "float32",
    "hour": "int8",
    "minute": "int8",
    "day_of_week": "int8",
    "target": "float32"
}


# Add previous charger availability values
def add_history(data, past_values):

    for minutes, column in past_features:

        past_times = data["time"] - pd.Timedelta(minutes=minutes)

        keys = pd.MultiIndex.from_arrays(
            [data["station_id"], past_times],
            names=["station_id", "time"]
        )

        data[column] = past_values.reindex(keys).to_numpy()

    data["change_last_30_min"] = (
        data["current_idle"] - data["idle_30_min_ago"]
    )

    return data


# Load training and validation data
train = pd.read_csv(
    folder + "train_" + str(horizon) + ".csv",
    usecols=needed_columns,
    dtype=data_types,
    parse_dates=["time"]
)

validation = pd.read_csv(
    folder + "validation_" + str(horizon) + ".csv",
    usecols=needed_columns,
    dtype=data_types,
    parse_dates=["time"]
)


# Select only known stations
train = train[train["holdout_station"] == False].copy()
validation = validation[validation["holdout_station"] == False].copy()

known_station_ids = set(train["station_id"].unique())

training_rows = len(train)

print("\nExperiment:", experiment)

print("\nKnown training stations:")
print(len(known_station_ids))

print("\nTraining rows:")
print(len(train))

print("\nValidation rows:")
print(len(validation))


# Create historical features for known stations
history = pd.concat([
    train[["station_id", "time", "current_idle"]],
    validation[["station_id", "time", "current_idle"]]
], ignore_index=True)

past_values = history.set_index(
    ["station_id", "time"]
)["current_idle"]

if not past_values.index.is_unique:
    raise ValueError("Duplicate station timestamps found")

del history

train = add_history(train, past_values)
validation = add_history(validation, past_values)

del past_values


# Separate features and target
X_train = train[feature_columns].astype("float32")
y_train = train["target"] - train["current_idle"]

X_validation = validation[feature_columns].astype("float32")
y_validation = validation["target"] - validation["current_idle"]


# Create XGBoost regression model
model = XGBRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    tree_method="hist",
    objective="reg:absoluteerror",
    eval_metric="mae",
    early_stopping_rounds=50,
    random_state=42
)


# Train model on known stations
model.fit(
    X_train,
    y_train,
    eval_set=[(X_validation, y_validation)],
    verbose=50
)


# Predict validation data
validation_prediction = (
    model.predict(X_validation)
    + validation["current_idle"].to_numpy()
)

validation_mae = mean_absolute_error(
    validation["target"],
    validation_prediction
)

validation_rmse = mean_squared_error(
    validation["target"],
    validation_prediction
) ** 0.5


print("\nKnown-station Validation Results:")

print("MAE:")
print(round(validation_mae, 4))

print("RMSE:")
print(round(validation_rmse, 4))

print("\nBest number of boosting rounds:")
print(model.best_iteration + 1)


# Save unseen-station model
model.save_model(
    str(models_folder / (experiment + "_unseen.json"))
)


# Free training data memory before testing
del train, validation
del X_train, y_train, X_validation, y_validation
del validation_prediction


# Load February test data
test = pd.read_csv(
    folder + "test_" + str(horizon) + ".csv",
    usecols=needed_columns,
    dtype=data_types,
    parse_dates=["time"]
)


# Select only unseen stations
test = test[test["holdout_station"] == True].copy()

unseen_station_ids = set(test["station_id"].unique())


# Check that unseen stations were not used in training
for station in unseen_station_ids:
    if station in known_station_ids:
        raise ValueError("Training and unseen stations overlap")


print("\nUnseen test stations:")
print(len(unseen_station_ids))

print("\nUnseen test rows:")
print(len(test))


# Load January history for unseen stations
january = pd.read_csv(
    folder + "validation_" + str(horizon) + ".csv",
    usecols=[
        "station_id",
        "holdout_station",
        "time",
        "current_idle"
    ],
    dtype={
        "station_id": "int32",
        "current_idle": "float32"
    },
    parse_dates=["time"]
)

january = january[
    (january["holdout_station"] == True)
    & (january["time"] >= "2023-01-25")
].copy()


# Create historical features for unseen stations
history = pd.concat([
    january[["station_id", "time", "current_idle"]],
    test[["station_id", "time", "current_idle"]]
], ignore_index=True)

past_values = history.set_index(
    ["station_id", "time"]
)["current_idle"]

if not past_values.index.is_unique:
    raise ValueError("Duplicate station timestamps found")

del january, history

test = add_history(test, past_values)

del past_values


# Separate test features and target
X_test = test[feature_columns].astype("float32")
y_test = test["target"]


# Predict future charger availability
test_prediction = (
    model.predict(X_test)
    + test["current_idle"].to_numpy()
)


# Calculate XGBoost test errors
test_mae = mean_absolute_error(
    y_test,
    test_prediction
)

test_rmse = mean_squared_error(
    y_test,
    test_prediction
) ** 0.5


# Persistence baseline on unseen stations
persistence_mae = mean_absolute_error(
    y_test,
    test["current_idle"]
)

persistence_rmse = mean_squared_error(
    y_test,
    test["current_idle"]
) ** 0.5


print("\nUnseen-station Test Results:", experiment)

print("XGBoost MAE:")
print(round(test_mae, 4))

print("XGBoost RMSE:")
print(round(test_rmse, 4))

print("\nPersistence Baseline:")

print("MAE:")
print(round(persistence_mae, 4))

print("RMSE:")
print(round(persistence_rmse, 4))


# Save unseen-station results
results = pd.DataFrame([{
    "experiment": experiment,
    "training_rows": training_rows,
    "unseen_stations": len(unseen_station_ids),
    "test_rows": len(test),
    "validation_mae": round(validation_mae, 4),
    "validation_rmse": round(validation_rmse, 4),
    "test_mae": round(test_mae, 4),
    "test_rmse": round(test_rmse, 4),
    "persistence_mae": round(persistence_mae, 4),
    "persistence_rmse": round(persistence_rmse, 4),
    "best_round": model.best_iteration + 1
}])

results.to_csv(
    results_folder / ("xgboost_" + experiment + "_unseen.csv"),
    index=False
)

print("\nUnseen-station results and model saved successfully!")
