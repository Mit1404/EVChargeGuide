import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pathlib import Path

experiment = "slow_60"  # fast_30, fast_60, slow_30, slow_60

charger_type, horizon = experiment.split("_")
horizon = int(horizon)

# Features prepared in File 08
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


# Load fast charger 30-minute data
folder = "data/processed/model_data/" + charger_type + "/"
train = pd.read_csv(folder + "train_" + str(horizon) + ".csv")
validation = pd.read_csv(folder + "validation_" + str(horizon) + ".csv")
#test = pd.read_csv("data/processed/model_data/fast/test_30.csv")


print("Training rows:")
print(len(train))

print("\nValidation rows:")
print(len(validation))
# Convert time columns to datetime
train["time"] = pd.to_datetime(train["time"])
validation["time"] = pd.to_datetime(validation["time"])


# Get past charger availability from training and validation data
history = pd.concat([
    train[["station_id", "time", "current_idle"]],
    validation[["station_id", "time", "current_idle"]]
])
# Create historical features
past_features = [
    (5, "idle_5_min_ago"),
    (15, "idle_15_min_ago"),
    (30, "idle_30_min_ago"),
    (60, "idle_60_min_ago"),
    (1440 - horizon, "idle_same_time_yesterday"),
    (10080 - horizon, "idle_same_time_last_week")
]
for minutes, column in past_features:

    past = history.copy()

    past["time"] = past["time"] + pd.Timedelta(minutes=minutes)

    past = past.rename(columns={"current_idle": column})

    train = train.merge(past, on=["station_id", "time"], how="left")
    validation = validation.merge(past, on=["station_id", "time"], how="left")


# Calculate recent availability change
train["change_last_30_min"] = (
    train["current_idle"] - train["idle_30_min_ago"]
)

validation["change_last_30_min"] = (
    validation["current_idle"] - validation["idle_30_min_ago"]
)




#print("\nTest rows:")
#print(len(test))


# Separate input features and target
X_train = train[feature_columns]
y_train = train["target"] - train["current_idle"]
X_validation = validation[feature_columns]
y_validation = validation["target"] - validation["current_idle"]


#X_test = test[feature_columns]
#y_test = test["target"]


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


# Train model
model.fit(
    X_train,
    y_train,
    eval_set=[(X_validation,y_validation)],
    verbose=50
)


# Predict validation data
validation_prediction = (model.predict(X_validation)+ validation["current_idle"].to_numpy())

validation_mae = mean_absolute_error(
    validation["target"],
    validation_prediction
)

validation_rmse = mean_squared_error(
    validation["target"],
    validation_prediction
) ** 0.5

print("\nExperiment:", experiment)
print("\nValidation results:")

print("MAE:")
print(round(validation_mae,4))

print("RMSE:")
print(round(validation_rmse,4))

persistence_mae = mean_absolute_error(
    validation["target"],
    validation["current_idle"]
)

persistence_rmse = mean_squared_error(
    validation["target"],
    validation["current_idle"]
) ** 0.5

print("\nPersistence Baseline:")
print("MAE:", round(persistence_mae, 4))
print("RMSE:", round(persistence_rmse, 4))


# Predict test data
#test_prediction = model.predict(X_test)

#test_mae = mean_absolute_error(
 #   y_test,
  #  test_prediction
#)

#test_rmse = mean_squared_error(

 #   y_test,
  #  test_prediction
#) ** 0.5


#print("\nTest results:")

#print("MAE:")
#print(round(test_mae,4))

#print("RMSE:")
#print(round(test_rmse,4))


print("\nBest number of boosting rounds:")
print(model.best_iteration + 1)

# Save validation results
results_folder = Path("outputs/results")
results_folder.mkdir(parents=True, exist_ok=True)

results = pd.DataFrame([{
    "experiment": experiment,
    "training_rows": len(train),
    "validation_rows": len(validation),
    "validation_mae": round(validation_mae, 4),
    "validation_rmse": round(validation_rmse, 4),
    "persistence_mae": round(persistence_mae, 4),
    "persistence_rmse": round(persistence_rmse, 4),
    "best_round": model.best_iteration + 1
}])

results.to_csv(
    results_folder / f"xgboost_{experiment}_validation.csv",
    index=False
)


# Save the trained XGBoost model
models_folder = Path("models/xgboost")
models_folder.mkdir(parents=True, exist_ok=True)

model.save_model(str(models_folder / f"{experiment}_main.json"))

print("\nResults and model saved successfully!")