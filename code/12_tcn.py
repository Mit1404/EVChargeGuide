import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv1D, Cropping1D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Change this variable to run different experiments
experiment = "fast_60"  # fast_30, fast_60, slow_30, slow_60

# Keep False until all choices are final, then run the February test ONCE
run_test = False
charger_type, horizon = experiment.split("_")
horizon = int(horizon)
# TCN settings
sequence_length = 12
epochs = 20
batch_size = 512
learning_rate = 0.001
# Make results reproducible
np.random.seed(42)
tf.random.set_seed(42)
# Features for every 5-minute step of the sequence
# longitude and latitude are NOT used: with capacity they work like a station ID
# The last two are the same "yesterday / last week" features XGBoost uses:
# free chargers at the same clock time as the TARGET, 1 day and 1 week earlier
feature_columns = [
    "current_idle",
    "current_busy",
    "capacity",
    "hour",
    "minute",
    "day_of_week",
    "idle_same_time_yesterday",
    "idle_same_time_last_week"
]

needed_columns = [
    "station_id",
    "time",
    "current_idle",
    "current_busy",
    "capacity",
    "hour",
    "minute",
    "day_of_week",
    "target"
]

# Same past values as File 10
past_features = [
    (1440 - horizon, "idle_same_time_yesterday"),
    (10080 - horizon, "idle_same_time_last_week")
]
folder = "data/processed/model_data/" + charger_type + "/"
results_folder = Path("outputs/results")
models_folder = Path("models/tcn")
results_folder.mkdir(parents=True,exist_ok=True)
models_folder.mkdir(parents=True,exist_ok=True)

# Add yesterday / last week values with a merge
def add_past_values(data, history):
    for minutes, column in past_features:
        past = history.copy()
        past["time"] = past["time"] + pd.Timedelta(minutes=minutes)
        past = past.rename(columns={"current_idle": column})
        data = data.merge(past, on=["station_id","time"], how="left")
    return data

# Scale features with TRAINING mean and std
# A missing yesterday / last week value becomes 0, which is the training mean after scaling
# (if we dropped those rows instead, TCN would be tested on different rows than XGBoost)
def scale_features(data):
    for column in feature_columns:
        data[column] = (data[column] - feature_means[column]) / feature_stds[column]
    data["idle_same_time_yesterday"] = data["idle_same_time_yesterday"].fillna(0)
    data["idle_same_time_last_week"] = data["idle_same_time_last_week"].fillna(0)
    return data

# Create temporal sequences for TCN
def create_sequences(data):
    X = []
    y = []
    current_values = []
    station_ids = data["station_id"].unique()
    for station_id in station_ids:
        station = data[data["station_id"] == station_id].copy()
        station = station.sort_values("time")
        station = station.reset_index(drop=True)
        times = station["time"].tolist()
        features = station[feature_columns].to_numpy(dtype="float32")
        target_change = (
                station["target"] - station["current_idle_raw"]
        ).to_numpy(dtype="float32")
        current_idle = station["current_idle_raw"].to_numpy(dtype="float32")
        for i in range(sequence_length - 1,len(station)):
            start = i - sequence_length + 1
            # 12 readings will cover exactly 55 minutes
            # 10:00, 10:05, ..., 10:55
            time_difference = times[i] - times[start]
            if time_difference != pd.Timedelta(minutes=55):
                continue
            sequence = features[start:i + 1]
            if np.isnan(sequence).any():
                continue
            X.append(sequence)
            y.append(target_change[i])
            current_values.append(current_idle[i])
    X = np.array(X,dtype="float32")
    y = np.array(y,dtype="float32")
    current_values = np.array(current_values,dtype="float32")
    return X,y,current_values
print("\nExperiment:")
print(experiment)
# Load training and validation data
train = pd.read_csv(
    folder + "train_" + str(horizon) + ".csv",
    usecols=needed_columns
)
validation = pd.read_csv(
    folder + "validation_" + str(horizon) + ".csv",
    usecols=needed_columns
)
train["time"] = pd.to_datetime(train["time"])
validation["time"] = pd.to_datetime(validation["time"])
# Past availability from training and validation data (only past values are looked up)
history = pd.concat([
    train[["station_id","time","current_idle"]],
    validation[["station_id","time","current_idle"]]
])

train = add_past_values(train, history)
validation = add_past_values(validation, history)

# January values are needed later for the February yesterday / last week features
january_history = validation.loc[
    validation["time"] >= "2023-01-24",
    ["station_id","time","current_idle"]
].copy()

del history

train["current_idle_raw"] = train["current_idle"]
validation["current_idle_raw"] = validation["current_idle"]
print("\nTraining rows:")
print(len(train))
print("\nValidation rows:")
print(len(validation))
# Calculate scaling values using TRAINING data only
feature_means = train[feature_columns].mean()
feature_stds = train[feature_columns].std()
# Prevent division by zero
for column in feature_columns:
    if feature_stds[column] == 0:
        feature_stds[column] = 1
train = scale_features(train)
validation = scale_features(validation)
# Save scaling values
scaling_results = pd.DataFrame({
    "feature": feature_columns,
    "mean": feature_means.values,
    "std": feature_stds.values
})
scaling_results.to_csv(
    results_folder / ("tcn_" + experiment + "_scaling.csv"),
    index=False
)
X_train,y_train,current_train = create_sequences(train)
X_validation,y_validation,current_validation = create_sequences(validation)
del train, validation
print("\nTraining sequences:")
print(len(X_train))
print("\nValidation sequences:")
print(len(X_validation))
print("\nSequence shape:")
print(X_train.shape)
# Create TCN model
model = Sequential([
    Input(shape=(sequence_length,len(feature_columns))),
    Conv1D(
        filters=16,
        kernel_size=3,
        padding="causal",
        dilation_rate=1,
        activation="relu"
    ),
    Conv1D(
        filters=16,
        kernel_size=3,
        padding="causal",
        dilation_rate=2,
        activation="relu"
    ),
    # Keep only the LAST time step (the current time), like a standard TCN
    # Cropping1D removes the first 11 steps; with causal convolutions the last step
    # has already seen the earlier steps
    Cropping1D(cropping=(sequence_length - 1, 0)),

    Flatten(),

    # Weights start at 0, so the first prediction is "no change" (= persistence)
    # Training then only learns corrections to persistence
    Dense(1, kernel_initializer="zeros")
])

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=learning_rate),
    loss="mae"
)


print("\nTCN model:")
model.summary()

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)
# Train the model
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_validation,y_validation),
    epochs=epochs,
    batch_size=batch_size,
    callbacks=[early_stopping]
)

# Predict validation data
validation_change = model.predict(
    X_validation,
    batch_size=batch_size
).flatten()
# Convert predicted change back to charger availability
validation_prediction = (validation_change + current_validation)
validation_actual = (y_validation + current_validation)
# Calculate validation errors
validation_mae = mean_absolute_error(validation_actual,validation_prediction)
validation_rmse = mean_squared_error(validation_actual,validation_prediction) ** 0.5
# Persistence baseline on the same rows
persistence_mae = mean_absolute_error(validation_actual,current_validation)
persistence_rmse = mean_squared_error(validation_actual,current_validation) ** 0.5
print("\nValidation Results:", experiment)
print("TCN MAE:")
print(round(validation_mae,4))
print("TCN RMSE:")
print(round(validation_rmse,4))
print("\nPersistence Baseline:")
print("MAE:")
print(round(persistence_mae,4))
print("RMSE:")
print(round(persistence_rmse,4))
# Save validation results
validation_results = pd.DataFrame([{
    "experiment": experiment,
    "training_sequences": len(X_train),
    "validation_sequences": len(X_validation),
    "validation_mae": round(validation_mae,4),
    "validation_rmse": round(validation_rmse,4),
    "persistence_mae": round(persistence_mae,4),
    "persistence_rmse": round(persistence_rmse,4),
    "epochs_trained": len(history.history["loss"])
}])
validation_results.to_csv(
    results_folder / ("tcn_" + experiment + "_validation.csv"),
    index=False)
# Save trained TCN model
model.save(models_folder / (experiment + "_main.keras"))
print("\nValidation results and model saved successfully!")
if run_test:
    # Free training memory before loading the test data
    del X_train, y_train, current_train
    # Load February test data
    test = pd.read_csv(
        folder + "test_" + str(horizon) + ".csv",
        usecols=needed_columns
    )
    test["time"] = pd.to_datetime(test["time"])
    # Yesterday / last week values come from late January and February itself (all in the past)
    history = pd.concat([
        january_history,
        test[["station_id","time","current_idle"]]
    ])
    test = add_past_values(test, history)
    test["current_idle_raw"] = test["current_idle"]
    print("\nFebruary test rows:")
    print(len(test))
    # Scale test data using TRAINING values only
    test = scale_features(test)
    X_test,y_test,current_test = create_sequences(test)
    print("\nTest sequences:")
    print(len(X_test))
    # Predict February data
    test_change = model.predict(
        X_test,
        batch_size=batch_size
    ).flatten()
    # Convert predicted change back to charger availability
    test_prediction = (test_change + current_test)
    test_actual = (y_test + current_test)
    # Calculate TCN test errors
    test_mae = mean_absolute_error(test_actual,test_prediction)
    test_rmse = mean_squared_error(test_actual,test_prediction) ** 0.5
    # Persistence baseline on exactly the same TCN rows
    persistence_test_mae = mean_absolute_error(test_actual,current_test)
    persistence_test_rmse = mean_squared_error(test_actual,current_test) ** 0.5
    print("\nFebruary Test Results:", experiment)
    print("TCN MAE:")
    print(round(test_mae,4))
    print("TCN RMSE:")
    print(round(test_rmse,4))
    print("\nPersistence Baseline:")
    print("MAE:")
    print(round(persistence_test_mae,4))
    print("RMSE:")
    print(round(persistence_test_rmse,4))
    # Save February test results
    test_results = pd.DataFrame([{
        "experiment": experiment,
        "test_sequences": len(X_test),
        "test_mae": round(test_mae,4),
        "test_rmse": round(test_rmse,4),
        "persistence_mae": round(persistence_test_mae,4),
        "persistence_rmse": round(persistence_test_rmse,4)
    }])
    test_results.to_csv(
        results_folder / ("tcn_" + experiment + "_test.csv"),
        index=False
    )
    print("\nTest results saved successfully!")