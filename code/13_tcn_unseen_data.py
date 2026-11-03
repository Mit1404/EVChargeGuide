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
experiment = "fast_30"  # fast_30, fast_60, slow_30, slow_60

# Keep False until the workflow is checked, then run the February unseen test ONCE
run_test = False
charger_type, horizon = experiment.split("_")
horizon = int(horizon)
# Same TCN settings as File 12 (not retuned for unseen stations)
sequence_length = 12
epochs = 20
batch_size = 512
learning_rate = 0.001
# Make results reproducible
np.random.seed(42)
tf.random.set_seed(42)
# Same features as File 12
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

# Same past values as File 10 and File 12
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

# Scale features with KNOWN-STATION TRAINING mean and std
# A missing yesterday / last week value becomes 0, which is the training mean after scaling
def scale_features(data):
    for column in feature_columns:
        data[column] = (data[column] - feature_means[column]) / feature_stds[column]
    data["idle_same_time_yesterday"] = data["idle_same_time_yesterday"].fillna(0)
    data["idle_same_time_last_week"] = data["idle_same_time_last_week"].fillna(0)
    return data
# Use every training sequence for fast, every 3rd for slow
if charger_type == "fast":
    train_step = 1
else:
    train_step = 3

# Create temporal sequences for TCN (same as File 12)
def create_sequences(data, step):
    data = data.sort_values(["station_id","time"])
    data = data.reset_index(drop=True)
    if data[feature_columns].isnull().values.any():
        raise ValueError("Missing feature values found")
    station_ids = data["station_id"].unique()
    end_positions = {}
    total = 0
    for station_id in station_ids:
        station = data[data["station_id"] == station_id]
        times = station["time"].to_numpy()
        ends = np.arange(sequence_length - 1, len(station))
        starts = ends - sequence_length + 1
        valid = ((times[ends] - times[starts])== np.timedelta64(55, "m"))
        ends = ends[valid]
        offset = int(station_id) % step
        ends = ends[offset::step]
        end_positions[station_id] = ends
        total = total + len(ends)
    X = np.zeros(
        (total, sequence_length, len(feature_columns)),
        dtype="float32"
    )
    y = np.zeros(total,dtype="float32")
    current_values = np.zeros(total,dtype="float32")
    row = 0
    for station_id in station_ids:
        station = data[data["station_id"] == station_id]
        features = station[feature_columns].to_numpy(dtype="float32")
        current_idle = (
            station["current_idle_raw"]
            .to_numpy(dtype="float32")
        )
        target_change = (
            station["target"] - station["current_idle_raw"]
        ).to_numpy(dtype="float32")
        for i in end_positions[station_id]:
            X[row] = features[i - sequence_length + 1:i + 1]
            y[row] = target_change[i]
            current_values[row] = current_idle[i]
            row = row + 1
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
# Keep only KNOWN stations for training and model selection
train = train[train["holdout_station"] == False].copy()
validation = validation[validation["holdout_station"] == False].copy()
known_station_ids = set(train["station_id"].unique())
known_validation_ids = set(validation["station_id"].unique())
train["time"] = pd.to_datetime(train["time"])
validation["time"] = pd.to_datetime(validation["time"])
# Past availability of known stations only
history = pd.concat([
    train[["station_id","time","current_idle"]],
    validation[["station_id","time","current_idle"]]
])

train = add_past_values(train, history)
validation = add_past_values(validation, history)

del history

train["current_idle_raw"] = train["current_idle"]
validation["current_idle_raw"] = validation["current_idle"]
print("\nKnown training stations:")
print(len(known_station_ids))
print("\nKnown validation stations:")
print(len(known_validation_ids))
print("\nTraining rows:")
print(len(train))
print("\nValidation rows:")
print(len(validation))
# Calculate scaling values using KNOWN-STATION TRAINING data only
feature_means = train[feature_columns].mean()
feature_stds = train[feature_columns].std()
# Prevent division by zero
for column in feature_columns:
    if feature_stds[column] == 0:
        feature_stds[column] = 1
train = scale_features(train)
validation = scale_features(validation)
# Save unseen-model scaling values (different from the main TCN scaling)
scaling_results = pd.DataFrame({
    "feature": feature_columns,
    "mean": feature_means.values,
    "std": feature_stds.values
})
scaling_results.to_csv(
    results_folder / ("tcn_" + experiment + "_unseen_scaling.csv"),
    index=False
)
X_train,y_train,current_train = create_sequences(train,train_step)
X_validation,y_validation,current_validation = create_sequences(validation,1)
del train, validation
print("\nTraining sequences:")
print(len(X_train))
print("\nValidation sequences:")
print(len(X_validation))
print("\nSequence shape:")
print(X_train.shape)
# Same finalized TCN model as File 12
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
    Conv1D(
        filters=16,
        kernel_size=3,
        padding="causal",
        dilation_rate=4,
        activation="relu"
    ),
    # Keep only the LAST time step (the current time)
    Cropping1D(cropping=(sequence_length - 1, 0)),

    Flatten(),

    # Weights start at 0, so the first prediction is "no change" (= persistence)
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
# Train on known stations, early stopping on known January stations
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_validation,y_validation),
    epochs=epochs,
    batch_size=batch_size,
    callbacks=[early_stopping]
)

# Predict known-station validation data
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
validation_persistence_mae = mean_absolute_error(validation_actual,current_validation)
validation_persistence_rmse = mean_squared_error(validation_actual,current_validation) ** 0.5
print("\nKnown-station Validation Results:", experiment)
print("TCN MAE:")
print(round(validation_mae,4))
print("TCN RMSE:")
print(round(validation_rmse,4))
print("\nPersistence Baseline:")
print("MAE:")
print(round(validation_persistence_mae,4))
print("RMSE:")
print(round(validation_persistence_rmse,4))
# Results that exist at this point
results = {
    "experiment": experiment,
    "training_sequences": len(X_train),
    "train_step": train_step,
    "known_training_stations": len(known_station_ids),
    "known_validation_stations": len(known_validation_ids),
    "validation_sequences": len(X_validation),
    "validation_mae": round(validation_mae,4),
    "validation_rmse": round(validation_rmse,4),
    "validation_persistence_mae": round(validation_persistence_mae,4),
    "validation_persistence_rmse": round(validation_persistence_rmse,4),
    "epochs_trained": len(history.history["loss"])
}
# Save unseen-station TCN model
model.save(models_folder / (experiment + "_unseen.keras"))
if run_test:
    # Free training memory before loading the test data
    del X_train, y_train, current_train
    # Load February test data and keep only HELD-OUT stations
    test = pd.read_csv(
        folder + "test_" + str(horizon) + ".csv",
        usecols=needed_columns
    )
    test = test[test["holdout_station"] == True].copy()
    test["time"] = pd.to_datetime(test["time"])
    unseen_station_ids = set(test["station_id"].unique())
    # Check that unseen stations were not used in training
    for station in unseen_station_ids:
        if station in known_station_ids:
            raise ValueError("Training and unseen stations overlap")
    print("\nUnseen test stations:")
    print(len(unseen_station_ids))
    print("\nUnseen test rows:")
    print(len(test))
    # Late-January availability of the held-out stations
    # (only used as past input for yesterday / last week, never for training)
    january = pd.read_csv(
        folder + "validation_" + str(horizon) + ".csv",
        usecols=["station_id","holdout_station","time","current_idle"]
    )
    january = january[january["holdout_station"] == True].copy()
    january["time"] = pd.to_datetime(january["time"])
    january = january[january["time"] >= "2023-01-24"]
    # Yesterday / last week values come from late January and February itself (all in the past)
    history = pd.concat([
        january[["station_id","time","current_idle"]],
        test[["station_id","time","current_idle"]]
    ])
    test = add_past_values(test, history)
    del january, history
    test["current_idle_raw"] = test["current_idle"]
    # Scale test data using KNOWN-STATION TRAINING values only
    test = scale_features(test)
    X_test,y_test,current_test = create_sequences(test,1)
    print("\nUnseen test sequences:")
    print(len(X_test))
    # Predict February data for unseen stations
    test_change = model.predict(
        X_test,
        batch_size=batch_size
    ).flatten()
    # Convert predicted change back to charger availability
    test_prediction = (test_change + current_test)
    test_actual = (y_test + current_test)
    # Calculate TCN unseen test errors
    test_mae = mean_absolute_error(test_actual,test_prediction)
    test_rmse = mean_squared_error(test_actual,test_prediction) ** 0.5
    # Persistence baseline on exactly the same unseen sequences
    persistence_mae = mean_absolute_error(test_actual,current_test)
    persistence_rmse = mean_squared_error(test_actual,current_test) ** 0.5
    print("\nUnseen-station Test Results:", experiment)
    print("TCN MAE:")
    print(round(test_mae,4))
    print("TCN RMSE:")
    print(round(test_rmse,4))
    print("\nPersistence Baseline:")
    print("MAE:")
    print(round(persistence_mae,4))
    print("RMSE:")
    print(round(persistence_rmse,4))
    results["unseen_test_stations"] = len(unseen_station_ids)
    results["test_sequences"] = len(X_test)
    results["test_mae"] = round(test_mae,4)
    results["test_rmse"] = round(test_rmse,4)
    results["persistence_mae"] = round(persistence_mae,4)
    results["persistence_rmse"] = round(persistence_rmse,4)
# Save unseen-station results
pd.DataFrame([results]).to_csv(
    results_folder / ("tcn_" + experiment + "_unseen.csv"),
    index=False
)
print("\nUnseen-station results and model saved successfully!")
