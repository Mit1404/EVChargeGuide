# EVChargeGuide
EVChargeGuide: ML-Based Charger Availability Prediction and Redirection

# Problem statment and Goal
If driver wants to use a Public EV charging station but by the time driver reaches there the chargers may not be available. This can lead to waiting, unnecessary travel, or reaching a station where the required charger type is not available. For example, a station may have free chargers, but none of them may be fast chargers.

The goal of EVChargeGuide is to use past charging-station data to learn usage patterns and predict how many fast and slow chargers are likely to be available at a specific station after a future time interval, such as 30 or 60 minutes. If the selected station is expected to have low or no suitable charger availability, the system will compare nearby stations and recommend a better option based on predicted charger availability, charger type, and distance .

# Proposed Methodology
We will use the UrbanEV real-world EV charging dataset, which contains station-level charging information collected at short time intervals(around 5 minutes). The data includes the number of busy and idle fast and slow chargers, station location, station chargers capacity, prices, and distance information between stations.

We will first clean and prepare the time-series data and create training, validation, and test sets using different time periods. Three machine learning models will be compared: XGBoost, Temporal Convolutional Network (TCN), and PatchTST, a Transformer-based time-series model. The models will predict how many fast and slow chargers are likely to be available at a station after a future time interval, such as 30 or 60 minutes. Their performance will be compared using metrics such as MAE and RMSE.

If the selected station is predicted to have low or no suitable charger availability, the system will check nearby stations and recommend an alternative based on predicted charger availability, charger type, and distance. We will also evaluate whether the recommended station actually had suitable chargers available during the test period.
