from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split

feat = add_window_features(df)
# turn rule outputs into columns: s_velocity, s_amount, s_geo, s_offhours,
# plus raw: amount, hour_of_day, merchant_category (encoded), channel,
# distance_to_merchant, device_change, running_24h_amount, tx_of_day...

X = feat[FEATURE_COLS]
y = feat["is_fraud"]

# CRITICAL: split by time, not randomly — fraud detection must be evaluated
# on the future, never on shuffled rows.
cut = feat["timestamp"].quantile(0.8)
train, test = feat[feat.timestamp <= cut], feat[feat.timestamp > cut]

model = LGBMClassifier(
    n_estimators=500,
    learning_rate=0.03,
    scale_pos_weight=(y == 0).sum() / (y == 1).sum(),  # handle class imbalance
)
model.fit(train[FEATURE_COLS], train["is_fraud"])