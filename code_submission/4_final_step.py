


import pandas as pd
import numpy as np
import lightgbm as lgb
import tensorflow as tf
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import os

# --- 0. Configuration & Paths ---
DATA_DIR = 'C:\\Amazon\\Preprocessed_data' # <-- IMPORTANT: Use your correct local path
TRAIN_CSV = os.path.join(DATA_DIR, 'preprocessed_data_train.csv')
TEST_CSV = os.path.join(DATA_DIR, 'preprocessed_data_test.csv')

TEXT_FEAT_TRAIN = os.path.join(DATA_DIR, 'text_features_train.npy')
IMAGE_FEAT_TRAIN = os.path.join(DATA_DIR, 'image_features_train.npy')
TEXT_FEAT_TEST = os.path.join(DATA_DIR, 'text_features_test.npy')
IMAGE_FEAT_TEST = os.path.join(DATA_DIR, 'image_features_test.npy')
OUTPUT_FILENAME = os.path.join(DATA_DIR, 'test_output.csv')

# --- 1. Load, Prepare, and Scale Data ---
try:
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)
    
    X_text_train = np.load(TEXT_FEAT_TRAIN)
    X_img_train = np.load(IMAGE_FEAT_TRAIN)
    X_text_test = np.load(TEXT_FEAT_TEST)
    X_img_test = np.load(IMAGE_FEAT_TEST)
    
except FileNotFoundError as e:
    print(f"Error: Required file not found. Missing file: {e.filename}")
    exit()

# Target Variable (Log Transformation)
y_train_log = np.log1p(train_df['price'])

# Select structural features (excluding all original string/non-feature columns)
EXCLUDE_COLS = ['sample_id', 'catalog_content', 'image_link', 'price', 
                'ipq', 'title_len', 'total_len', 'unit_final']

feature_cols = [col for col in train_df.columns if col not in EXCLUDE_COLS]

X_struct_train = train_df[feature_cols].values.astype(float)
X_struct_test = test_df[feature_cols].values.astype(float)

# Concatenate and Scale
X_train_final = np.concatenate([X_struct_train, X_text_train, X_img_train], axis=1)
X_test_final = np.concatenate([X_struct_test, X_text_test, X_img_test], axis=1)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_final)
X_test_scaled = scaler.transform(X_test_final)

X_train_np = X_train_scaled
X_test_scaled_np = X_test_scaled
y_train_series = pd.Series(y_train_log)
input_dim = X_train_scaled.shape[1]

print(f"✅ Data loaded and scaled. Input dimension: {input_dim}")
print("-" * 50)


# --- 2. LightGBM Setup and Training ---

# LIGHTGBM PARAMETERS (Aggressive settings for competition baseline)
lgbm_params = {
    'objective': 'regression_l1', 
    'metric': 'rmse',
    'n_estimators': 3000,
    'learning_rate': 0.02,
    'num_leaves': 64,
    'max_depth': 10,
    'colsample_bytree': 0.7,
    'subsample': 0.7,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'n_jobs': -1,
    'seed': 42,
    'verbose': 50 # <--- VERBOSE SETTING for LGBM (logs every 50 boosting rounds)
}

print("Starting K-Fold LightGBM Training...")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
test_preds_log_lgbm = np.zeros(X_test_scaled_np.shape[0])

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_final, y_train_series)):
    print(f"\n--- LGBM Fold {fold+1}/5 ---")
    
    X_train, y_train = X_train_final[train_idx], y_train_series.iloc[train_idx]
    X_val, y_val = X_train_final[val_idx], y_train_series.iloc[val_idx]
    
    model = lgb.LGBMRegressor(**lgbm_params)
    
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              eval_metric='rmse',
              callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)]) # Early stopping logs only the final result
    
    test_preds_log_lgbm += model.predict(X_test_scaled_np, num_iteration=model.best_iteration_) / 5

P_LGBM = test_preds_log_lgbm
print(f"LGBM Predictions complete.")
print("-" * 50)


# --- 3. Deep Learning Setup and Training ---

def create_dnn_model(input_dim):
    model = Sequential([
        tf.keras.Input(shape=(input_dim,)),
        Dense(1024, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(256, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='linear') 
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model

def train_dnn_cv(X, y, X_test, input_dim, epochs=100, batch_size=64):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    test_preds_log_dnn = np.zeros(X_test.shape[0])

    X_test_tf = tf.constant(X_test, dtype=tf.float32)

    print("Starting K-Fold DNN Training on GPU...")
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        print(f"\n--- DNN Fold {fold+1}/5 ---")
        
        X_train_tf = tf.constant(X[train_idx], dtype=tf.float32)
        y_train_tf = tf.constant(y.iloc[train_idx].values.reshape(-1, 1), dtype=tf.float32) 
        X_val_tf = tf.constant(X[val_idx], dtype=tf.float32)
        y_val_tf = tf.constant(y.iloc[val_idx].values.reshape(-1, 1), dtype=tf.float32)
        
        model = create_dnn_model(input_dim)

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1) # <--- VERBOSE SETTING (logs learning rate changes)
        ]
        
        model.fit(X_train_tf, y_train_tf,
                  validation_data=(X_val_tf, y_val_tf),
                  epochs=epochs,
                  batch_size=batch_size,
                  callbacks=callbacks,
                  verbose=2) # <--- VERBOSE SETTING (logs one line per epoch)
        
        test_preds_log_dnn += model.predict(X_test_tf, verbose=0).flatten() / 5
        
    return test_preds_log_dnn

test_preds_log_dnn = train_dnn_cv(X_train_scaled, y_train_series, X_test_scaled_np, input_dim)
P_DNN = test_preds_log_dnn
print("DNN Predictions complete.")
print("-" * 50)


# --- 4. Ensemble and Submission ---

# Weighted Averaging (Baseline Ensemble: 65% LGBM, 35% DNN)
W_LGBM = 0.65
W_DNN = 0.35

combined_preds_log = (W_LGBM * P_LGBM) + (W_DNN * P_DNN)

# Inverse Transform and Submission
final_predicted_prices = np.expm1(combined_preds_log)
final_predicted_prices = np.maximum(0.01, final_predicted_prices)

submission_df = pd.DataFrame({
    'sample_id': test_df['sample_id'],
    'price': final_predicted_prices
})

submission_df.to_csv(OUTPUT_FILENAME, index=False)
print(f"\n✅ Final ensemble submission file created and saved to {OUTPUT_FILENAME}")