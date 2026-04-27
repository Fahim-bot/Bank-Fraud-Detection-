import warnings
warnings.filterwarnings('ignore')

from datetime import datetime, timedelta
import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler

np.random.seed(42)

MODEL_FEATURES = [
    'amount',
    'hour_of_day',
    'day_of_week',
    'hours_since_last_txn',
    'txn_count_24h',
    'txn_count_7d',
    'amount_deviation',
    'is_high_amount',
    'is_unusual_location',
    'location_changed'
]


def generate_fraud_dataset(n_transactions=15000):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)
    timestamps = [
        start_date + timedelta(seconds=np.random.randint(0, 90 * 24 * 3600))
        for _ in range(n_transactions)
    ]
    timestamps.sort()

    user_ids = np.random.choice(range(1, 1001), n_transactions, replace=True)
    amounts = []
    fraud_flags = []
    locations = []

    for i in range(n_transactions):
        rand = np.random.random()
        is_fraud = 0
        amount = 0.0
        location = 0

        if rand < 0.05:
            is_fraud = 1
            pattern_rand = np.random.random()
            if pattern_rand < 0.4:
                amount = np.random.uniform(2000, 10000)
                location = np.random.choice([0, 1, 2, 3])
            elif pattern_rand < 0.75:
                amount = np.random.uniform(100, 800)
                location = np.random.choice([2, 3])
            else:
                amount = np.random.uniform(20, 150)
                location = np.random.choice([0, 1, 2, 3])
        else:
            amount = min(np.random.exponential(150) + 20, 3000)
            location = np.random.choice([0, 1], p=[0.6, 0.4])

        amounts.append(float(amount))
        fraud_flags.append(is_fraud)
        locations.append(int(location))

    location_map = {0: 'Home', 1: 'Regular_City', 2: 'New_City', 3: 'Foreign'}

    return pd.DataFrame({
        'user_id': user_ids,
        'timestamp': timestamps,
        'amount': amounts,
        'location_code': locations,
        'location': [location_map[i] for i in locations],
        'is_fraud': fraud_flags,
    })


def create_frequency_features(df):
    df = df.sort_values(['user_id', 'timestamp']).copy()

    df['hour_of_day'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['hours_since_last_txn'] = (
        df.groupby('user_id')['timestamp'].diff().dt.total_seconds() / 3600
    )
    df['hours_since_last_txn'] = df['hours_since_last_txn'].fillna(999)

    def txn_count_within_hours(row, hours):
        mask = (
            (df['user_id'] == row['user_id'])
            & (df['timestamp'] > row['timestamp'] - timedelta(hours=hours))
            & (df['timestamp'] <= row['timestamp'])
        )
        return int(df.loc[mask].shape[0])

    def avg_amount_within_hours(row, hours):
        mask = (
            (df['user_id'] == row['user_id'])
            & (df['timestamp'] > row['timestamp'] - timedelta(hours=hours))
            & (df['timestamp'] < row['timestamp'])
        )
        previous = df.loc[mask, 'amount']
        return float(previous.mean()) if len(previous) > 0 else float(row['amount'])

    df['txn_count_24h'] = df.apply(lambda row: txn_count_within_hours(row, 24), axis=1)
    df['txn_count_7d'] = df.apply(lambda row: txn_count_within_hours(row, 24 * 7), axis=1)
    df['avg_amount_24h'] = df.apply(lambda row: avg_amount_within_hours(row, 24), axis=1)

    user_avg_amount = df.groupby('user_id')['amount'].transform('mean')
    df['amount_deviation'] = (df['amount'] - user_avg_amount) / (user_avg_amount + 1)
    df['is_high_amount'] = (df['amount'] > 3 * user_avg_amount).astype(int)
    df['prev_location'] = df.groupby('user_id')['location_code'].shift(1)
    df['location_changed'] = (df['location_code'] != df['prev_location']).astype(int)
    df['location_changed'] = df['location_changed'].fillna(0)
    df['is_unusual_location'] = df['location_code'].isin([2, 3]).astype(int)

    return df


def train_and_save_model():
    print('Generating dataset...')
    df = generate_fraud_dataset(12000)
    df = create_frequency_features(df)

    X = df[MODEL_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = df['is_fraud']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42, stratify=y
    )

    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    param_dist = {
        'n_estimators': [50, 100, 150],
        'max_depth': [10, 15, 20],
        'min_samples_split': [5, 10],
        'min_samples_leaf': [2, 4],
    }

    X_train_small, _, y_train_small, _ = train_test_split(
        X_train_resampled,
        y_train_resampled,
        train_size=min(5000, len(X_train_resampled)),
        random_state=42,
        stratify=y_train_resampled,
    )

    search = RandomizedSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_distributions=param_dist,
        n_iter=10,
        cv=3,
        scoring='roc_auc',
        n_jobs=1,
        random_state=42,
        verbose=1,
    )
    search.fit(X_train_small, y_train_small)

    model = RandomForestClassifier(**search.best_params_, random_state=42)
    model.fit(X_train_resampled, y_train_resampled)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)

    print(f'Best parameters: {search.best_params_}')
    print(f'ROC-AUC: {auc:.4f}')
    print('Classification Report:')
    print(classification_report(y_test, y_pred))
    print('Confusion Matrix:')
    print(confusion_matrix(y_test, y_pred))

    joblib.dump(model, 'fraud_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(MODEL_FEATURES, 'model_features.pkl')
    print('Saved fraud_model.pkl, scaler.pkl, and model_features.pkl')


if __name__ == '__main__':
    train_and_save_model()
