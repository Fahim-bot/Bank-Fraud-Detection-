from pathlib import Path

import joblib
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS

BASE_DIR = Path(__file__).resolve().parent
app = Flask(__name__)
CORS(app)

model = joblib.load(BASE_DIR / 'fraud_model.pkl')
scaler = joblib.load(BASE_DIR / 'scaler.pkl')
model_features = joblib.load(BASE_DIR / 'model_features.pkl')


def build_feature_payload(data):
    amount = float(data.get('amount', 0))
    hour_of_day = int(data.get('hour_of_day', 0))
    day_of_week = int(data.get('day_of_week', 0))
    hours_since_last_txn = float(data.get('hours_since_last_txn', 0))
    txn_count_24h = int(data.get('txn_count_24h', 0))
    txn_count_7d = int(data.get('txn_count_7d', 0))
    amount_deviation = float(data.get('amount_deviation', 0))
    is_high_amount = int(data.get('is_high_amount', 0))
    is_unusual_location = int(data.get('is_unusual_location', 0))
    location_changed = int(data.get('location_changed', 0))

    return [
        amount,
        hour_of_day,
        day_of_week,
        hours_since_last_txn,
        txn_count_24h,
        txn_count_7d,
        amount_deviation,
        is_high_amount,
        is_unusual_location,
        location_changed,
    ]


def risk_level(probability):
    if probability < 0.3:
        return 'Low'
    if probability < 0.7:
        return 'Medium'
    return 'High'


@app.get('/')
def home():
    return jsonify({
        'message': 'Fraud Detection API is running.',
        'features': model_features,
    })


@app.post('/predict')
def predict():
    try:
        data = request.get_json(force=True)
        payload = np.array(build_feature_payload(data)).reshape(1, -1)
        scaled = scaler.transform(payload)
        probability = float(model.predict_proba(scaled)[0, 1])
        prediction = int(probability >= 0.5)

        return jsonify({
            'success': True,
            'prediction': prediction,
            'label': 'Fraud' if prediction == 1 else 'Normal',
            'probability': probability,
            'risk': risk_level(probability),
        })
    except Exception as exc:
        return jsonify({'success': False, 'error': str(exc)}), 400


if __name__ == '__main__':
    app.run(debug=True)
