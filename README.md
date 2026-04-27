# Fraud Detection Website

This project converts your fraud detection machine learning code into a complete website with:
- a styled frontend
- a Flask backend API
- saved machine learning model files
- real-time prediction from form input

The website is based on your uploaded fraud detection code and keeps the same core ML idea: amount, frequency, and location-driven fraud scoring.

## Project Structure

```text
fraud_website/
├── backend/
│   ├── app.py
│   ├── train_model.py
│   ├── fraud_model.pkl
│   ├── scaler.pkl
│   └── model_features.pkl
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
├── requirements.txt
└── README.md
```

## How to Run in VS Code

### 1. Open the folder in VS Code
Open the extracted `fraud_website` folder.

### 2. Install dependencies
Open terminal inside VS Code and run:

```bash
pip install -r requirements.txt
```

### 3. Train and save the model
Move to backend folder and run:

```bash
cd backend
python train_model.py
```

This creates:
- `fraud_model.pkl`
- `scaler.pkl`
- `model_features.pkl`

### 4. Start the backend server
Still inside backend folder:

```bash
python app.py
```

The backend will run at:

```text
http://127.0.0.1:5000
```

### 5. Open the frontend
Now open `frontend/index.html` in your browser.

## Input Fields Used by the Model

The frontend sends these features to the backend:
- amount
- hour_of_day
- day_of_week
- hours_since_last_txn
- txn_count_24h
- txn_count_7d
- amount_deviation
- is_high_amount
- is_unusual_location
- location_changed

## Notes

- Run `train_model.py` once before running `app.py`.
- Your original code was notebook-style and included training, plotting, and dataset generation together. This project separates that into clean website-ready files.
- If you want, the next upgrade can add CSV upload, dashboard charts, login, and deployment.
