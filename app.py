from flask import Flask, render_template, request, jsonify, redirect, url_for
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import random

app = Flask(__name__)

API_URL = "https://api.exchangerate-api.com/v4/latest/"


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/convert-with-me', methods=['GET', 'POST'])
def convert_with_me():
    # This route will handle the "Convert with me" button click
    # and redirect to the index page
    return redirect(url_for('index'))


@app.route('/convert', methods=['POST'])
def convert():
    data = request.json
    amount = float(data['amount'])
    from_currency = data['from']
    to_currency = data['to']

    response = requests.get(API_URL + from_currency)
    if response.status_code == 200:
        rates = response.json()["rates"]
        if to_currency in rates:
            converted_amount = round(amount * rates[to_currency], 2)
            result_text = f"The current exchange rate shows that {amount} {from_currency} equals {converted_amount} {to_currency}."
            return jsonify({"result": result_text})

    return jsonify({"result": "Error fetching exchange rates. Try again later."})


def get_historical_data(base_currency, target_currency):
    """
    Get historical exchange rate data
    Since the free API doesn't support historical data, we'll create a synthetic dataset
    based on the current rate with realistic variations to simulate historical data
    """
    # Get current exchange rate as baseline
    response = requests.get(API_URL + base_currency)

    if response.status_code != 200:
        return None

    current_rate = response.json()["rates"].get(target_currency)
    if not current_rate:
        return None

    # Create dates for the past 90 days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')

    # Generate synthetic rates with realistic patterns:
    # 1. Overall trend (up/down)
    # 2. Weekly patterns
    # 3. Random noise

    # Decide on a trend direction and magnitude (slight upward or downward trend)
    trend_factor = random.uniform(-0.0003, 0.0003)  # Small daily trend

    rates = []
    trend_component = []
    weekly_component = []
    noise_component = []

    # Start with a rate that's slightly different from current to create realistic history
    base_rate = current_rate * random.uniform(0.9, 1.1)

    for i, date in enumerate(date_range):
        # Trend component (cumulative)
        trend = i * trend_factor
        trend_component.append(trend)

        # Weekly component (markets often have weekly patterns)
        day_of_week = date.dayofweek
        weekly = 0.002 * np.sin(day_of_week * np.pi / 3.5)
        weekly_component.append(weekly)

        # Random noise component (markets are volatile)
        noise = random.uniform(-0.004, 0.004)
        noise_component.append(noise)

        # Combine components
        rate = base_rate * (1 + trend + weekly + noise)
        rates.append(rate)

    # Create DataFrame
    df = pd.DataFrame({
        'date': date_range,
        'rate': rates,
        'trend': trend_component,
        'weekly': weekly_component,
        'noise': noise_component
    })

    # Ensure the most recent rate is very close to the actual current rate for realism
    df.iloc[-1, df.columns.get_loc('rate')] = current_rate

    # Extract features from dates
    df['day'] = df['date'].dt.day
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['day_of_week'] = df['date'].dt.dayofweek
    df['day_of_year'] = df['date'].dt.dayofyear

    # Add time-based features
    df['days_from_start'] = (df['date'] - df['date'].min()).dt.days

    return df


def preprocess_data(df):
    """Preprocess the dataframe for model training"""
    # Drop components used for generation and the date column
    df = df.drop(['date', 'trend', 'weekly', 'noise'], axis=1, errors='ignore')

    # Check for missing values and fill them
    df.fillna(method='ffill', inplace=True)

    return df


def train_model(df):
    """Train a random forest model on the historical data"""
    # Features are all columns except 'rate'
    X = df.drop('rate', axis=1)
    y = df['rate']

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train the model - using RandomForest for better capture of non-linear patterns
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)

    return model, scaler, X.columns.tolist()


def predict_future_rates(model, scaler, feature_names, days_ahead):
    """Predict exchange rates for future days with varied predictions"""
    future_dates = []

    start_date = datetime.now()

    for i in range(1, days_ahead + 1):
        future_date = start_date + timedelta(days=i)
        future_dates.append({
            'day': future_date.day,
            'month': future_date.month,
            'year': future_date.year,
            'day_of_week': future_date.weekday(),
            'day_of_year': future_date.timetuple().tm_yday,
            'days_from_start': 90 + i  # 90 days of history + future days
        })

    # Convert to DataFrame
    future_df = pd.DataFrame(future_dates)

    # Ensure column order matches training data
    future_df = future_df[feature_names]

    # Scale features
    future_scaled = scaler.transform(future_df)

    # Make predictions
    predictions = model.predict(future_scaled)

    return predictions, future_dates


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        from_currency = data['from']
        to_currency = data['to']
        days_ahead = int(data['days'])

        if days_ahead > 30:
            return jsonify({"result": "Please limit predictions to 30 days ahead for better accuracy."})

        print(f"Received data: from={from_currency}, to={to_currency}, days={days_ahead}")

        # Get historical data
        df = get_historical_data(from_currency, to_currency)

        if df is not None and not df.empty:
            print("Generated historical data successfully")

            # Preprocess data
            processed_df = preprocess_data(df)

            # Train model
            model, scaler, feature_names = train_model(processed_df)

            # Predict future rates
            predictions, future_dates = predict_future_rates(model, scaler, feature_names, days_ahead)

            # Format results for display
            results = []
            for i, pred in enumerate(predictions):
                date_str = (datetime.now() + timedelta(days=i + 1)).strftime('%Y-%m-%d')
                pred_rate = round(float(pred), 4)
                results.append({
                    "date": date_str,
                    "rate": pred_rate,
                    "day": future_dates[i]['day_of_week']
                })

            if days_ahead == 1:
                result_text = f"Predicted exchange rate tomorrow: 1 {from_currency} = {results[0]['rate']} {to_currency}"
            else:
                result_text = f"Predicted exchange rate in {days_ahead} days: 1 {from_currency} = {results[-1]['rate']} {to_currency}"

                # Add additional details for multi-day predictions
                additional_info = "<br><br>Detailed forecast:<br>"
                for r in results:
                    day_name = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][r['day']]
                    additional_info += f"{r['date']} ({day_name}): 1 {from_currency} = {r['rate']} {to_currency}<br>"

                result_text += additional_info

            return jsonify({"result": result_text})
        else:
            print("Failed to fetch historical data")

    except Exception as e:
        print(f"Exception occurred: {e}")
        import traceback
        traceback.print_exc()

    return jsonify({"result": "Error in fetching or processing data. Please try again with different parameters."})


if __name__ == '__main__':
    app.run(debug=True)