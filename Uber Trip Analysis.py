# Uber Trip Analysis and Dashboard

## Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, jsonify
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px
import plotly.graph_objects as go

# Initialize Flask app
app = Flask(_name_)

# Load and preprocess data
def load_and_preprocess_data(filepath):
    data = pd.read_csv(filepath)
    data['Date/Time'] = pd.to_datetime(data['Date/Time'])
    data['Hour'] = data['Date/Time'].dt.hour
    data['Day'] = data['Date/Time'].dt.day
    data['DayOfWeek'] = data['Date/Time'].dt.dayofweek
    data['Month'] = data['Date/Time'].dt.month
    return data

# Train model
def train_model(data):
    data['Trips'] = 1  # Create a Trips column if not present
    X = data[['Hour', 'Day', 'DayOfWeek', 'Month', 'Lat', 'Lon']]
    y = data['Trips']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return model, mse, r2

# Create plots for dashboard
def create_plots(data):
    hourly_trips = data.groupby('Hour').size().reset_index(name='Trips')
    daily_trips = data.groupby('DayOfWeek').size().reset_index(name='Trips')

    fig_hourly = px.bar(hourly_trips, x='Hour', y='Trips', title='Trips per Hour')
    fig_daily = px.bar(daily_trips, x='DayOfWeek', y='Trips', title='Trips per Day of the Week')

    return fig_hourly, fig_daily

# Flask routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    filepath = request.form['filepath']
    data = load_and_preprocess_data(filepath)

    model, mse, r2 = train_model(data)
    fig_hourly, fig_daily = create_plots(data)

    hourly_plot = fig_hourly.to_html(full_html=False)
    daily_plot = fig_daily.to_html(full_html=False)

    response = {
        "mse": mse,
        "r2": r2,
        "hourly_plot": hourly_plot,
        "daily_plot": daily_plot
    }
    return jsonify(response)

if _name_ == '_main_':
    app.run(debug=True)
