import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from flask import Flask, render_template_string

# Load data from CSV
data = pd.read_csv('crimes.csv')

# Function to fit ARIMA model
def fit_arima(series):
    model = ARIMA(series, order=(5,1,0))
    fit_model = model.fit()
    return fit_model

# Function to fit SARIMA model
def fit_sarima(series):
    model = SARIMAX(series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    fit_model = model.fit()
    return fit_model

# Function to predict crime rate using the fitted model
def predict_crime_rate(model, steps):
    return model.get_forecast(steps=steps).predicted_mean

# Flask web application
app = Flask(__name__)

# Route to display the HTML page
@app.route('/')
def display_table():
    state_input = input("Enter the State/UT: ")
    purpose_input = input("Enter the Purpose: ")

    # Filter data based on user input
    filtered_data = data[(data['STATE/UT'] == state_input) & (data['Purpose'] == purpose_input)]

    # Use Total No. of cases reported as the time series data
    time_series_data = filtered_data.set_index('Year')['Total No. of cases reported']

    # Fit ARIMA and SARIMA models
    arima_model = fit_arima(time_series_data)
    sarima_model = fit_sarima(time_series_data)

    # Predict future crime rates (next 5 years)
    steps = 5
    arima_predictions = predict_crime_rate(arima_model, steps)
    sarima_predictions = predict_crime_rate(sarima_model, steps)

    # Create HTML table
    table_html = filtered_data.to_html()

    # Render HTML template
    template = """
    <html>
    <head>
        <title>Crime Rate Prediction</title>
    </head>
    <body>
        <h1>Crime Rate Prediction for {{ state_input }} - {{ purpose_input }}</h1>
        <h2>ARIMA Model Predictions:</h2>
        <p>{{ arima_predictions }}</p>
        <h2>SARIMA Model Predictions:</h2>
        <p>{{ sarima_predictions }}</p>
        <h2>Data Table:</h2>
        {{ table_html | safe }}
    </body>
    </html>
    """

    return render_template_string(template, state_input=state_input, purpose_input=purpose_input,
                                   arima_predictions=arima_predictions, sarima_predictions=sarima_predictions,
                                   table_html=table_html)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
