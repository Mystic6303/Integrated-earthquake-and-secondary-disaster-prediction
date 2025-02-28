from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import requests
import numpy as np
import pandas as pd
import pickle
import joblib
from obspy.clients.fdsn import Client
from obspy import UTCDateTime
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import time
from obspy import UTCDateTime
from datetime import datetime, timedelta
app = Flask(__name__)
CORS(app)


# Define parameters for the earthquake catalog
starttime = UTCDateTime("2024-1-1")
#endtime = UTCDateTime.utcnow()
endtime = UTCDateTime("2024-12-12")

@app.route('/')
def home():
    return render_template('index.html')

#current_time = UTCDateTime.utcnow()
#starttime = current_time - timedelta(days=5)  # Fetch data for the last 24 hours
#endtime = current_time
def query_usgs_earthquake_by_bbox(min_latitude, max_latitude, min_longitude, max_longitude):
    timeout_seconds_first_url = 10 # Adjust as needed
    session_first_url = requests.Session()
    retries = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    session_first_url.mount('http://', HTTPAdapter(max_retries=retries))
    session_first_url.mount('https://', HTTPAdapter(max_retries=retries))
    url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
    response_first_url = session_first_url.get(url, timeout=timeout_seconds_first_url)
    response_first_url.raise_for_status()
    parameters = {
        "format": "geojson",
        "minlatitude": min_latitude,
        "maxlatitude": max_latitude,
        "minlongitude": min_longitude,
        "maxlongitude": max_longitude,
        "minmagnitude": 5.5,
        'starttime': starttime.strftime('%Y-%m-%dT%H:%M:%SZ'),  # Format starttime to ISO 8601 format
        'endtime': endtime.strftime('%Y-%m-%dT%H:%M:%SZ'), 
        "orderby": "time",
        "limit": 10
    }

    try:
        response = requests.get(url, params=parameters)
        response.raise_for_status()

        data = response.json()
        magnitudes = []
        earthquake_properties = []
        tsunami_properties = []
        if "features" in data and len(data["features"]) > 0:
            # Extract magnitudes and the last event's properties
            for feature in data["features"]:
                properties = feature.get("properties", {})
                geometry = feature.get("geometry", {}).get("coordinates", [])
                
                magnitude = properties.get("mag", "Unknown")
                if magnitude != "Unknown":
                    magnitudes.append(magnitude)

            # Get the last event's properties
            last_event = data["features"][-1]
            last_properties = last_event.get("properties", {})
            last_geometry = last_event.get("geometry", {}).get("coordinates", [])
            
            earthquake_properties = [
                last_properties.get("mag", 0),  # Magnitude  # Tsunami flag
                last_properties.get("sig", 0),
                last_properties.get("nst", 0),     # Significance
                last_properties.get("dmin", 0),
                last_properties.get("rms", 0),
                last_properties.get("gap", 0),        # Minimum distance
                last_geometry[2] if len(last_geometry) > 2 else 0  # Depth
                  # RMS
            ]
            tsunami_properties = [
                last_properties.get("mag", 0),  # Magnitude
                last_geometry[2] if len(last_geometry) > 2 else 0,  # Tsunami flag
                last_properties.get("sig", 0),     # Significance
                last_properties.get("dmin", 0),
                last_properties.get("rms", 0)   # Minimum distance
            ]

            mag = np.array(magnitudes)
            earthquake = np.array(earthquake_properties)
            tsunami = np.array(tsunami_properties)
            return mag, earthquake, tsunami
        else:
            print("No earthquake data found.")
            return np.array([]), np.array([])

    except requests.exceptions.RequestException as e:
        print(f"Error fetching earthquake data: {e}")
        return np.array([])


@app.route('/process', methods=['POST'])
def handle_coordinates():
    try:
        data = request.get_json()
        min_lat = float(data.get('min_latitude'))
        max_lat = float(data.get('max_latitude'))
        min_lon = float(data.get('min_longitude'))
        max_lon = float(data.get('max_longitude'))
        mag,earthquake,tsunami = query_usgs_earthquake_by_bbox(min_lat, max_lat, min_lon, max_lon)
        
        if mag.size == 0:
            return jsonify({'result': 'No earthquake data found in the specified range.'})

        with open('lstm.pkl', 'rb') as file:
         lstm_model = pickle.load(file)
        mag = np.array(mag).reshape(1, -1)   
        print(f"Magnitudes: {mag}")
        lstm_pred=lstm_model.predict(mag)
        lstm_pred = float(lstm_pred[0][0])
        print(f"lstm prediction {lstm_pred}")
        if lstm_pred>=0:
            with open("gradient_boosting_model.pkl", "rb") as model_file:
              gbm_model = pickle.load(model_file)# Load the scaler
            with open("earthquake_scaler.pkl", "rb") as scaler_file:
              scaler1 = pickle.load(scaler_file)
        
            with open("svm_model.pkl", "rb") as model_file:
              svm_model = pickle.load(model_file)# Load the scaler
            with open("tsunami_scaler.pkl", "rb") as scaler_file:
              scaler2 = pickle.load(scaler_file)
            
            np.set_printoptions(suppress=True)
            earthquake = np.array(earthquake).reshape(1, -1)   
            print(f"Earthquake properties: {earthquake}")
            tsunami = np.array(tsunami).reshape(1, -1)   
            print(f"Tsunami properties: {tsunami}")
            scaled_earthquake = scaler1.transform(earthquake)
            predictions1 = gbm_model.predict(scaled_earthquake)
            print("GBM Earthquake severity prediction:", predictions1)
            
            
            scaled_tsunami = scaler2.transform(tsunami)
            predictions2 = svm_model.predict(scaled_tsunami)
            print("SVM Tsunami prediction:", predictions2)
    except requests.exceptions.RequestException as e:
        print(f"Error processing data: {e}")
        return {'result': 'Error: Could not process data.'}, 500


if __name__ == '__main__':
    app.run(debug=True)
