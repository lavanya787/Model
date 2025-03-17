from flask import Flask, request, jsonify
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the trained model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# Load the scaler (if you saved it during training)
# If you didn't save the scaler, you need to create it again in this file
# For demonstration, I'm assuming you saved it as 'scaler.pkl'
with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

@app.route('/')
def index():
    return "Welcome to the Real Estate Price Prediction API!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the request
        data = request.get_json(force=True)
        
        # Convert JSON data to a DataFrame
        input_data = pd.DataFrame(data, index=[0])
        
        # Preprocess the input data (similar to how you did for training)
        # Ensure to apply the same transformations as in your training code
        # Example: scaling, encoding, etc.
        
        # Scale the input data
        input_data_scaled = scaler.transform(input_data)
        
        # Make predictions
        prediction = model.predict(input_data_scaled)
        
        # Reverse the log transformation if necessary
        prediction = np.expm1(prediction)
        
        return jsonify({'predicted_price': prediction[0]})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)