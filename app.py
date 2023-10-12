from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('financial_risk_model.pkl')

@app.route('/predict', methods=['POST'])
def predict_risk():
    data = request.json  # Get financial data from the POST request
    
    # Extract the features
    revenue = data['revenue']
    expenses = data['expenses']
    
    # Predict the risk score using the ML model
    risk_prediction = model.predict([[revenue, expenses]])[0]
    
    return jsonify({'risk': risk_prediction})

if __name__ == '__main__':
    app.run()