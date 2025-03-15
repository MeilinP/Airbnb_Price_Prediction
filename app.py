from flask import Flask, request, jsonify
import pandas as pd
import joblib
import json

app = Flask(__name__)

# Load trained model
model = joblib.load("airbnb_model.pkl")

# Define expected feature names (matching second version of Random Forest)
feature_names = [
    "latitude", "longitude", "minimum_nights", "number_of_reviews",
    "reviews_per_month", "availability_365", "room_type_Private room",
    "room_type_Shared room", "neighbourhood_group_Brooklyn",
    "neighbourhood_group_Manhattan", "neighbourhood_group_Queens",
    "neighbourhood_group_Staten Island"
]
print("Expected Feature Names:", model.feature_names_in_)
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON input from user
        data = request.get_json()
        df = pd.DataFrame([data])

        # Ensure input matches training features
        df = df.reindex(columns=feature_names, fill_value=0)

        # Make prediction
        predicted_price = model.predict(df)[0]

        return jsonify({"predicted_price": round(predicted_price, 2)})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
