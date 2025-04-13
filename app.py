from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pickle

# Load model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Flask app
app = Flask(__name__)
CORS(app)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    iq = data.get("IQ")
    cgpa = data.get("CGPA")

    input_array = np.array([[iq, cgpa]])
    input_scaled = scaler.transform(input_array)
    prediction = model.predict(input_scaled)[0]

    result = "✅ Placement Ho Jayegi! 🎉" if prediction == 1 else "❌ Placement Nhi Hogi. Skills pe dhyaan doooooooo!"
    return jsonify({"message": result})

if __name__ == "__main__":
    app.run(debug=True)
