from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import pickle
import os

app = Flask(__name__)
CORS(app)

# Load model and scaler with a robust fallback
has_model = False
model = None
scaler = None

try:
    if os.path.exists("model.pkl") and os.path.exists("scaler.pkl"):
        model = pickle.load(open("model.pkl", "rb"))
        scaler = pickle.load(open("scaler.pkl", "rb"))
        has_model = True
        print("[INFO] Machine Learning model and scaler loaded successfully!")
    else:
        print("[WARN] Model or scaler files not found. Fallback heuristics will be used.")
except Exception as e:
    # Use generic characters to avoid Windows encoding issues
    print(f"[WARN] Error loading ML model: {str(e)[:100]}. Fallback heuristics will be used.")

def calculate_fallback(iq, cgpa):
    """
    Simulates the ML model logic: Placement probability is highly dependent
    on CGPA (70% weight) and cognitive ability/IQ (30% weight).
    """
    # Normalize inputs
    cgpa_norm = min(10.0, max(0.0, float(cgpa))) / 10.0
    iq_norm = min(150.0, max(50.0, float(iq))) / 150.0
    
    # Heuristic score calculation
    score = (cgpa_norm * 0.72) + (iq_norm * 0.28)
    
    # Sigmoid function centered around score=0.62 to represent placement threshold
    prob = 1.0 / (1.0 + np.exp(-12.0 * (score - 0.61)))
    prob_percent = round(prob * 100, 2)
    placed = 1 if prob_percent >= 50.0 else 0
    return placed, prob_percent

@app.route("/")
def home():
    return send_from_directory(".", "index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No input data provided"}), 400
            
        iq = float(data.get("IQ", 100))
        cgpa = float(data.get("CGPA", 7.5))
        
        if has_model:
            try:
                input_array = np.array([[iq, cgpa]])
                input_scaled = scaler.transform(input_array)
                prediction = int(model.predict(input_scaled)[0])
                
                # Get exact classification probability using LogisticRegression.predict_proba
                prob = model.predict_proba(input_scaled)[0][1]
                prob_percent = round(prob * 100, 2)
                
                # Double-check prediction matches probability threshold
                placed = 1 if prob_percent >= 50.0 else 0
            except Exception as ml_err:
                print(f"[WARN] Model prediction error: {ml_err}. Defaulting to fallback.")
                placed, prob_percent = calculate_fallback(iq, cgpa)
        else:
            placed, prob_percent = calculate_fallback(iq, cgpa)
            
        result_msg = "✅ Placement Ho Jayegi! 🎉" if placed == 1 else "❌ Placement Nhi Hogi. Skills pe dhyaan doooooooo!"
        
        return jsonify({
            "placed": placed,
            "probability": prob_percent,
            "message": result_msg
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
