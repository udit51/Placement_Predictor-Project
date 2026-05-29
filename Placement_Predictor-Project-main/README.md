# AI Placement Predictor & Cognitive IQ Evaluator

An end-to-end Machine Learning classification and psychometric evaluation web application that assesses a student's placement probability based on their academic performance (**CGPA**) and cognitive intelligence (**IQ**).

---

## 📌 Project Overview

This repository hosts a predictive analysis system combining supervised machine learning with standardized cognitive assessments. The application is divided into two distinct functional modes to offer both quick evaluations and immersive cognitive profiling:

1. **Direct Prediction Mode (Phase 1):** Instantly inputs two parameters—**CGPA** (Academic GPA, range: 0.0 to 10.0) and **IQ** (Cognitive Score, range: 50 to 150)—and leverages a Flask-served machine learning model to calculate binary placement results ("Placed" or "Not Placed") alongside exact probability progress bars.
2. **MCQ Cognitive IQ Assessment Mode (Phase 2):** An immersive 20-question multiple-choice examination testing verbal analogies, quantitative aptitude, sequence logic, and spatial pattern recognition. Based on correct answers, it dynamically maps the user's intelligence score to a standard bell-curve distribution and couples it with their CGPA to compute final placement probabilities.

---

## 📊 Core Features & Architecture

### 1. Curated IQ Question Bank (20 Standardized MCQs)
The system contains an embedded database of 20 high-quality, college-level reasoning questions categorized into:
* **Logical Reasoning:** Syllabus deductions, truth/lie deduction scenarios, relative scoring order matrices.
* **Pattern Recognition:** Numerical sequences, alphabetical patterns, perfect squares, geometric progressions.
* **Quantitative Aptitude:** Rate-collaboration problems, distance-velocity parameters, mathematical puzzles.
* **Verbal Metrics:** Verbal analogies, vocabulary synonyms, lexical classifications ("odd-one-out").

### 2. Standard Bell-Curve IQ Score Mapping
To accurately calculate an IQ score based on MCQ results, the frontend utilizes standard psychometric score normalization. Assuming a typical average of 10 correct answers out of 20, and a standard deviation of 3.2, the Z-score is computed:
$$z = \frac{\text{Correct Answers} - 10}{3.2}$$
$$\text{IQ} = 100 + z \times 15$$
The final evaluated score is capped between a realistic $70$ and $150$.

### 3. Machine Learning Classification Pipeline
* **Model:** Logistic Regression classification.
* **Feature Scaler:** StandardScaler (Standardization of inputs).
* **API Probabilities:** The Flask backend queries the trained model and returns predictions alongside precise probability percentages (`predict_proba`).
* **Robust Sigmoid Fallback:** In cases where scikit-learn is missing or model loading fails, a fallback sigmoid heuristic serves calculations cleanly:
  $$\text{Score} = (\text{CGPA}_{\text{norm}} \times 0.72) + (\text{IQ}_{\text{norm}} \times 0.28)$$
  $$\text{Probability} = \frac{1}{1 + e^{-12(\text{Score} - 0.61)}}$$

---

## 🛠️ Tech Stack & Web Technologies

* **Backend:** Python, Flask, Flask-CORS, NumPy, Pandas, Scikit-Learn
* **Frontend:** HTML5, Vanilla CSS3 (Modern Dark-Mode Glassmorphism styling, SVG Radial Progress Bars, holographic laser sweep keyframes), Vanilla JavaScript (Bell-curve scoring engine, timers, canvas particle confetti systems)

---

## 🚀 How to Run the Application

### 1. Install Dependencies
Ensure you have Python 3 installed. Navigate to the project root and install the required modules:
```bash
pip install flask flask-cors numpy pandas scikit-learn
```

### 2. Generate Synthetic Data and Train Model (Optional)
If you wish to retrain the underlying Logistic Regression model from scratch:
```bash
# Generate synthetic CSV data
python dataset.py

# Train the model and save model.pkl/scaler.pkl
python model.py
```

### 3. Run the Predictive Server
Start the Flask development server:
```bash
python app.py
```
The server will initialize locally at: **[http://127.0.0.1:5000](http://127.0.0.1:5000)**. 

Open this address in any standard web browser to interact with both the Direct Prediction Mode and the MCQ Cognitive IQ Assessment wizard.
