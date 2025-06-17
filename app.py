from flask import Flask, render_template, request, redirect, url_for, flash
import os
import pandas as pd
import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

app = Flask(__name__)
app.secret_key = "secret_key"  # For flash messages

MODEL_FILE = "gradient_boosting_model.pkl"

# Train and save the model
def train_model(dataset_path):
    data = pd.read_csv(dataset_path)
    features = [
        'Age_Of_Mother', 'weight_before_preg', 'wt_before_delivery',
        'Height(cm)', 'BMI', 'Hemoglobin', 'PCOS', 'Heartbeat_Rate',
        'Motion_of_Baby', 'Stress_Level'
    ]
    target = 'Preterm_Delivery'
    data_cleaned = data.dropna(subset=[target])
    X = data_cleaned[features]
    y = data_cleaned[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    gb_model = GradientBoostingClassifier(random_state=42)
    gb_model.fit(X_train, y_train)
    joblib.dump(gb_model, MODEL_FILE)
    y_pred = gb_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

# Process sensor data
def process_sensor_files(gsr_file, pulse_file, motion_file):
    pulse_data = pd.read_csv(pulse_file)
    pulse_data['BPM'] = pd.to_numeric(pulse_data['BPM'].replace(r'\D', '', regex=True), errors='coerce')
    pulse_values = pulse_data['BPM'].dropna().mean()

    gsr_data = pd.read_csv(gsr_file)
    gsr_data['Voltage (V)'] = pd.to_numeric(gsr_data['Voltage (V)'], errors='coerce')
    gsr_values = gsr_data['Voltage (V)'].dropna().mean()

    motion_data = pd.read_csv(motion_file)
    motion_values = motion_data['Motion Detected'].mode()[0].lower()

    return {"Pulse": pulse_values, "GSR": gsr_values, "Motion": motion_values}

# Predict using sensor data
def predict_with_sensor_data(sensor_data):
    model = joblib.load(MODEL_FILE)
    default_values = {
        'Age_Of_Mother': 25,
        'weight_before_preg': 60,
        'wt_before_delivery': 70,
        'Height(cm)': 160,
        'BMI': 22.5,
        'Hemoglobin': 12,
        'PCOS': 0
    }
    feature_values = default_values.copy()
    feature_values['Heartbeat_Rate'] = sensor_data['Pulse']
    feature_values['Motion_of_Baby'] = sensor_data['Motion']
    feature_values['Stress_Level'] = sensor_data['GSR']
    X_new = pd.DataFrame([feature_values])
    prediction = model.predict(X_new)[0]
    prediction_proba = model.predict_proba(X_new)[0][1] * 100
    return prediction, prediction_proba

# Flask routes
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/train", methods=["POST"])
def train():
    dataset_path = request.form["dataset"]
    if not os.path.exists(dataset_path):
        flash("Dataset not found!")
        return redirect(url_for("home"))
    accuracy = train_model(dataset_path)
    flash(f"Model trained successfully with accuracy: {accuracy * 100:.2f}%")
    return redirect(url_for("home"))

@app.route("/predict", methods=["POST"])
def predict():
    gsr_file = request.files["gsr_file"]
    pulse_file = request.files["pulse_file"]
    motion_file = request.files["motion_file"]
    gsr_path = os.path.join("uploads", gsr_file.filename)
    pulse_path = os.path.join("uploads", pulse_file.filename)
    motion_path = os.path.join("uploads", motion_file.filename)
    gsr_file.save(gsr_path)
    pulse_file.save(pulse_path)
    motion_file.save(motion_path)

    sensor_data = process_sensor_files(gsr_path, pulse_path, motion_path)
    prediction, prediction_proba = predict_with_sensor_data(sensor_data)

    result = {
        "Pulse": sensor_data["Pulse"],
        "GSR": sensor_data["GSR"],
        "Motion": sensor_data["Motion"],
        "Prediction": "Preterm" if prediction == 1 else "Not Preterm",
        "Probability": f"{prediction_proba:.2f}%"
    }
    return render_template("result.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)






