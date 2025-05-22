from flask import Flask, render_template, request
import numpy as np
import joblib
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load model and scaler
model = load_model("netflix_churn_model.h5")
scaler = joblib.load("netflix_scaler.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction_text = None
    confidence = None
    prediction_class = None

    if request.method == "POST":
        try:
            # Get input values from the form
            monthly_hours = float(request.form["monthly_hours"])
            data_usage = float(request.form["data_usage"])
            plan_price = float(request.form["plan_price"])

            # Prepare and scale input
            input_data = np.array([[monthly_hours, data_usage, plan_price]])
            input_scaled = scaler.transform(input_data)

            # Make prediction
            pred_prob = float(model.predict(input_scaled)[0][0])
            prediction_class = 1 if pred_prob >= 0.5 else 0
            confidence = round(pred_prob * 100, 2) if prediction_class == 1 else round((1 - pred_prob) * 100, 2)
            prediction_text = "⚠️ Likely to Churn" if prediction_class == 1 else "✅ Will Stay Subscribed"

        except Exception as e:
            prediction_text = f"Error: {str(e)}"

    return render_template(
        "index.html",
        prediction_text=prediction_text,
        confidence=confidence,
        prediction_class=prediction_class
    )

if __name__ == "__main__":
    app.run(debug=True)
