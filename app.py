import joblib
import pandas as pd
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Random Forest (trained on SCALED data) and the scaler it was trained with
model = joblib.load("customer_spending.pkl")
scaler = joblib.load("scaler.pkl")

FEATURE_COLUMNS = ["Annual Income (k$)", "Spending Score (1-100)"]

cluster_names = {
    0: "Steady Regulars",
    1: "VIP Shoppers",
    2: "Trend Chasers",
    3: "Cautious Wealthy",
    4: "Thrifty Savers",
}


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        annual_income = float(data["annual_income"])
        spending_score = float(data["spending_score"])

        features = pd.DataFrame([[annual_income, spending_score]], columns=FEATURE_COLUMNS)
        features_scaled = scaler.transform(features)

        predicted_cluster_id = int(model.predict(features_scaled)[0])

        # How sure the model is: share of trees voting for the winning group
        probabilities = model.predict_proba(features_scaled)[0]
        confidence = float(max(probabilities))

        return jsonify({
            "predicted_cluster_id": predicted_cluster_id,
            "predicted_cluster_name": cluster_names.get(predicted_cluster_id, "Unknown"),
            "confidence": round(confidence, 3),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)