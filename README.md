# Customer Spending Segmentation

A machine learning project that uses a trained Random Forest model to group customers into spending segments based on their annual income and spending score. The model is integrated into a Flask API that returns the predicted customer segment and a confidence score.

## Project Overview

Customer segmentation helps businesses understand different customer groups and tailor their marketing strategies, services, and engagement efforts. This project uses two customer attributes to assign each customer to one of five predefined segments.

## Features

- Customer segmentation using a trained Random Forest model.
- StandardScaler preprocessing.
- Flask API for making predictions.
- Five named customer segments.
- Prediction confidence score based on the model's class probabilities.
- JSON-based request and response.

## Technology Stack

- Python
- Flask
- pandas
- scikit-learn
- Joblib
- Random Forest

## Customer Segments

The model maps its predicted cluster IDs to the following names:

| Cluster ID | Segment Name |
|---|---|
| 0 | Steady Regulars |
| 1 | VIP Shoppers |
| 2 | Trend Chasers |
| 3 | Cautious Wealthy |
| 4 | Thrifty Savers |

These names are descriptive labels assigned in the application code.

## Input Features

The model uses two features:

- **Annual Income (k$):** Customer's annual income, represented in thousands of dollars.
- **Spending Score (1–100):** Customer spending score on a scale from 1 to 100.

## Project Structure

```text
customer-spending-segmentation/
├── app.py
├── customer_spending.pkl
├── scaler.pkl
├── templates/
│   └── index.html
└── README.md
```

This is an example structure. Your actual filenames and folders may differ.

## Getting Started

### Prerequisites

- Python
- pip
- The trained model and scaler files

### Install Dependencies

```bash
pip install flask pandas scikit-learn joblib
```

### Run the API

Ensure the model and scaler filenames match the paths used in your Python application.

Start the application:

```bash
python app.py
```

The Flask server is configured to listen on port `5000`.

## API Usage

### Endpoint

```text
POST /predict
```

### Example Request

Send a JSON request containing annual income and spending score:

```json
{
  "annual_income": 75,
  "spending_score": 80
}
```

The values are illustrative examples.

### Example Response

```json
{
  "predicted_cluster_id": 1,
  "predicted_cluster_name": "VIP Shoppers",
  "confidence": 0.875
}
```

The response above is illustrative; actual predictions and confidence values depend on the trained model.

## How It Works

1. The Flask API receives annual income and spending score.
2. The input values are arranged into a DataFrame.
3. The saved scaler transforms the input features.
4. The Random Forest model predicts a customer segment.
5. The application maps the predicted cluster ID to a segment name.
6. The API returns the segment ID, name, and confidence score as JSON.

## Limitations

- Predictions depend on the training data and preprocessing used to build the model.
- Model accuracy and evaluation metrics have not been documented here.
- The segment names are application-defined labels and should be validated against actual customer behavior.
- The confidence score represents the model's highest predicted class probability; it is not necessarily a calibrated probability of correctness.

## Future Improvements

- Add model evaluation metrics.
- Include visualizations of customer segments.
- Build a dashboard for exploring customer groups.
- Add input validation and automated API tests.
- Deploy the application to a suitable hosting platform.

## Author

**Tayyaba Maqsood**

BSCS Student | AI/ML Enthusiast | Python & Machine Learning

## Disclaimer

This project is intended for educational and portfolio purposes. Business decisions should consider additional customer information and appropriate validation.
