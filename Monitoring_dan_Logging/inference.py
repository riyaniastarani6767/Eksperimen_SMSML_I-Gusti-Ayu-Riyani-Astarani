import time
import joblib
import pandas as pd
from prometheus_client import start_http_server, Counter, Histogram

model = joblib.load("model.pkl")

prediction_requests = Counter(
    "prediction_requests_total",
    "Total number of prediction requests"
)

prediction_errors = Counter(
    "prediction_errors_total",
    "Total number of prediction errors"
)

prediction_latency = Histogram(
    "prediction_latency_seconds",
    "Prediction latency in seconds"
)

FEATURE_NAMES = [
    'age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous',
    'job_blue-collar', 'job_entrepreneur', 'job_housemaid', 'job_management',
    'job_retired', 'job_self-employed', 'job_services', 'job_student',
    'job_technician', 'job_unemployed', 'job_unknown', 'marital_married',
    'marital_single', 'education_secondary', 'education_tertiary',
    'education_unknown', 'default_yes', 'housing_yes', 'loan_yes',
    'contact_telephone', 'contact_unknown', 'month_aug', 'month_dec',
    'month_feb', 'month_jan', 'month_jul', 'month_jun', 'month_mar',
    'month_may', 'month_nov', 'month_oct', 'month_sep', 'poutcome_other',
    'poutcome_success', 'poutcome_unknown'
]

def predict(data):
    prediction_requests.inc()
    start_time = time.time()
    try:
        df = pd.DataFrame([data], columns=FEATURE_NAMES)
        result = model.predict(df)
    except Exception as e:
        prediction_errors.inc()
        raise
    latency = time.time() - start_time
    prediction_latency.observe(latency)
    return result

if __name__ == "__main__":
    print("Starting Prometheus exporter on port 8000...")
    start_http_server(8000)
    while True:
        try:
            sample_data = [
                35, 1500, 15, 200, 2, -1, 0,
                0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                1, 0,
                1, 0, 0,
                0, 1, 0,
                0, 0,
                0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                0, 0, 0
            ]
            prediction = predict(sample_data)
            print("Prediction:", prediction)
        except Exception as e:
            print("Prediction error:", e)
        time.sleep(5)