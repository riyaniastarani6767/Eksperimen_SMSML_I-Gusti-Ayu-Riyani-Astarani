from prometheus_client import start_http_server, Counter, Histogram
import time
import random

prediction_requests = Counter(
    'prediction_requests_total',
    'Total number of prediction requests'
)

prediction_errors = Counter(
    'prediction_errors_total',
    'Total number of prediction errors'
)

prediction_latency = Histogram(
    'prediction_latency_seconds',
    'Prediction latency in seconds'
)

def simulate_prediction():
    start = time.time()
    prediction_requests.inc()
    time.sleep(random.uniform(0.1, 0.5))
    if random.random() < 0.1:
        prediction_errors.inc()
    prediction_latency.observe(time.time() - start)

if __name__ == "__main__":
    print("Starting Prometheus exporter at http://localhost:8000")
    start_http_server(8000)
    while True:
        simulate_prediction()