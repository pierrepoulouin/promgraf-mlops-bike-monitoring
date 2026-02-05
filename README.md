# Bike Sharing – MLOps Monitoring with Prometheus & Grafana

Prometheus & Grafana exam – Datascientest  
End-to-end monitoring of a regression model using the Bike Sharing UCI dataset.

---

## Project Overview

This project implements a complete MLOps monitoring stack for a regression model predicting the number of shared bikes (cnt).

The solution includes:
- Model inference through a FastAPI service
- Model performance monitoring
- Data drift detection with Evidently
- Infrastructure monitoring with node-exporter
- Automated Grafana dashboards (Dashboards as Code)
- Alerting with Prometheus and Grafana
- Fully reproducible deployment using Docker and Makefile

---

## Architecture

The application is deployed using Docker Compose and includes the following services:
- FastAPI API (model inference and evaluation)
- Prometheus (metrics collection and alerting)
- Grafana (visualization and ML alerts)
- Node Exporter (host infrastructure monitoring)

---

## Repository Structure

    .
    ├── deployment
    │   ├── grafana
    │   │   ├── dashboards
    │   │   │   ├── api_performance.json
    │   │   │   ├── model_performance_drift.json
    │   │   │   └── infrastructure_overview.json
    │   │   └── provisioning
    │   │       ├── datasources
    │   │       │   └── datasources.yaml
    │   │       └── dashboards
    │   │           └── dashboards.yaml
    │   └── prometheus
    │       ├── prometheus.yml
    │       └── rules
    │           └── alert_rules.yml
    │
    ├── src
    │   ├── api
    │   │   ├── Dockerfile
    │   │   ├── main.py
    │   │   └── requirements.txt
    │   └── evaluation
    │       ├── Dockerfile
    │       ├── requirements.txt
    │       └── run_evaluation.py
    │
    ├── docker-compose.yml
    ├── Makefile
    └── README.md

---

## Model and API

- Model: RandomForestRegressor
- Reference training data: January 2011
- Training strategy: model trained once at API startup
- Main endpoints:
  - POST /predict: model inference
  - POST /evaluate: model evaluation and drift detection
  - GET /metrics: Prometheus metrics
  - GET /health: health check

---

## Prometheus Metrics

### API Metrics
- api_requests_total
- api_request_duration_seconds

### Model Metrics
- model_rmse_score
- model_mae_score
- model_r2_score
- model_mape_score

MAPE was chosen as an additional metric because it provides a relative error that is easier to interpret from a business perspective.

### Data Drift
- data_drift_detected (0 or 1)
- Computed using Evidently DataDriftPreset

---

## Grafana Dashboards

All dashboards are provisioned automatically at Grafana startup.

### API Performance Dashboard
- Request rate
- Latency (P95)
- Error rate

### Model Performance and Drift Dashboard
- RMSE, MAE, R2, MAPE
- Data drift detected status

### Infrastructure Overview Dashboard
- CPU usage
- Memory usage
- Disk usage

---

## Alerting

### Prometheus Alert
- BikeAPI_Down
- Triggered when the API is unreachable for more than one minute

### Grafana Alert
- Triggered when the model RMSE exceeds a defined threshold
- Allows early detection of model performance degradation

---

## How to Run the Project

Start all services:

    make all

Stop all services:

    make stop

Run model evaluation:

    make evaluation

Trigger alerts intentionally:

    make fire-alert

---

## Traffic Simulation

Example request to generate traffic on the prediction endpoint:

    curl -X POST http://localhost:8081/predict \
      -H "Content-Type: application/json" \
      -d '{ "temp": 0.3, "atemp": 0.31, "hum": 0.8, "windspeed": 0.1,
            "mnth": 2, "hr": 14, "weekday": 3,
            "season": 1, "holiday": 0, "workingday": 1, "weathersit": 1,
            "dteday": "2011-02-10" }'

---

## Key MLOps Concepts Demonstrated

- Machine learning observability
- Model performance monitoring
- Data drift detection
- Dashboards as Code
- Infrastructure monitoring
- Automated alerting
- Reproducible deployment

---

## Conclusion

This project demonstrates a complete MLOps monitoring pipeline, combining model metrics, data drift detection, infrastructure observability, and alerting using Prometheus and Grafana.
