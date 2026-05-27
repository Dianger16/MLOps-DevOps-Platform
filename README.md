# MLOps + DevOps Platform
### Stack: MLflow + scikit-learn + FastAPI + GitHub Actions + EKS

> Project #26 from the DevOps + AI Project Sheet

---

## What This Builds

```
Developer pushes code
        │
        ▼
┌───────────────────────────────────────────────┐
│         GitHub Actions MLOps Pipeline          │
│                                               │
│  Job 1: Train & Evaluate                      │
│    └─ Train sentiment analysis model          │
│    └─ Log metrics to MLflow                   │
│    └─ Quality gate: accuracy > 70%            │
│                                               │
│  Job 2: Test API                              │
│    └─ pytest — 15 API tests                   │
│                                               │
│  Job 3: Build & Push to ECR                   │
│    └─ Docker image → AWS ECR                  │
│                                               │
│  Job 4: Deploy to EKS                         │
│    └─ Rolling deploy (zero downtime)          │
│    └─ Post-deploy smoke test                  │
└───────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────┐
│              AWS EKS                          │
│  sentiment-api (2 replicas + HPA)             │
│  POST /predict  → sentiment analysis          │
│  GET  /health   → health check                │
│  GET  /metrics  → request metrics             │
└───────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────┐
│              MLflow (local)                   │
│  http://localhost:5000                        │
│  - Experiment tracking                        │
│  - Metrics: accuracy, F1, precision, recall   │
│  - Artifacts: model, classification report    │
│  - Model registry                             │
└───────────────────────────────────────────────┘
```

---

## ML Pipeline (3 classes)

```
Text input
    └─ TF-IDF Vectorizer (5000 features, bigrams)
    └─ Logistic Regression (multinomial)
    └─ Prediction: positive / neutral / negative
    └─ Confidence score + probabilities
```

---

## Project Structure

```
mlops-platform/
├── model/
│   └── train.py          → Train + log to MLflow
├── api/
│   └── serve.py          → FastAPI model serving
├── pipeline/
│   └── .github/workflows/
│       └── mlops.yml     → Full CI/CD + MLOps pipeline
├── terraform/
│   └── main.tf           → EKS + ECR
├── k8s/
│   └── deployment.yaml   → Deployment + Service + HPA
├── tests/
│   └── test_api.py       → 15 API tests
├── Dockerfile
├── docker-compose.yml    → MLflow + API locally
└── requirements.txt
```

---

## Quick Start (Local)

### Step 1 — Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

### Step 2 — Start MLflow
```bash
docker compose up -d mlflow
# Open: http://localhost:5000
```

### Step 3 — Train the model
```bash
python model/train.py
# Logs metrics to MLflow automatically
```

### Step 4 — Run tests
```bash
python -m pytest tests/ -v
```

### Step 5 — Start the API
```bash
uvicorn api.serve:app --reload --port 8080
# Open: http://localhost:8080/docs
```

### Step 6 — Make a prediction
```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This product is absolutely amazing!"}'
```

Response:
```json
{
  "text": "This product is absolutely amazing!",
  "sentiment": "positive",
  "confidence": 0.9821,
  "probabilities": {"positive": 0.9821, "neutral": 0.0123, "negative": 0.0056},
  "latency_ms": 2.4
}
```

---

## Deploy to EKS

### Step 1 — Provision infrastructure
```bash
cd terraform
terraform init && terraform apply
```

### Step 2 — Add GitHub Secrets
| Secret | Value |
|---|---|
| `AWS_ACCESS_KEY_ID` | IAM user key |
| `AWS_SECRET_ACCESS_KEY` | IAM user secret |

### Step 3 — Push to trigger pipeline
```bash
git push origin main
```

The pipeline trains → tests → builds → deploys automatically.

### Step 4 — Destroy when done
```bash
cd terraform && terraform destroy
```

---

## MLflow Dashboard

Open http://localhost:5000 to see:
- All training runs with metrics
- Accuracy, F1, precision, recall per run
- Model artifacts (pkl, classification report, confusion matrix)
- Model registry with version history

---

## What This Demonstrates

| Skill | Evidence |
|---|---|
| MLflow | Experiment tracking, metrics logging, model registry |
| scikit-learn | TF-IDF + Logistic Regression pipeline |
| FastAPI | Model serving, batch predictions, metrics endpoint |
| Docker | Multi-stage build, health checks |
| GitHub Actions | 4-job MLOps pipeline with quality gates |
| AWS ECR | Private image registry |
| AWS EKS | Managed Kubernetes for model serving |
| Quality gates | Block deployment if accuracy < 70% |
| HPA | Auto-scale based on CPU |
| Rolling deploy | Zero-downtime model updates |
