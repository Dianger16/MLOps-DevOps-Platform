# tests/test_api.py
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# ── Train a minimal model for tests if not exists ─────────────────────────────
def ensure_model():
    model_path = "model/artifacts/model.pkl"
    os.makedirs("model/artifacts", exist_ok=True)
    if not os.path.exists(model_path):
        import pickle
        from sklearn.pipeline import Pipeline
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression

        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=100)),
            ("clf",   LogisticRegression(max_iter=100, random_state=42)),
        ])
        texts  = ["great product", "terrible product", "okay product"] * 10
        labels = ["positive", "negative", "neutral"] * 10
        pipeline.fit(texts, labels)
        with open(model_path, "wb") as f:
            pickle.dump(pipeline, f)
    return model_path

model_path = ensure_model()

# ── Force load model before importing app ─────────────────────────────────────
import api.serve as serve_module
serve_module.load_model()   # load directly — bypasses startup event

from fastapi.testclient import TestClient
client = TestClient(serve_module.app)


class TestHealth:
    def test_health_returns_200(self):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_contains_status(self):
        resp = client.get("/health")
        assert resp.json()["status"] == "healthy"

    def test_health_model_loaded(self):
        resp = client.get("/health")
        assert resp.json()["model_loaded"] is True


class TestPredict:
    def test_positive_prediction(self):
        resp = client.post("/predict", json={"text": "This is absolutely amazing!"})
        assert resp.status_code == 200
        data = resp.json()
        assert "sentiment" in data
        assert data["sentiment"] in ["positive", "negative", "neutral"]
        assert 0 <= data["confidence"] <= 1

    def test_negative_prediction(self):
        resp = client.post("/predict", json={"text": "This is terrible and awful"})
        assert resp.status_code == 200
        assert resp.json()["sentiment"] in ["positive", "negative", "neutral"]

    def test_response_has_probabilities(self):
        resp = client.post("/predict", json={"text": "Average product"})
        data = resp.json()
        assert "probabilities" in data
        assert len(data["probabilities"]) == 3

    def test_response_has_latency(self):
        resp = client.post("/predict", json={"text": "Test text"})
        assert "latency_ms" in resp.json()
        assert resp.json()["latency_ms"] >= 0

    def test_empty_text_rejected(self):
        resp = client.post("/predict", json={"text": ""})
        assert resp.status_code == 422

    def test_response_has_timestamp(self):
        resp = client.post("/predict", json={"text": "Test"})
        assert "timestamp" in resp.json()


class TestBatchPredict:
    def test_batch_prediction(self):
        resp = client.post("/predict/batch", json={
            "texts": ["Great product", "Terrible product", "Average product"]
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 3
        assert len(data["predictions"]) == 3

    def test_batch_empty_rejected(self):
        resp = client.post("/predict/batch", json={"texts": []})
        assert resp.status_code == 422


class TestMetrics:
    def test_metrics_endpoint(self):
        resp = client.get("/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_requests" in data
        assert "avg_latency_ms" in data

    def test_model_info(self):
        resp = client.get("/model-info")
        assert resp.status_code == 200
        data = resp.json()
        assert "pipeline_steps" in data
        assert "classes" in data
        assert len(data["classes"]) == 3