# model/train.py
# Trains a sentiment analysis model and logs everything to MLflow
# Run: python model/train.py

import os
import json
import pickle
import logging
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── Training data ─────────────────────────────────────────────────────────────
# In production this would come from S3/database
# For the demo we generate a realistic dataset
TRAINING_DATA = [
    # Positive samples
    ("This product is absolutely amazing, I love it!", "positive"),
    ("Exceeded all my expectations, highly recommend", "positive"),
    ("Best purchase I've made this year", "positive"),
    ("Outstanding quality and fast delivery", "positive"),
    ("Five stars, would buy again without hesitation", "positive"),
    ("Fantastic experience from start to finish", "positive"),
    ("Works perfectly, exactly as described", "positive"),
    ("Very happy with this purchase, great value", "positive"),
    ("Incredible product, changed my life", "positive"),
    ("Superb quality, will definitely recommend", "positive"),
    ("Great customer service and amazing product", "positive"),
    ("So pleased with this, exceeded expectations", "positive"),
    ("Absolutely love it, perfect in every way", "positive"),
    ("Wonderful experience, will buy again", "positive"),
    ("Top quality, fast shipping, very satisfied", "positive"),
    ("This is exactly what I needed, perfect fit", "positive"),
    ("Brilliant product, very well made", "positive"),
    ("Delighted with my purchase, great quality", "positive"),
    ("Impressive build quality, worth every penny", "positive"),
    ("Love this product, makes life so much easier", "positive"),

    # Negative samples
    ("Terrible product, complete waste of money", "negative"),
    ("Broke after two days, very disappointed", "negative"),
    ("Nothing like the description, awful quality", "negative"),
    ("Do not buy this, absolute garbage", "negative"),
    ("Worst purchase ever, total scam", "negative"),
    ("Very poor quality, fell apart immediately", "negative"),
    ("Misleading description, very unhappy", "negative"),
    ("Stopped working after one week", "negative"),
    ("Cheap materials, not worth the price", "negative"),
    ("Extremely disappointed, would not recommend", "negative"),
    ("Complete waste of time and money", "negative"),
    ("Defective product, terrible customer service", "negative"),
    ("Horrible experience, avoid at all costs", "negative"),
    ("Poor quality control, arrived damaged", "negative"),
    ("Does not work as advertised, very frustrated", "negative"),
    ("Returned immediately, absolute rubbish", "negative"),
    ("Overpriced junk, fell apart on day one", "negative"),
    ("Not as described, very misleading", "negative"),
    ("Cheap and nasty, avoid this product", "negative"),
    ("Total disappointment, nothing like advertised", "negative"),

    # Neutral samples
    ("It arrived on time and works as expected", "neutral"),
    ("Average product, nothing special", "neutral"),
    ("Does the job, nothing more nothing less", "neutral"),
    ("Okay for the price, not amazing", "neutral"),
    ("Standard quality, meets basic requirements", "neutral"),
    ("It is what it is, functions correctly", "neutral"),
    ("Decent product, but nothing extraordinary", "neutral"),
    ("Works fine, packaging could be better", "neutral"),
    ("Acceptable quality for the price point", "neutral"),
    ("Mediocre, but does what it says", "neutral"),
    ("Neither great nor terrible, just average", "neutral"),
    ("Fair quality, took a while to arrive", "neutral"),
    ("OK product, not bad but not brilliant", "neutral"),
    ("Gets the job done, nothing to complain about", "neutral"),
    ("Reasonable quality, acceptable delivery time", "neutral"),
    ("Average experience overall, nothing special", "neutral"),
    ("Works as advertised, basic but functional", "neutral"),
    ("Satisfactory product, met basic expectations", "neutral"),
    ("It does the job, nothing impressive", "neutral"),
    ("Moderate quality, typical for this price range", "neutral"),
]


def prepare_data():
    """Prepare training data"""
    df = pd.DataFrame(TRAINING_DATA, columns=["text", "label"])
    logger.info(f"Dataset: {len(df)} samples")
    logger.info(f"Label distribution:\n{df['label'].value_counts()}")
    return df


def build_pipeline(max_features: int = 5000, C: float = 1.0) -> Pipeline:
    """Build sklearn ML pipeline: TF-IDF → Logistic Regression"""
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),        # unigrams + bigrams
            stop_words="english",
            lowercase=True,
            strip_accents="unicode",
        )),
        ("clf", LogisticRegression(
            C=C,
            max_iter=1000,
            multi_class="multinomial",
            solver="lbfgs",
            random_state=42,
        )),
    ])


def train():
    """Train model with MLflow tracking"""

    # ── MLflow setup ──────────────────────────────────────────────────────────
    # Respects MLFLOW_TRACKING_URI and MLFLOW_EXPERIMENT_NAME env vars
    # so CI (GitHub Actions) and local dev both work without code changes
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "sentiment-analysis"))

    df = prepare_data()
    X = df["text"].values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── Hyperparameters ───────────────────────────────────────────────────────
    params = {
        "max_features":  5000,
        "C":             1.0,
        "ngram_range":   "(1, 2)",
        "test_size":     0.2,
        "train_samples": len(X_train),
        "test_samples":  len(X_test),
    }

    with mlflow.start_run(run_name="sentiment-classifier-v1") as run:
        logger.info(f"MLflow run ID: {run.info.run_id}")

        # ── Log parameters ────────────────────────────────────────────────────
        mlflow.log_params(params)

        # ── Train ─────────────────────────────────────────────────────────────
        logger.info("Training model...")
        pipeline = build_pipeline(
            max_features=params["max_features"],
            C=params["C"],
        )
        pipeline.fit(X_train, y_train)

        # ── Evaluate ──────────────────────────────────────────────────────────
        y_pred = pipeline.predict(X_test)
        cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring="f1_macro")

        metrics = {
            "accuracy":        float(accuracy_score(y_test, y_pred)),
            "precision_macro": float(precision_score(y_test, y_pred, average="macro")),
            "recall_macro":    float(recall_score(y_test, y_pred, average="macro")),
            "f1_macro":        float(f1_score(y_test, y_pred, average="macro")),
            "cv_f1_mean":      float(cv_scores.mean()),
            "cv_f1_std":       float(cv_scores.std()),
        }

        logger.info("Metrics:")
        for k, v in metrics.items():
            logger.info(f"  {k}: {v:.4f}")

        # ── Log metrics ───────────────────────────────────────────────────────
        mlflow.log_metrics(metrics)

        # ── Save classification report ────────────────────────────────────────
        report = classification_report(y_test, y_pred)
        os.makedirs(os.path.join("model", "artifacts"), exist_ok=True)
        report_path = os.path.join("model", "artifacts", "classification_report.txt")
        with open(report_path, "w") as f:
            f.write(report)
        mlflow.log_artifact(report_path)

        # ── Save confusion matrix ─────────────────────────────────────────────
        cm = confusion_matrix(y_test, y_pred, labels=["positive", "neutral", "negative"])
        cm_data = {
            "labels": ["positive", "neutral", "negative"],
            "matrix": cm.tolist(),
        }
        cm_path = os.path.join("model", "artifacts", "confusion_matrix.json")
        with open(cm_path, "w") as f:
            json.dump(cm_data, f)
        mlflow.log_artifact(cm_path)

        # ── Save model locally ────────────────────────────────────────────────
        model_path = os.path.join("model", "artifacts", "model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(pipeline, f)
        mlflow.log_artifact(model_path)

        logger.info(f"\nModel saved to: {model_path}")
        logger.info(f"MLflow tracking URI: {tracking_uri}")
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"F1 Score: {metrics['f1_macro']:.4f}")

        return run.info.run_id, metrics


if __name__ == "__main__":
    run_id, metrics = train()
    print(f"\nTraining complete!")
    print(f"Run ID: {run_id}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1 Score: {metrics['f1_macro']:.4f}")