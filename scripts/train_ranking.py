import json
import pickle
import random
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR))

from app.mlflow_utils import (  # noqa: E402
    log_lightgbm_model,
    log_model_artifacts,
    log_training_metrics,
    log_training_params,
    setup_mlflow,
)


def build_training_matrix(events, users, items, negatives_per_positive=1):
    user_map = {u["user_id"]: np.array(u["features"], dtype=np.float32) for u in users}
    item_map = {i["item_id"]: np.array(i["features"], dtype=np.float32) for i in items}
    features = []
    labels = []

    for event in events:
        user_vec = user_map.get(event["user_id"])
        item_vec = item_map.get(event["item_id"])
        if user_vec is None or item_vec is None:
            continue

        label = 1 if event["event_type"] in {"click", "purchase"} else 0
        features.append(np.hstack([user_vec, item_vec]))
        labels.append(label)

        for _ in range(negatives_per_positive):
            neg_item_id = random.choice(list(item_map.keys()))
            neg_vec = item_map[neg_item_id]
            features.append(np.hstack([user_vec, neg_vec]))
            labels.append(0)

    return np.vstack(features), np.array(labels)


def train(
    n_estimators: int = 200,
    learning_rate: float = 0.05,
    val_fraction: float = 0.2,
    seed: int = 42,
    use_mlflow: bool = True,
    mlflow_experiment: str = "ranking-model",
):
    data_dir = BASE_DIR / "data"
    model_dir = BASE_DIR / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    users = json.loads((data_dir / "sample_users.json").read_text())
    items = json.loads((data_dir / "sample_items.json").read_text())
    events = json.loads((data_dir / "sample_events.json").read_text())

    X, y = build_training_matrix(events, users, items)
    
    # Setup MLflow tracking
    if use_mlflow:
        setup_mlflow(experiment_name=mlflow_experiment)
        
        # Log hyperparameters
        params = {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "val_fraction": val_fraction,
            "seed": seed,
        }
        log_training_params(params)
    
    # Split data for validation
    if val_fraction > 0:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=val_fraction, random_state=seed, stratify=y
        )
    else:
        X_train, X_val, y_train, y_val = X, None, y, None
    
    # Train model
    model = lgb.LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        random_state=seed,
        verbose=-1,
    )
    
    if X_val is not None:
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="binary_logloss",
            callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)],
        )
        
        # Evaluate on validation set
        y_val_pred = model.predict(X_val)
        y_val_pred_proba = model.predict_proba(X_val)[:, 1]
        val_acc = accuracy_score(y_val, y_val_pred)
        val_auc = roc_auc_score(y_val, y_val_pred_proba)
        
        print(f"Validation accuracy: {val_acc:.4f}, AUC: {val_auc:.4f}")
        
        if use_mlflow:
            log_training_metrics(
                {
                    "val_accuracy": val_acc,
                    "val_auc": val_auc,
                }
            )
    else:
        model.fit(X_train, y_train)
    
    # Final evaluation on full training set
    y_train_pred = model.predict(X_train)
    y_train_pred_proba = model.predict_proba(X_train)[:, 1]
    train_acc = accuracy_score(y_train, y_train_pred)
    train_auc = roc_auc_score(y_train, y_train_pred_proba)
    
    print(f"Training accuracy: {train_acc:.4f}, AUC: {train_auc:.4f}")
    
    if use_mlflow:
        log_training_metrics(
            {
                "train_accuracy": train_acc,
                "train_auc": train_auc,
            }
        )

    # Save model
    with open(model_dir / "ranker.pkl", "wb") as f:
        pickle.dump(model, f)
    print(f"Saved ranking model to {model_dir / 'ranker.pkl'}")

    # Log to MLflow
    if use_mlflow:
        import mlflow
        
        # Log model artifacts
        log_model_artifacts(model_dir, artifact_path="ranking_models")
        
        # Log LightGBM model
        log_lightgbm_model(model, artifact_path="ranker")
        
        mlflow.end_run()
        print("MLflow run completed")


if __name__ == "__main__":
    train()
