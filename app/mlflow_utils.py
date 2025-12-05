"""MLflow utilities for experiment tracking and model registry."""
import os
from pathlib import Path
from typing import Any, Dict, Optional

import mlflow
import mlflow.pytorch
import mlflow.sklearn
from loguru import logger

# Try to import mlflow.lightgbm, fallback to sklearn if not available
try:
    import mlflow.lightgbm
    HAS_LIGHTGBM_SUPPORT = True
except ImportError:
    HAS_LIGHTGBM_SUPPORT = False


def setup_mlflow(
    tracking_uri: Optional[str] = None,
    experiment_name: str = "recommendation-engine",
    run_name: Optional[str] = None,
) -> None:
    """
    Setup MLflow tracking and start a new run.
    
    Args:
        tracking_uri: MLflow tracking server URI. If None, uses local file store.
        experiment_name: Name of the MLflow experiment.
        run_name: Optional name for this run.
    """
    tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    
    # Get or create experiment
    try:
        experiment_id = mlflow.create_experiment(experiment_name)
        logger.info(f"Created new experiment: {experiment_name}")
    except Exception:
        experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
        logger.info(f"Using existing experiment: {experiment_name}")
    
    mlflow.set_experiment(experiment_name)
    
    # Start run
    mlflow.start_run(run_name=run_name)
    run_id = mlflow.active_run().info.run_id
    logger.info(f"Started MLflow run: {run_id}")


def log_model_artifacts(
    model_dir: Path,
    artifact_path: str = "models",
    include_patterns: Optional[list] = None,
) -> None:
    """
    Log model artifacts to MLflow.
    
    Args:
        model_dir: Directory containing model files.
        artifact_path: Path within MLflow run artifacts.
        include_patterns: List of file patterns to include (e.g., ["*.pt", "*.pkl"]).
    """
    if not model_dir.exists():
        logger.warning(f"Model directory {model_dir} does not exist")
        return
    
    include_patterns = include_patterns or ["*.pt", "*.pkl", "*.npy", "*.faiss"]
    
    for pattern in include_patterns:
        for file_path in model_dir.glob(pattern):
            mlflow.log_artifact(str(file_path), artifact_path=artifact_path)
            logger.debug(f"Logged artifact: {file_path}")


def log_pytorch_model(
    model: Any,
    artifact_path: str = "pytorch_model",
    registered_model_name: Optional[str] = None,
) -> None:
    """
    Log PyTorch model to MLflow.
    
    Args:
        model: PyTorch model to log.
        artifact_path: Path within MLflow run artifacts.
        registered_model_name: Optional name for model registry.
    """
    mlflow.pytorch.log_model(
        model,
        artifact_path=artifact_path,
        registered_model_name=registered_model_name,
    )
    logger.info(f"Logged PyTorch model to {artifact_path}")


def log_lightgbm_model(
    model: Any,
    artifact_path: str = "lightgbm_model",
    registered_model_name: Optional[str] = None,
) -> None:
    """
    Log LightGBM model to MLflow.
    
    Args:
        model: LightGBM model to log.
        artifact_path: Path within MLflow run artifacts.
        registered_model_name: Optional name for model registry.
    """
    if HAS_LIGHTGBM_SUPPORT:
        mlflow.lightgbm.log_model(
            model,
            artifact_path=artifact_path,
            registered_model_name=registered_model_name,
        )
    else:
        # Fallback to sklearn flavor
        mlflow.sklearn.log_model(
            model,
            artifact_path=artifact_path,
            registered_model_name=registered_model_name,
        )
    logger.info(f"Logged LightGBM model to {artifact_path}")


def log_training_params(params: Dict[str, Any]) -> None:
    """Log training hyperparameters to MLflow."""
    mlflow.log_params(params)
    logger.debug(f"Logged parameters: {list(params.keys())}")


def log_training_metrics(metrics: Dict[str, float], step: Optional[int] = None) -> None:
    """Log training metrics to MLflow."""
    mlflow.log_metrics(metrics, step=step)
    logger.debug(f"Logged metrics at step {step}: {list(metrics.keys())}")

