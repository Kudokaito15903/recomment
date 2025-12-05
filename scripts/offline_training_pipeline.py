"""Offline training pipeline with ETL, model training, evaluation, and MLflow registry."""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd
from loguru import logger

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR))

from app.mlflow_utils import setup_mlflow


class ETLPipeline:
    """ETL pipeline for preparing training data from events."""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir

    def extract_events(self, events_path: Path) -> pd.DataFrame:
        """Extract events from JSON file."""
        logger.info(f"Extracting events from {events_path}")
        with events_path.open() as f:
            events = json.load(f)
        df = pd.DataFrame(events)
        logger.info(f"Extracted {len(df)} events")
        return df

    def transform_events(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform events into training features."""
        logger.info("Transforming events into features")
        
        # Add time-based features
        if "timestamp" in df.columns:
            df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
            df["hour"] = df["datetime"].dt.hour
            df["day_of_week"] = df["datetime"].dt.dayofweek
            df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
        
        # Encode event types
        event_type_map = {
            "view": 0,
            "click": 1,
            "purchase": 2,
            "impression": 3,
        }
        df["event_type_encoded"] = df["event_type"].map(event_type_map).fillna(-1)
        
        # Create labels (positive = click or purchase)
        df["label"] = df["event_type"].isin(["click", "purchase"]).astype(int)
        
        logger.info(f"Transformed {len(df)} events")
        return df

    def load_to_training_format(
        self, df: pd.DataFrame, output_path: Path
    ) -> Dict[str, int]:
        """Load transformed data to training format and save statistics."""
        logger.info(f"Saving transformed data to {output_path}")
        df.to_parquet(output_path, index=False)
        
        stats = {
            "total_events": len(df),
            "positive_labels": int(df["label"].sum()),
            "negative_labels": int((~df["label"].astype(bool)).sum()),
            "unique_users": df["user_id"].nunique(),
            "unique_items": df["item_id"].nunique(),
            "event_types": df["event_type"].value_counts().to_dict(),
        }
        
        logger.info(f"Statistics: {stats}")
        return stats


class ModelEvaluator:
    """Evaluate trained models on test set."""

    def __init__(self):
        pass

    def evaluate_retrieval(
        self, model_path: Path, test_data: pd.DataFrame
    ) -> Dict[str, float]:
        """Evaluate retrieval model."""
        # Placeholder for retrieval evaluation
        # In production, this would compute metrics like recall@k, NDCG, etc.
        logger.info("Evaluating retrieval model")
        return {
            "recall@10": 0.0,
            "recall@50": 0.0,
            "ndcg@10": 0.0,
        }

    def evaluate_ranking(
        self, model_path: Path, test_data: pd.DataFrame
    ) -> Dict[str, float]:
        """Evaluate ranking model."""
        # Placeholder for ranking evaluation
        # In production, this would compute metrics like AUC, precision, recall, etc.
        logger.info("Evaluating ranking model")
        return {
            "auc": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }


def run_pipeline(
    data_dir: str = "data",
    output_dir: str = "data/processed",
    use_mlflow: bool = True,
):
    """Run complete offline training pipeline."""
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup MLflow
    if use_mlflow:
        setup_mlflow(experiment_name="offline-training-pipeline")

    # ETL Phase
    logger.info("=" * 50)
    logger.info("ETL Phase")
    logger.info("=" * 50)
    
    etl = ETLPipeline(data_dir)
    events_path = data_dir / "sample_events.json"
    
    if not events_path.exists():
        logger.error(f"Events file not found: {events_path}")
        return
    
    # Extract
    events_df = etl.extract_events(events_path)
    
    # Transform
    transformed_df = etl.transform_events(events_df)
    
    # Load
    output_path = output_dir / "training_data.parquet"
    stats = etl.load_to_training_format(transformed_df, output_path)
    
    if use_mlflow:
        import mlflow
        
        mlflow.log_params({"pipeline_stage": "etl"})
        mlflow.log_metrics(stats)
        mlflow.log_artifact(str(output_path), artifact_path="training_data")

    # Training Phase
    logger.info("=" * 50)
    logger.info("Training Phase")
    logger.info("=" * 50)
    
    logger.info("Training retrieval model...")
    # Import and run training scripts
    from scripts.train_retrieval import train as train_retrieval
    
    train_retrieval(use_mlflow=use_mlflow)
    
    logger.info("Training ranking model...")
    from scripts.train_ranking import train as train_ranking
    
    train_ranking(use_mlflow=use_mlflow)

    # Evaluation Phase
    logger.info("=" * 50)
    logger.info("Evaluation Phase")
    logger.info("=" * 50)
    
    evaluator = ModelEvaluator()
    
    # Split data for evaluation
    test_df = transformed_df.sample(frac=0.2, random_state=42)
    
    retrieval_metrics = evaluator.evaluate_retrieval(
        Path("models/retrieval_user.pt"), test_df
    )
    ranking_metrics = evaluator.evaluate_ranking(Path("models/ranker.pkl"), test_df)
    
    if use_mlflow:
        import mlflow
        
        mlflow.log_metrics(retrieval_metrics)
        mlflow.log_metrics(ranking_metrics)
        mlflow.log_params({"pipeline_stage": "evaluation"})
        mlflow.end_run()
    
    logger.info("Pipeline completed successfully")
    logger.info(f"Retrieval metrics: {retrieval_metrics}")
    logger.info(f"Ranking metrics: {ranking_metrics}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offline training pipeline")
    parser.add_argument("--data-dir", default="data", help="Input data directory")
    parser.add_argument("--output-dir", default="data/processed", help="Output directory")
    parser.add_argument("--no-mlflow", action="store_true", help="Disable MLflow tracking")
    args = parser.parse_args()

    run_pipeline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        use_mlflow=not args.no_mlflow,
    )

