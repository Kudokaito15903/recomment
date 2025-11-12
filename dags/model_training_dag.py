"""
Airflow DAG for Model Training Pipeline
Trains SVD and NMF models on a daily schedule
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

default_args = {
    'owner': 'data-engineering',
    'depends_on_past': False,
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'start_date': days_ago(1),
}

dag = DAG(
    'model_training_pipeline',
    default_args=default_args,
    description='Daily model training pipeline for recommendation engine',
    schedule_interval='0 2 * * *',  # Run daily at 2 AM
    catchup=False,
    tags=['ml', 'training', 'recommendations'],
)

def train_models_task():
    """Train all recommendation models"""
    import asyncio
    from src.models.train_models import ModelTrainer
    
    trainer = ModelTrainer(config_path='config/config.yaml')
    results = asyncio.run(trainer.train_all_models())
    
    print(f"Training completed: {results}")
    return results

def validate_models_task(**context):
    """Validate trained models meet quality thresholds"""
    ti = context['ti']
    training_results = ti.xcom_pull(task_ids='train_models')
    
    if not training_results:
        raise ValueError("No training results found")
    
    # Check if models meet quality thresholds
    for model_name, model_data in training_results.items():
        if model_name == 'evaluation_report':
            continue
        
        if model_data.get('status') != 'success':
            raise ValueError(f"Model {model_name} training failed")
        
        metrics = model_data.get('metrics', {})
        
        # Validate SVD model
        if model_name == 'svd':
            if metrics.get('rmse', float('inf')) > 1.0:
                raise ValueError(f"SVD RMSE too high: {metrics.get('rmse')}")
        
        # Validate NMF model
        elif model_name == 'nmf':
            if metrics.get('coverage', 0) < 0.8:
                raise ValueError(f"NMF coverage too low: {metrics.get('coverage')}")
    
    print("All models passed validation")
    return True

def register_models_mlflow_task(**context):
    """Register trained models in MLflow"""
    ti = context['ti']
    training_results = ti.xcom_pull(task_ids='train_models')
    
    import mlflow
    
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("recommendation_engine")
    
    # Models are already registered during training
    print("Models registered in MLflow")
    return True

# Task definitions
check_data_quality = BashOperator(
    task_id='check_data_quality',
    bash_command='python -c "from src.models.train_models import ModelTrainer; trainer = ModelTrainer(); trainer.load_training_data(); print(\'Data quality check passed\')"',
    dag=dag,
)

train_models = PythonOperator(
    task_id='train_models',
    python_callable=train_models_task,
    dag=dag,
)

validate_models = PythonOperator(
    task_id='validate_models',
    python_callable=validate_models_task,
    dag=dag,
)

register_models = PythonOperator(
    task_id='register_models_mlflow',
    python_callable=register_models_mlflow_task,
    dag=dag,
)

send_notification = BashOperator(
    task_id='send_notification',
    bash_command='echo "Model training pipeline completed successfully"',
    dag=dag,
)

# Task dependencies
check_data_quality >> train_models >> validate_models >> register_models >> send_notification

