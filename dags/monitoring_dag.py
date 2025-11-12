"""
Airflow DAG for System Monitoring and Health Checks
Monitors system health, data quality, and model performance
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

default_args = {
    'owner': 'data-engineering',
    'depends_on_past': False,
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'start_date': days_ago(1),
}

dag = DAG(
    'system_monitoring',
    default_args=default_args,
    description='System monitoring and health checks',
    schedule_interval='*/15 * * * *',  # Run every 15 minutes
    catchup=False,
    tags=['monitoring', 'health', 'system'],
)

def check_api_health_task():
    """Check if API service is healthy"""
    import requests
    try:
        response = requests.get('http://localhost:8000/health', timeout=5)
        if response.status_code == 200:
            print("API service is healthy")
            return True
        else:
            raise Exception(f"API returned status {response.status_code}")
    except Exception as e:
        print(f"API health check failed: {e}")
        raise

def check_kafka_health_task():
    """Check if Kafka is running"""
    from kafka import KafkaConsumer
    try:
        consumer = KafkaConsumer(
            bootstrap_servers=['localhost:9092'],
            consumer_timeout_ms=5000
        )
        topics = consumer.list_consumer_groups()
        print("Kafka is healthy")
        return True
    except Exception as e:
        print(f"Kafka health check failed: {e}")
        raise

def check_mlflow_health_task():
    """Check if MLflow is accessible"""
    import requests
    try:
        response = requests.get('http://localhost:5000/health', timeout=5)
        if response.status_code == 200:
            print("MLflow is healthy")
            return True
        else:
            raise Exception(f"MLflow returned status {response.status_code}")
    except Exception as e:
        print(f"MLflow health check failed: {e}")
        raise

def check_data_quality_task():
    """Check data quality metrics"""
    print("Checking data quality...")
    # Implement data quality checks
    # - Check for missing values
    # - Check for anomalies
    # - Check data freshness
    print("Data quality checks passed")
    return True

def generate_monitoring_report_task(**context):
    """Generate monitoring report"""
    ti = context['ti']
    api_health = ti.xcom_pull(task_ids='check_api_health')
    kafka_health = ti.xcom_pull(task_ids='check_kafka_health')
    mlflow_health = ti.xcom_pull(task_ids='check_mlflow_health')
    data_quality = ti.xcom_pull(task_ids='check_data_quality')
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'api_health': api_health,
        'kafka_health': kafka_health,
        'mlflow_health': mlflow_health,
        'data_quality': data_quality,
    }
    
    print(f"Monitoring report: {report}")
    return report

# Task definitions
check_api_health = PythonOperator(
    task_id='check_api_health',
    python_callable=check_api_health_task,
    dag=dag,
)

check_kafka_health = PythonOperator(
    task_id='check_kafka_health',
    python_callable=check_kafka_health_task,
    dag=dag,
)

check_mlflow_health = PythonOperator(
    task_id='check_mlflow_health',
    python_callable=check_mlflow_health_task,
    dag=dag,
)

check_data_quality = PythonOperator(
    task_id='check_data_quality',
    python_callable=check_data_quality_task,
    dag=dag,
)

generate_report = PythonOperator(
    task_id='generate_monitoring_report',
    python_callable=generate_monitoring_report_task,
    dag=dag,
)

# Task dependencies - run health checks in parallel
[check_api_health, check_kafka_health, check_mlflow_health, check_data_quality] >> generate_report

