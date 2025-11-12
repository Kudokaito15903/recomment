#!/bin/bash
# Setup script for Airflow

set -e

echo "Setting up Apache Airflow..."

# Create necessary directories
echo "Creating directories..."
mkdir -p logs
mkdir -p plugins
mkdir -p dags

# Set permissions (for Linux/Mac)
if [[ "$OSTYPE" != "msys" && "$OSTYPE" != "win32" ]]; then
    echo "Setting permissions..."
    chmod -R 755 logs
    chmod -R 755 plugins
    chmod -R 755 dags
fi

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "Error: Docker is not running. Please start Docker and try again."
    exit 1
fi

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null; then
    echo "Error: docker-compose is not installed. Please install docker-compose and try again."
    exit 1
fi

echo "Starting Airflow services..."
docker-compose up -d postgres-airflow

echo "Waiting for PostgreSQL to be ready..."
sleep 10

echo "Initializing Airflow database..."
docker-compose up airflow-init

echo "Starting Airflow webserver and scheduler..."
docker-compose up -d airflow-webserver airflow-scheduler

echo ""
echo "=========================================="
echo "Airflow setup completed!"
echo "=========================================="
echo ""
echo "Airflow UI: http://localhost:8081"
echo "Username: airflow"
echo "Password: airflow"
echo ""
echo "To view logs: docker-compose logs -f airflow-webserver"
echo "To stop: docker-compose stop airflow-webserver airflow-scheduler"
echo ""

