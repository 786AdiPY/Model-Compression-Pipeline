FROM python:3.11-slim

RUN pip install --no-cache-dir \
    mlflow==2.13.0 \
    psycopg2-binary==2.9.9 \
    boto3==1.34.0

EXPOSE 5000

# Shell form so ${VAR} expands from env
CMD mlflow server \
    --host 0.0.0.0 \
    --port 5000 \
    --backend-store-uri ${MLFLOW_BACKEND_STORE_URI:-sqlite:///mlflow/mlflow.db} \
    --default-artifact-root ${MLFLOW_ARTIFACT_ROOT:-/mlflow/artifacts}
