FROM python:3.11-slim

WORKDIR /app
RUN pip install --no-cache-dir \
    xgboost==2.0.3 \
    scikit-learn==1.4.2 \
    pandas==2.2.2 \
    numpy==1.26.4 \
    mlflow==2.13.0

COPY train.py .
ENTRYPOINT ["python", "train.py"]
