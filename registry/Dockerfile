FROM python:3.11-slim

WORKDIR /app
RUN pip install --no-cache-dir \
    mlflow==2.13.0 \
    boto3==1.34.0

COPY register.py .
ENTRYPOINT ["python", "register.py"]
