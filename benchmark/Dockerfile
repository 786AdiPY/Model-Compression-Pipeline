FROM python:3.11-slim

WORKDIR /app
RUN pip install --no-cache-dir \
    xgboost==2.0.3 \
    scikit-learn==1.4.2 \
    onnxruntime==1.18.0 \
    numpy==1.26.4 \
    pandas==2.2.2

COPY benchmark.py .
ENTRYPOINT ["python", "benchmark.py"]
