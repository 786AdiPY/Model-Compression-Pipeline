FROM python:3.11-slim

WORKDIR /app
RUN pip install --no-cache-dir \
    fastapi==0.111.0 \
    uvicorn[standard]==0.29.0 \
    xgboost==2.0.3 \
    scikit-learn==1.4.2 \
    onnxruntime==1.18.0 \
    numpy==1.26.4 \
    pydantic==2.7.1

COPY app.py compare.py ./
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
