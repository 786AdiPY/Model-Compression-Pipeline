FROM python:3.11-slim

WORKDIR /app
RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    pandas==2.2.2

COPY drift.py .
ENTRYPOINT ["python", "drift.py"]
