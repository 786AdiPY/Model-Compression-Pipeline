@echo off
setlocal enabledelayedexpansion

set ROOT=%~dp0
cd /d "%ROOT%"
set PYTHONUTF8=1
set MLFLOW_TRACKING_URI=http://localhost:5000
set DATA_DIR=data
set MODEL_OUT=artifacts/model.pkl
set MODEL_PKL=artifacts/model.pkl
set MODEL_ONNX=artifacts/model_fp32.onnx
set MODEL_TRT=artifacts/model_int8.trt
set TEST_CSV=data/test.csv
set RESULTS_OUT=artifacts/benchmark_results.json
set RESULTS_PATH=artifacts/benchmark_results.json
set GATE_REPORT_OUT=artifacts/gate_report.json
set GATE_REPORT=artifacts/gate_report.json
set META_PATH=artifacts/model_meta.json
set TRAIN_STATS_PATH=artifacts/model_train_stats.json
set DRIFT_REPORT_OUT=artifacts/drift_report.json

echo ============================================================
echo  XGB Compression MLOps Pipeline
echo ============================================================

:: Step 0 — create dirs
mkdir artifacts 2>nul
mkdir mlflow_store\artifacts 2>nul
mkdir data 2>nul

:: Step 1 — Start MLflow in background
echo [1/8] Starting MLflow server...
start "MLflow" cmd /k "cd /d "%ROOT%" && set PYTHONUTF8=1 && mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow_store/mlflow.db --default-artifact-root ./mlflow_store/artifacts"
echo Waiting for MLflow to start...
timeout /t 10 /nobreak >nul

:: Step 2 — Generate data
echo [2/8] Generating synthetic dataset...
python data/generate.py --n 10000 --out data
if errorlevel 1 ( echo FAILED: data generation & goto :error )
echo Done.

:: Step 3 — Train
echo [3/8] Training XGBoost...
python train/train.py
if errorlevel 1 ( echo FAILED: training & goto :error )
echo Done.

:: Step 4 — Compress pkl to ONNX
echo [4/8] Converting pkl to ONNX...
python compress/to_onnx.py
if errorlevel 1 ( echo FAILED: onnx conversion & goto :error )
echo Done.

:: Step 5 — TRT (CPU fallback expected)
echo [5/8] TensorRT conversion (CPU fallback if no GPU)...
python compress/to_trt.py
echo Done.

:: Step 6 — Benchmark
echo [6/8] Benchmarking all model versions...
python benchmark/benchmark.py
if errorlevel 1 ( echo FAILED: benchmark & goto :error )
echo Done.

:: Step 7 — Quality gate
echo [7/8] Running quality gate...
python gate/gate.py
if errorlevel 1 ( echo FAILED: quality gate blocked deployment & goto :error )
echo Done.

:: Step 8 — Register
echo [8/8] Registering best model to MLflow...
python registry/register.py
if errorlevel 1 ( echo FAILED: registry & goto :error )
echo Done.

:: Drift monitor
echo [+] Running drift detection...
set INCOMING_CSV=data/test.csv
python monitor/drift.py
echo Done.

echo.
echo ============================================================
echo  Pipeline complete. Starting API server...
echo  Swagger UI  : http://localhost:8000/docs
echo  MLflow UI   : http://localhost:5000
echo ============================================================
echo.

:: Start serve in new window
start "Serve" cmd /k "cd /d "%ROOT%" && set PYTHONUTF8=1 && set MODEL_PKL=artifacts/model.pkl && set MODEL_ONNX=artifacts/model_fp32.onnx && set MODEL_TRT=artifacts/model_int8.trt && uvicorn serve.app:app --host 0.0.0.0 --port 8000"

echo API server starting in new window...
echo Press any key to exit this window (server keeps running).
pause >nul
exit /b 0

:error
echo.
echo Pipeline stopped due to error. Check output above.
pause
exit /b 1
