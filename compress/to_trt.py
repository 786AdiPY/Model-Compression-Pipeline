"""Convert ONNX FP32 → TensorRT INT8 engine.

Requires NVIDIA GPU + TensorRT 8.x+ runtime.
Falls back gracefully to CPU ONNX when TRT unavailable (CI/dev).
"""
import os
import sys
import json
import numpy as np

ONNX_PATH  = os.getenv("MODEL_ONNX", "artifacts/model_fp32.onnx")
TRT_PATH   = os.getenv("MODEL_TRT",  "artifacts/model_int8.trt")
CALIB_ROWS = int(os.getenv("CALIB_ROWS", "1000"))


def build_trt_engine(onnx_path: str, trt_path: str, calib_rows: int):
    try:
        import tensorrt as trt
    except ImportError:
        print("TensorRT not available — skipping TRT conversion. Install tensorrt to enable.")
        _write_fallback_marker(trt_path, onnx_path)
        return

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    class Int8Calibrator(trt.IInt8EntropyCalibrator2):
        def __init__(self, data: np.ndarray, cache_file: str = ""):
            super().__init__()
            self._data       = data.astype(np.float32)
            self._idx        = 0
            self._batch_size = 128
            self._cache_file = cache_file
            import pycuda.driver as cuda
            import pycuda.autoinit  # noqa
            self._device_mem = cuda.mem_alloc(self._data[0:self._batch_size].nbytes)

        def get_batch_size(self):
            return self._batch_size

        def get_batch(self, names):
            import pycuda.driver as cuda
            if self._idx + self._batch_size > len(self._data):
                return None
            batch = self._data[self._idx: self._idx + self._batch_size]
            cuda.memcpy_htod(self._device_mem, batch)
            self._idx += self._batch_size
            return [int(self._device_mem)]

        def read_calibration_cache(self):
            if os.path.exists(self._cache_file):
                with open(self._cache_file, "rb") as f:
                    return f.read()
            return None

        def write_calibration_cache(self, cache):
            with open(self._cache_file, "wb") as f:
                f.write(cache)

    # Generate calibration data
    rng       = np.random.default_rng(42)
    n_features = 10
    calib_data = rng.uniform(0, 1, (calib_rows, n_features)).astype(np.float32)

    with trt.Builder(TRT_LOGGER) as builder, \
         builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, \
         trt.OnnxParser(network, TRT_LOGGER) as parser:

        config = builder.create_builder_config()
        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator = Int8Calibrator(calib_data, trt_path + ".calib")

        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    print(parser.get_error(i))
                raise RuntimeError("ONNX parse failed")

        profile = builder.create_optimization_profile()
        inp     = network.get_input(0)
        profile.set_shape(inp.name, (1, n_features), (64, n_features), (256, n_features))
        config.add_optimization_profile(profile)

        engine_bytes = builder.build_serialized_network(network, config)
        os.makedirs(os.path.dirname(trt_path), exist_ok=True)
        with open(trt_path, "wb") as f:
            f.write(engine_bytes)

    sz_kb = os.path.getsize(trt_path) / 1024
    print(f"TensorRT INT8 engine saved -> {trt_path}  ({sz_kb:.1f} KB)")


def _write_fallback_marker(trt_path: str, onnx_path: str):
    """Write a JSON marker so serve/ knows to fall back to ONNX for v3."""
    os.makedirs(os.path.dirname(trt_path), exist_ok=True)
    marker = {"fallback": True, "use_onnx": onnx_path}
    with open(trt_path + ".json", "w") as f:
        json.dump(marker, f)
    print(f"Fallback marker written -> {trt_path}.json")


if __name__ == "__main__":
    build_trt_engine(ONNX_PATH, TRT_PATH, CALIB_ROWS)
