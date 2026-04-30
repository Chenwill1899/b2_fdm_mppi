import pytest


def require_cuda_device():
    pycuda = pytest.importorskip("pycuda")
    try:
        import pycuda.driver as cuda

        cuda.init()
        if cuda.Device.count() < 1:
            pytest.skip("PyCUDA is installed but no CUDA device is available")
    except Exception as exc:
        pytest.skip(f"PyCUDA is installed but CUDA is unavailable: {exc}")
    return pycuda
