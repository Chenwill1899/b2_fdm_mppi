import pytest


def test_cuda_controller_imports_when_pycuda_available():
    pytest.importorskip("pycuda")

    from b2_fdm_mppi.controllers.mppi_cbf import MPPI_Controller

    assert MPPI_Controller.__name__ == "MPPI_Controller"
