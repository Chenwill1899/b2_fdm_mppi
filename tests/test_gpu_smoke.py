import pytest

from tests.cuda_test_utils import require_cuda_device


def test_cuda_controller_imports_when_pycuda_available():
    require_cuda_device()

    from b2_fdm_mppi.controllers.mppi_cbf import MPPI_Controller

    assert MPPI_Controller.__name__ == "MPPI_Controller"
