import pytest

torch = pytest.importorskip("torch")


@pytest.mark.gpu
def test_cuda_is_available():
    if not torch.cuda.is_available():
        pytest.skip("CUDA-capable GPU is not available")

    assert torch.cuda.get_device_name(0)
