import importlib

import pytest

torch = pytest.importorskip("torch")
get_device = importlib.import_module("robot_ai.common.device").get_device


@pytest.mark.ml
def test_get_device_returns_torch_device():
    device = get_device()

    assert isinstance(device, torch.device)


@pytest.mark.ml
def test_get_device_is_cpu_or_cuda():
    device = get_device()

    assert device.type in {"cpu", "cuda"}
