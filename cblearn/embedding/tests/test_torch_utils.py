import pytest

from cblearn.embedding._torch_utils import torch_device, _torch_device_is_available


def test_torch_device_auto():
    """ "auto" selects cuda if and only if cuda is available. """
    torch = pytest.importorskip('torch', reason='torch is not installed')

    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert torch_device("auto") == expected


def test_torch_device_cpu():
    """ An explicit device is returned, not None (see audit B1). """
    pytest.importorskip('torch', reason='torch is not installed')

    assert torch_device("cpu") == "cpu"


def test_torch_device_cuda():
    """ "cuda" is returned if available and rejected otherwise. """
    torch = pytest.importorskip('torch', reason='torch is not installed')

    if torch.cuda.is_available():
        assert torch_device("cuda") == "cuda"
        assert torch_device("cuda:0") == "cuda:0"
    else:
        with pytest.raises(ValueError, match="cuda"):
            torch_device("cuda")


def test_torch_device_unavailable_cuda(monkeypatch):
    """ Asking for cuda without cuda available raises instead of falling back silently. """
    torch = pytest.importorskip('torch', reason='torch is not installed')

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert torch_device("auto") == "cpu"
    with pytest.raises(ValueError, match="cuda"):
        torch_device("cuda")


def test_torch_device_unavailable_cuda_index(monkeypatch):
    """ A cuda index beyond the number of devices raises. """
    torch = pytest.importorskip('torch', reason='torch is not installed')

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    assert torch_device("cuda:0") == "cuda:0"
    with pytest.raises(ValueError, match="cuda:7"):
        torch_device("cuda:7")


def test_torch_device_invalid_name():
    """ An unknown device name raises a ValueError naming the input. """
    pytest.importorskip('torch', reason='torch is not installed')

    with pytest.raises(ValueError, match="not_a_device"):
        torch_device("not_a_device")


def test_torch_device_unsupported_type():
    """ Device types other than cpu and cuda are rejected, although torch parses them. """
    pytest.importorskip('torch', reason='torch is not installed')

    with pytest.raises(ValueError, match="meta"):
        torch_device("meta")


def test_torch_device_is_available():
    """ The helper accepts cpu and rejects device types the estimators do not support. """
    torch = pytest.importorskip('torch', reason='torch is not installed')

    assert _torch_device_is_available(torch.device("cpu")) is True
    assert _torch_device_is_available(torch.device("meta")) is False
    assert _torch_device_is_available(torch.device("cuda")) is torch.cuda.is_available()
