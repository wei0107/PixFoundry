import importlib
import numpy as np
import pytest


@pytest.fixture(scope="session")
def pf():
    # 延後 import，避免 conftest 載入時 sys.path / editable 安裝尚未就緒
    return importlib.import_module("pixfoundry")


def _make_test_images():
    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 256, size=(64, 80, 3), dtype=np.uint8)
    gray = rng.integers(0, 256, size=(64, 80), dtype=np.uint8)
    return rgb, gray


@pytest.fixture(scope="session")
def test_images():
    return _make_test_images()


def _has_openmp_backend(pf_mod):
    """
    runtime probe:
    - 若 openmp backend 可用（或你設計為沒 openmp 也會 fallback single 且不 crash）→ True
    - 若你設計為沒 openmp 直接丟 exception → False（測試自動 skip openmp 分支）
    """
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    try:
        _ = pf_mod.invert(img, backend="openmp")
        return True
    except Exception:
        return False


@pytest.fixture(scope="session")
def backends(pf):
    b = ["single"]
    if _has_openmp_backend(pf):
        b.append("openmp")
    return b


@pytest.fixture(scope="session")
def assert_equal():
    def _assert_equal(a, b):
        assert isinstance(a, np.ndarray) and isinstance(b, np.ndarray)
        assert a.dtype == b.dtype == np.uint8
        assert a.shape == b.shape
        assert np.array_equal(a, b)
    return _assert_equal
