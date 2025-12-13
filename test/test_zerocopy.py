import numpy as np

def _ptr(a: np.ndarray) -> int:
    return int(a.__array_interface__["data"][0])

def test_zerocopy_roundtrip_ptr_equal(pf):
    # contiguous uint8 HWC
    arr = np.zeros((16, 17, 3), dtype=np.uint8)
    out = pf._debug_zerocopy_roundtrip_u8(arr)

    assert out.dtype == np.uint8
    assert out.shape == arr.shape

    # 最硬證據：指標一樣 = 沒有 copy
    assert _ptr(out) == _ptr(arr), "roundtrip pointer mismatch => not zero-copy"

def test_zerocopy_roundtrip_base_kept_alive(pf):
    arr = np.zeros((8, 9, 3), dtype=np.uint8)
    out = pf._debug_zerocopy_roundtrip_u8(arr)

    # 有 base 通常代表 out 是 view，底層被某個 owner/capsule 保活
    assert out.base is not None, "out.base is None => likely allocated/copy"

def test_zerocopy_requires_contiguous(pf):
    # 製造 non-contiguous view（例如 slice）
    arr = np.zeros((10, 10, 3), dtype=np.uint8)
    view = arr[:, ::2, :]  # stride != contiguous

    # 如果你的 check_uint8_hw_or_hwc 會拒絕 non-contiguous，這裡應該拋例外
    import pytest
    with pytest.raises(Exception):
        pf._debug_zerocopy_roundtrip_u8(view)
