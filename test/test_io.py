import numpy as np


def test_io_roundtrip_png(pf, tmp_path):
    img = (np.arange(64 * 80 * 3) % 256).astype(np.uint8).reshape(64, 80, 3)
    p = tmp_path / "x.png"

    pf.save_image(str(p), img)
    img2 = pf.load_image(str(p))

    assert isinstance(img2, np.ndarray)
    assert img2.dtype == np.uint8
    assert img2.shape[:2] == img.shape[:2]
    # 讀回來如果是彩色，應該是 3 channel
    if img2.ndim == 3:
        assert img2.shape[2] == 3
