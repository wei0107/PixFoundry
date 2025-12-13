# test/test_correctness.py
import numpy as np


def _clip_u8(x: float) -> np.uint8:
    return np.uint8(np.clip(np.round(x), 0, 255))


def _ptr(a: np.ndarray) -> int:
    return int(a.__array_interface__["data"][0])


def test_invert_formula_u8(pf, backends):
    # 1x1x3
    src = np.array([[[100, 0, 255]]], dtype=np.uint8)  # (R,G,B)
    expected = np.array([[[155, 255, 0]]], dtype=np.uint8)

    for b in backends:
        out = pf.invert(src, backend=b)
        assert out.dtype == np.uint8
        assert out.shape == src.shape
        assert np.array_equal(out, expected), f"invert formula mismatch on backend={b}"


def test_to_grayscale_weighted_rounding(pf, backends):
    # 1x1 RGB
    # 公式：round(0.299R + 0.587G + 0.114B), clamp to [0,255]
    src = np.array([[[10, 20, 30]]], dtype=np.uint8)
    expected_v = _clip_u8(0.299 * 10 + 0.587 * 20 + 0.114 * 30)
    expected = np.array([[expected_v]], dtype=np.uint8)  # gray -> (H,W)

    for b in backends:
        out = pf.to_grayscale(src, backend=b)
        assert out.dtype == np.uint8
        assert out.shape == expected.shape
        assert np.array_equal(out, expected), f"grayscale mismatch on backend={b}"


def test_adjust_brightness_contrast_formula_clip(pf, backends):
    # 1x2 gray
    src = np.array([[10, 250]], dtype=np.uint8)
    alpha = 1.2
    beta = 20.0

    expected = np.array(
        [[_clip_u8(alpha * 10 + beta), _clip_u8(alpha * 250 + beta)]],
        dtype=np.uint8,
    )

    for b in backends:
        out = pf.adjust_brightness_contrast(src, alpha=alpha, beta=beta, backend=b)
        assert out.dtype == np.uint8
        assert out.shape == src.shape
        assert np.array_equal(out, expected), f"brightness/contrast mismatch on backend={b}"


def test_gamma_correct_formula(pf, backends):
    # 1x3 gray
    src = np.array([[0, 64, 255]], dtype=np.uint8)
    gamma = 2.0

    def ref(v: int) -> np.uint8:
        x = v / 255.0
        return _clip_u8((x ** gamma) * 255.0)

    expected = np.array([[ref(0), ref(64), ref(255)]], dtype=np.uint8)

    for b in backends:
        out = pf.gamma_correct(src, gamma=gamma, backend=b)
        assert out.dtype == np.uint8
        assert out.shape == src.shape
        assert np.array_equal(out, expected), f"gamma mismatch on backend={b}"


def test_flip_horizontal_3x3_indexing(pf, backends):
    # 3x3 gray with unique values
    src = np.array(
        [[1, 2, 3],
         [4, 5, 6],
         [7, 8, 9]],
        dtype=np.uint8,
    )
    expected = np.array(
        [[3, 2, 1],
         [6, 5, 4],
         [9, 8, 7]],
        dtype=np.uint8,
    )

    for b in backends:
        out = pf.flip_horizontal(src, backend=b)
        assert np.array_equal(out, expected), f"flip_horizontal mismatch on backend={b}"


def test_flip_vertical_3x3_indexing(pf, backends):
    src = np.array(
        [[1, 2, 3],
         [4, 5, 6],
         [7, 8, 9]],
        dtype=np.uint8,
    )
    expected = np.array(
        [[7, 8, 9],
         [4, 5, 6],
         [1, 2, 3]],
        dtype=np.uint8,
    )

    for b in backends:
        out = pf.flip_vertical(src, backend=b)
        assert np.array_equal(out, expected), f"flip_vertical mismatch on backend={b}"


def test_crop_corners_match(pf, backends):
    # 5x6 gray with identifiable values: val = y*10 + x
    H, W = 5, 6
    src = np.fromfunction(lambda y, x: y * 10 + x, (H, W), dtype=int).astype(np.uint8)

    # crop region: y=1, x=2, height=3, width=2 -> rows 1..3, cols 2..3
    y, x, height, width = 1, 2, 3, 2
    expected = src[y:y+height, x:x+width]

    # corners in original
    exp_tl = src[y, x]
    exp_tr = src[y, x + width - 1]
    exp_bl = src[y + height - 1, x]
    exp_br = src[y + height - 1, x + width - 1]

    for b in backends:
        out = pf.crop(src, y=y, x=x, height=height, width=width, backend=b)
        assert out.shape == (height, width)
        assert np.array_equal(out, expected), f"crop content mismatch on backend={b}"
        assert out[0, 0] == exp_tl
        assert out[0, -1] == exp_tr
        assert out[-1, 0] == exp_bl
        assert out[-1, -1] == exp_br
