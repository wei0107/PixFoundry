import numpy as np
import pytest


def test_to_grayscale_all_backends(pf, test_images, backends):
    rgb, gray = test_images

    for b in backends:
        out1 = pf.to_grayscale(rgb, backend=b)
        assert out1.dtype == np.uint8
        assert out1.ndim == 2
        assert out1.shape == rgb.shape[:2]

        out2 = pf.to_grayscale(gray, backend=b)
        assert out2.dtype == np.uint8
        assert out2.shape == gray.shape


def test_invert_single_vs_openmp_equal(pf, test_images, backends, assert_equal):
    rgb, gray = test_images
    if "openmp" not in backends:
        pytest.skip("openmp backend not available")

    a = pf.invert(rgb, backend="single")
    b = pf.invert(rgb, backend="openmp")
    assert_equal(a, b)

    a2 = pf.invert(gray, backend="single")
    b2 = pf.invert(gray, backend="openmp")
    assert_equal(a2, b2)


def test_sepia_rgb_only(pf, test_images, backends, assert_equal):
    rgb, gray = test_images

    # gray 不應該能 sepia（若你實作允許也沒關係，這裡只要改成不 raises 即可）
    for b in backends:
        with pytest.raises(Exception):
            _ = pf.sepia(gray, backend=b)

    if "openmp" in backends:
        a = pf.sepia(rgb, backend="single")
        b = pf.sepia(rgb, backend="openmp")
        assert_equal(a, b)
    else:
        out = pf.sepia(rgb, backend="single")
        assert out.shape == rgb.shape
        assert out.dtype == np.uint8


def test_adjust_brightness_contrast(pf, test_images, backends, assert_equal):
    rgb, _ = test_images

    params = [(1.0, 0.0), (1.2, 10.0), (0.8, -20.0)]
    for alpha, beta in params:
        out_s = pf.adjust_brightness_contrast(rgb, alpha=alpha, beta=beta, backend="single")
        assert out_s.shape == rgb.shape
        assert out_s.dtype == np.uint8

        if "openmp" in backends:
            out_o = pf.adjust_brightness_contrast(rgb, alpha=alpha, beta=beta, backend="openmp")
            assert_equal(out_s, out_o)


def test_gamma_correct(pf, test_images, backends, assert_equal):
    rgb, _ = test_images

    with pytest.raises(Exception):
        _ = pf.gamma_correct(rgb, gamma=0.0, backend="single")

    out_s = pf.gamma_correct(rgb, gamma=0.8, backend="single")
    assert out_s.shape == rgb.shape
    assert out_s.dtype == np.uint8

    if "openmp" in backends:
        out_o = pf.gamma_correct(rgb, gamma=0.8, backend="openmp")
        assert_equal(out_s, out_o)
