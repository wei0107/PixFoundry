import numpy as np


def test_resize(pf, test_images, backends, assert_equal):
    rgb, gray = test_images
    for img in [gray, rgb]:
        h, w = img.shape[:2]
        out_s = pf.resize(img, height=h // 2, width=w // 2, backend="single")
        assert out_s.dtype == np.uint8
        assert out_s.shape[:2] == (h // 2, w // 2)

        if "openmp" in backends:
            out_o = pf.resize(img, height=h // 2, width=w // 2, backend="openmp")
            assert_equal(out_s, out_o)


def test_flip(pf, test_images, backends, assert_equal):
    rgb, gray = test_images
    for img in [gray, rgb]:
        h_s = pf.flip_horizontal(img, backend="single")
        v_s = pf.flip_vertical(img, backend="single")
        assert h_s.shape == img.shape
        assert v_s.shape == img.shape

        if "openmp" in backends:
            h_o = pf.flip_horizontal(img, backend="openmp")
            v_o = pf.flip_vertical(img, backend="openmp")
            assert_equal(h_s, h_o)
            assert_equal(v_s, v_o)


def test_crop(pf, test_images, backends, assert_equal):
    rgb, gray = test_images
    for img in [gray, rgb]:
        h, w = img.shape[:2]
        out_s = pf.crop(img, y=h // 4, x=w // 4, height=h // 2, width=w // 2, backend="single")
        assert out_s.dtype == np.uint8
        assert out_s.shape[:2] == (h // 2, w // 2)

        if "openmp" in backends:
            out_o = pf.crop(img, y=h // 4, x=w // 4, height=h // 2, width=w // 2, backend="openmp")
            assert_equal(out_s, out_o)


def test_rotate(pf, test_images, backends, assert_equal):
    rgb, gray = test_images
    for img in [gray, rgb]:
        out_s = pf.rotate(img, angle_deg=30.0, backend="single")
        assert out_s.shape == img.shape
        assert out_s.dtype == np.uint8

        if "openmp" in backends:
            out_o = pf.rotate(img, angle_deg=30.0, backend="openmp")
            assert_equal(out_s, out_o)
