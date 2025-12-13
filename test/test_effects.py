import numpy as np


def test_sharpen(pf, test_images, backends, assert_equal):
    rgb, _ = test_images
    out_s = pf.sharpen(rgb, amount=1.0, backend="single")
    assert out_s.shape == rgb.shape
    assert out_s.dtype == np.uint8

    if "openmp" in backends:
        out_o = pf.sharpen(rgb, amount=1.0, backend="openmp")
        assert_equal(out_s, out_o)


def test_emboss(pf, test_images, backends, assert_equal):
    rgb, _ = test_images
    out_s = pf.emboss(rgb, strength=1.0, backend="single")
    assert out_s.shape == rgb.shape
    assert out_s.dtype == np.uint8

    if "openmp" in backends:
        out_o = pf.emboss(rgb, strength=1.0, backend="openmp")
        assert_equal(out_s, out_o)


def test_cartoonize(pf, test_images, backends, assert_equal):
    rgb, _ = test_images
    out_s = pf.cartoonize(rgb, sigma_space=2.0, edge_threshold=40, backend="single")
    assert out_s.shape == rgb.shape
    assert out_s.dtype == np.uint8

    if "openmp" in backends:
        out_o = pf.cartoonize(rgb, sigma_space=2.0, edge_threshold=40, backend="openmp")
        assert_equal(out_s, out_o)
