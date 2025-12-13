import numpy as np

BORDERS = ["reflect", "replicate", "wrap", "constant"]


def test_mean_filter_all_borders(pf, test_images, backends, assert_equal):
    rgb, gray = test_images

    for img in [gray, rgb]:
        for border in BORDERS:
            out_s = pf.mean_filter(img, ksize=7, backend="single", border=border, border_value=42)
            assert out_s.shape == img.shape
            assert out_s.dtype == np.uint8

            if "openmp" in backends:
                out_o = pf.mean_filter(img, ksize=7, backend="openmp", border=border, border_value=42)
                assert_equal(out_s, out_o)


def test_gaussian_filter_all_borders(pf, test_images, backends, assert_equal):
    rgb, gray = test_images

    for img in [gray, rgb]:
        for border in BORDERS:
            out_s = pf.gaussian_filter(img, sigma=2.0, backend="single", border=border, border_value=7)
            assert out_s.shape == img.shape
            assert out_s.dtype == np.uint8

            if "openmp" in backends:
                out_o = pf.gaussian_filter(img, sigma=2.0, backend="openmp", border=border, border_value=7)
                assert_equal(out_s, out_o)


def test_median_filter_basic(pf, test_images, backends, assert_equal):
    rgb, gray = test_images
    for img in [gray, rgb]:
        out_s = pf.median_filter(img, ksize=3, backend="single", border="reflect", border_value=0)
        assert out_s.shape == img.shape
        assert out_s.dtype == np.uint8

        if "openmp" in backends:
            out_o = pf.median_filter(img, ksize=3, backend="openmp", border="reflect", border_value=0)
            assert_equal(out_s, out_o)


def test_bilateral_filter_basic(pf, test_images, backends, assert_equal):
    rgb, gray = test_images

    for img in [gray, rgb]:
        out_s = pf.bilateral_filter(
            img, ksize=7, sigma_color=25.0, sigma_space=2.0,
            backend="single", border="reflect", border_value=0
        )
        assert out_s.shape == img.shape
        assert out_s.dtype == np.uint8

        if "openmp" in backends:
            out_o = pf.bilateral_filter(
                img, ksize=7, sigma_color=25.0, sigma_space=2.0,
                backend="openmp", border="reflect", border_value=0
            )
            assert_equal(out_s, out_o)
