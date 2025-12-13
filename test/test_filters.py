import os
import glob
import numpy as np
import pixfoundry as pf


def test_filters_basic():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    inp_dir = os.path.join(root, "images_example", "input")
    out_dir = os.path.join(root, "images_example", "output")
    os.makedirs(out_dir, exist_ok=True)

    # 選一張測試圖
    imgp = sorted(glob.glob(os.path.join(inp_dir, "*.jpg")))[1]
    img = pf.load_image(imgp)   # uint8 HxW or HxWx3
    assert img.dtype == np.uint8

    # 測試不同 border 模式
    borders = ["reflect", "replicate", "wrap", "constant"]
    for border in borders:
        # mean
        out1 = pf.mean_filter(img, 15, backend="auto", border=border, border_value=0)
        assert out1.dtype == np.uint8
        assert out1.shape == img.shape
        out3 = pf.mean_filter(img, 15, backend="openmp", border=border, border_value=0)
        assert out3.dtype == np.uint8
        assert out3.shape == img.shape

        # gaussian
        out2 = pf.gaussian_filter(img, 2.5, backend="single", border=border, border_value=0)
        assert out2.dtype == np.uint8
        assert out2.shape == img.shape
        out4 = pf.gaussian_filter(img, 2.5, backend="single", border=border, border_value=0)
        assert out4.dtype == np.uint8
        assert out4.shape == img.shape

        # 簡單數值檢查：濾波後不應該增加全域變異度（超粗略）
        def var(a): return float(np.var(a.astype(np.float32)))
        if img.ndim == 2:
            assert var(out1) <= var(img) + 1e-3
            assert var(out2) <= var(img) + 1e-3
        else:
            for c in range(img.shape[2]):
                assert var(out1[..., c]) <= var(img[..., c]) + 1e-3
                assert var(out2[..., c]) <= var(img[..., c]) + 1e-3

        # 存檔，看一下視覺效果
        pf.save_image(os.path.join(out_dir, f"test_mean_{border}.png"), out1)
        pf.save_image(os.path.join(out_dir, f"test_gaussian_{border}.png"), out2)
        pf.save_image(os.path.join(out_dir, f"test_mean_omp_{border}.png"), out3)
        pf.save_image(os.path.join(out_dir, f"test_gaussian_omp_{border}.png"), out4)

    # 額外：用小圖精準測試 border="constant" 的行為
    small = np.zeros((1, 1), dtype=np.uint8)
    border_val = 42
    out_const = pf.mean_filter(small, 3,
                               backend="single",
                               border="constant",
                               border_value=border_val)
    assert out_const.shape == (1, 1)
    val = int(out_const[0, 0])

    # 對 1x1 圖、ksize=3、constant=c，
    # 兩次 1D box filter（水平+垂直）得到 2/3 * c
    expected = int(round(8.0 / 9.0 * border_val))
    assert abs(val - expected) <= 1, f"unexpected constant-border result: {val} (expected ~{expected})"


if __name__ == "__main__":
    test_filters_basic()
