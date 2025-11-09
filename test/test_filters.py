import os
import glob
import numpy as np
import pixfoundry as pf

def test_filters_basic():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    inp_dir = os.path.join(root, "images_example", "input")
    imgp = sorted(glob.glob(os.path.join(inp_dir, "*.jpg")))[1]
    img = pf.load_image(imgp)   # uint8 HxW or HxWx3

    # mean
    out1 = pf.mean_filter(img, 15)
    assert out1.dtype == np.uint8
    assert out1.shape == img.shape

    # gaussian
    out2 = pf.gaussian_filter(img, 2.5)
    assert out2.dtype == np.uint8
    assert out2.shape == img.shape

    # 簡單數值檢查：濾波後不應該增加全域變異度（超粗略）
    def var(a): return float(np.var(a.astype(np.float32)))
    if img.ndim == 2:
        assert var(out1) <= var(img) + 1e-3
        assert var(out2) <= var(img) + 1e-3
    else:
        for c in range(img.shape[2]):
            assert var(out1[..., c]) <= var(img[..., c]) + 1e-3
            assert var(out2[..., c]) <= var(img[..., c]) + 1e-3

    pf.save_image(os.path.join(root, "images_example", "output", "test_mean.png"), out1)
    pf.save_image(os.path.join(root, "images_example", "output", "test_gaussian.png"), out2)
    

if __name__ == "__main__":
    test_filters_basic()
