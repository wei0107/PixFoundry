import os
import glob
import numpy as np
import pixfoundry as pf


def test_filters_week3():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    inp_dir = os.path.join(root, "images_example", "input")
    out_dir = os.path.join(root, "images_example", "output")
    os.makedirs(out_dir, exist_ok=True)

    # 選一張測試圖（跟 basic 測試用同一張，方便對照）
    imgp = sorted(glob.glob(os.path.join(inp_dir, "*.jpg")))[1]
    img = pf.load_image(imgp)   # uint8 HxW or HxWx3
    assert img.dtype == np.uint8

    # ---- 1. 建立雜訊圖（鹽胡椒雜訊） ----
    noisy = img.copy()
    rng = np.random.default_rng(0)

    if img.ndim == 2:
        H, W = img.shape
        num_sp = H * W // 10  # 10% pixel 加鹽胡椒
        ys = rng.integers(0, H, size=num_sp)
        xs = rng.integers(0, W, size=num_sp)
        vals = rng.integers(0, 2, size=num_sp) * 255  # 0 或 255
        noisy[ys, xs] = vals
    else:
        H, W, C = img.shape
        num_sp = H * W // 10
        ys = rng.integers(0, H, size=num_sp)
        xs = rng.integers(0, W, size=num_sp)
        vals = rng.integers(0, 2, size=num_sp) * 255  # 0 或 255
        # 對該 pixel 的所有 channel 加鹽胡椒
        noisy[ys, xs, :] = vals[:, None]

    assert noisy.shape == img.shape
    assert noisy.dtype == np.uint8

    # 儲存雜訊圖
    pf.save_image(os.path.join(out_dir, "week3_noisy.png"), noisy)

    # ---- 2. 套用 Week3 濾波器：Median & Bilateral ----
    # Median：針對鹽胡椒雜訊效果通常很好
    den_med = pf.median_filter(
        noisy, 3,
        backend="single",
        border="reflect",
        border_value=0,
    )
    assert den_med.shape == img.shape
    assert den_med.dtype == np.uint8

    # Bilateral：在平滑的同時嘗試保留邊緣
    den_bil = pf.bilateral_filter(
        noisy,
        ksize=5,
        sigma_color=30.0,
        sigma_space=2.0,
        backend="single",
        border="reflect",
        border_value=0,
    )
    assert den_bil.shape == img.shape
    assert den_bil.dtype == np.uint8

    # ---- 3. 簡單數值檢查（變異度） ----
    def var(a): return float(np.var(a.astype(np.float32)))

    if img.ndim == 2:
        v_clean = var(img)
        v_noisy = var(noisy)
        v_med   = var(den_med)
        v_bil   = var(den_bil)

        # noisy 應該比原圖變異度大（加雜訊）
        assert v_noisy >= v_clean

        # 濾波後應該比 noisy 平滑一些（變異度下降）
        assert v_med <= v_noisy + 1e-3
        assert v_bil <= v_noisy + 1e-3
    else:
        v_clean = [var(img[..., c]) for c in range(img.shape[2])]
        v_noisy = [var(noisy[..., c]) for c in range(img.shape[2])]
        v_med   = [var(den_med[..., c]) for c in range(img.shape[2])]
        v_bil   = [var(den_bil[..., c]) for c in range(img.shape[2])]

        for c in range(img.shape[2]):
            # 每個 channel 的 noisy 應該比 clean 更抖
            assert v_noisy[c] >= v_clean[c]
            # 濾波後的變異度應該不大於 noisy
            assert v_med[c] <= v_noisy[c] + 1e-3
            assert v_bil[c] <= v_noisy[c] + 1e-3

    # ---- 4. 儲存濾波結果，方便肉眼檢查 ----
    pf.save_image(os.path.join(out_dir, "week3_median.png"), den_med)
    pf.save_image(os.path.join(out_dir, "week3_bilateral.png"), den_bil)


if __name__ == "__main__":
    test_filters_week3()
