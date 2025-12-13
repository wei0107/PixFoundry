import os
import glob
import numpy as np
import pixfoundry as pf
# 為了計算梯度變異度來衡量邊緣保留，我們需要 scipy 的卷積函式
# 注意：使用前請確保已安裝 `pip install scipy`
from scipy.ndimage import convolve1d 


def var_grad_channel(image_f32, axis):
    """計算指定軸向的梯度變異度。"""
    # 簡化：只使用 [1, 0, -1] 核心計算一階梯度
    grad = convolve1d(image_f32, [1, 0, -1], axis=axis, mode='nearest')
    # 梯度變異度越高，代表邊緣（高頻資訊）保留越多
    return np.var(grad)

def test_filters_week3():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    inp_dir = os.path.join(root, "images_example", "input")
    out_dir = os.path.join(root, "images_example", "output")
    os.makedirs(out_dir, exist_ok=True)

    # 選擇一張測試圖
    imgp = sorted(glob.glob(os.path.join(inp_dir, "*.jpg")))[1]
    img = pf.load_image(imgp)
    
    # 雜訊和濾波參數
    RNG_SEED = 0
    KSIZE = 7
    SIGMA_GAUSS_NOISE = 20.0
    SIGMA_BIL_COLOR = 30.0
    SIGMA_BIL_SPACE = 3.0

    rng = np.random.default_rng(RNG_SEED)

    # ---- 通用檢查函式 ----
    def var(a): return float(np.var(a.astype(np.float32)))
    
    # --- A. 針對中值濾波器：使用鹽胡椒雜訊 (Salt-and-Pepper) ---

    print("\n--- A. Testing Median Filter (Salt-and-Pepper) ---")
    
    # 建立鹽胡椒雜訊圖
    sp_noisy = img.copy()
    num_sp = sp_noisy.size // 10  # 10% pixel
    
    if img.ndim == 2:
        H, W = img.shape
        ys = rng.integers(0, H, size=num_sp)
        xs = rng.integers(0, W, size=num_sp)
        vals = rng.integers(0, 2, size=num_sp) * 255
        sp_noisy[ys, xs] = vals
    else:
        H, W, C = img.shape
        ys = rng.integers(0, H, size=num_sp)
        xs = rng.integers(0, W, size=num_sp)
        vals = rng.integers(0, 2, size=num_sp) * 255
        sp_noisy[ys, xs, :] = vals[:, None]

    # 濾波處理
    den_med = pf.median_filter(sp_noisy, 3, border="reflect", border_value=0)
    
    # 數值檢查
    if img.ndim == 2:
        v_clean = var(img)
        v_noisy = var(sp_noisy)
        v_med   = var(den_med)
    else:
        v_clean = np.mean([var(img[..., c]) for c in range(img.shape[2])])
        v_noisy = np.mean([var(sp_noisy[..., c]) for c in range(img.shape[2])])
        v_med   = np.mean([var(den_med[..., c]) for c in range(img.shape[2])])

    print(f"  Variance (Clean): {v_clean:.2f}")
    print(f"  Variance (SP Noisy): {v_noisy:.2f}")
    print(f"  Variance (Median Filtered): {v_med:.2f}")
    
    # 斷言：中值濾波後，變異度應該顯著下降，且接近原圖（v_noisy * 0.5 是一個粗略的閾值）
    assert v_noisy >= v_clean
    assert v_med <= v_noisy * 0.6  # 優秀的鹽胡椒去除效果
    assert v_med <= v_clean * 1.5  # 濾波後不應比原圖模糊太多

    pf.save_image(os.path.join(out_dir, "week3_median_sp.png"), den_med)
    
    
    # --- B. 針對雙邊濾波器：使用高斯雜訊 (Gaussian Noise) ---

    print("\n--- B. Testing Bilateral Filter (Gaussian Noise) ---")

    # 建立高斯雜訊圖
    noise = rng.normal(loc=0.0, scale=SIGMA_GAUSS_NOISE, size=img.shape).astype(np.float32)
    gauss_noisy = np.clip(img.astype(np.float32) + noise, 0.0, 255.0).astype(np.uint8)
    
    # 濾波處理：
    # 1. 雙邊濾波（測試對象）
    den_bil = pf.bilateral_filter(
        gauss_noisy, KSIZE, SIGMA_BIL_COLOR, SIGMA_BIL_SPACE, 
        border="reflect", border_value=0, backend="openmp"
    )
    
    # 2. 標準高斯濾波（對照組：會模糊邊緣）
    den_gauss_ctrl = pf.gaussian_filter(
        gauss_noisy, SIGMA_BIL_SPACE, # 使用相同的 sigma_space
        border="reflect", border_value=0, backend="openmp"
    )

    pf.save_image(os.path.join(out_dir, "week3_bilateral_gauss.png"), den_bil)
    pf.save_image(os.path.join(out_dir, "week3_gauss_control.png"), den_gauss_ctrl)


    # 邊緣保留驗證：使用梯度變異度
    
    def get_avg_vg(image_u8):
        """計算平均水平/垂直梯度變異度"""
        img_f = image_u8.astype(np.float32)
        if img.ndim == 2:
            return (var_grad_channel(img_f, 0) + var_grad_channel(img_f, 1)) / 2.0
        else: # HxWxC
            vg_h = np.mean([var_grad_channel(img_f[..., c], 0) for c in range(img.shape[2])])
            vg_v = np.mean([var_grad_channel(img_f[..., c], 1) for c in range(img.shape[2])])
            return (vg_h + vg_v) / 2.0


    vg_noisy = get_avg_vg(gauss_noisy)
    vg_bil   = get_avg_vg(den_bil)
    vg_gauss = get_avg_vg(den_gauss_ctrl)

    print(f"  Gradient Variance (Noisy): {vg_noisy:.2f}")
    print(f"  Gradient Variance (Bilateral): {vg_bil:.2f}")
    print(f"  Gradient Variance (Gaussian Control): {vg_gauss:.2f}")
    
    # 斷言 1: 兩種濾波器都平滑了雜訊，所以銳利度會下降
    assert vg_bil < vg_noisy
    assert vg_gauss < vg_noisy

    # 斷言 2: 邊緣保留特性 - 雙邊濾波的銳利度必須顯著高於標準高斯濾波
    # vg_bil 應該比 vg_gauss 大，允許 vg_bil > vg_gauss * 1.05
    # 如果 sigma_color 設得合適，這個差異會很明顯
    assert vg_bil > vg_gauss * 1.05 
    
    print("\nTests passed successfully.")


if __name__ == "__main__":
    test_filters_week3()