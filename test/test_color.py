import os
import glob

import numpy as np
import pixfoundry as pf


def test_color_basic():
    # 專案根目錄
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    inp_dir = os.path.join(root, "images_example", "input")
    out_dir = os.path.join(root, "images_example", "output")
    os.makedirs(out_dir, exist_ok=True)

    # 選一張測試圖（跟 test_filters 類似邏輯）
    img_list = sorted(glob.glob(os.path.join(inp_dir, "*.jpg")))
    assert img_list, "no input jpg found in images_example/input/"
    imgp = img_list[6]

    img = pf.load_image(imgp)
    assert img.dtype == np.uint8

    # 1. to_grayscale
    gray = pf.to_grayscale(img)
    assert gray.dtype == np.uint8
    if img.ndim == 2:
        # already grayscale: shape 應該維持一樣
        assert gray.shape == img.shape
    else:
        # RGB -> 單通道
        assert gray.ndim == 2
        assert gray.shape[0] == img.shape[0]
        assert gray.shape[1] == img.shape[1]
    pf.save_image(os.path.join(out_dir, "color_gray.png"), gray)

    # 2. invert
    inv = pf.invert(img)
    assert inv.dtype == np.uint8
    assert inv.shape == img.shape
    pf.save_image(os.path.join(out_dir, "color_invert.png"), inv)

    # 3. sepia（只支援 RGB）
    if img.ndim == 3 and img.shape[2] == 3:
        sep = pf.sepia(img)
        assert sep.dtype == np.uint8
        assert sep.shape == img.shape
        pf.save_image(os.path.join(out_dir, "color_sepia.png"), sep)

    # 4. brightness / contrast
    #    alpha > 1: 對比增加, beta: 亮度偏移
    bc = pf.adjust_brightness_contrast(img, alpha=1.2, beta=10.0)
    assert bc.dtype == np.uint8
    assert bc.shape == img.shape
    pf.save_image(os.path.join(out_dir, "color_bc.png"), bc)

    # 5. gamma correction
    #    gamma < 1: 提亮; gamma > 1: 壓暗
    gamma_img = pf.gamma_correct(img, gamma=0.8)
    assert gamma_img.dtype == np.uint8
    assert gamma_img.shape == img.shape
    pf.save_image(os.path.join(out_dir, "color_gamma_0_8.png"), gamma_img)


if __name__ == "__main__":
    test_color_basic()
