import os
import glob
import numpy as np
import pixfoundry as pf

def test_batch_io():
    # 專案根目錄（test/ 的上一層）
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    inp_dir = os.path.join(ROOT, "images_example", "input")
    out_dir = os.path.join(ROOT, "images_example", "output")
    os.makedirs(out_dir, exist_ok=True)

    inputs = sorted(glob.glob(os.path.join(inp_dir, "*.jpg")))
    assert inputs, f"No input JPG files found in {inp_dir}!"

    for p in inputs:
        img = pf.load_image(p)
        assert isinstance(img, np.ndarray)
        assert img.dtype == np.uint8
        assert img.ndim in (2, 3)
        if img.ndim == 3:
            assert img.shape[2] == 3

        base = os.path.splitext(os.path.basename(p))[0]
        out_path = os.path.join(out_dir, base + ".png")
        pf.save_image(out_path, img)

        assert os.path.exists(out_path)
        img2 = pf.load_image(out_path)
        assert img2.shape[:2] == img.shape[:2]

    print(f"Processed {len(inputs)} images successfully. Output -> {out_dir}")

if __name__ == "__main__":
    test_batch_io()
