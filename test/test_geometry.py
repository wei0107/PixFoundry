import os
import glob
import numpy as np
import pixfoundry as pf


def test_geometry_basic():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    inp_dir = os.path.join(root, "images_example", "input")
    out_dir = os.path.join(root, "images_example", "output")
    os.makedirs(out_dir, exist_ok=True)

    imgs = sorted(glob.glob(os.path.join(inp_dir, "*.jpg")))
    assert imgs, "no input jpg in images_example/input"
    img_path = imgs[0]

    img = pf.load_image(img_path)
    assert isinstance(img, np.ndarray)
    h, w = img.shape[:2]

    # resize 測試
    resized = pf.resize(img, height=h // 2, width=w // 2, backend="openmp")
    assert resized.shape[0] == h // 2
    assert resized.shape[1] == w // 2
    pf.save_image(os.path.join(out_dir, "week6_resize.jpg"), resized)

    # flip 測試
    flip_h = pf.flip_horizontal(img, backend="openmp")
    flip_v = pf.flip_vertical(img, backend="openmp")
    assert flip_h.shape == img.shape
    assert flip_v.shape == img.shape
    pf.save_image(os.path.join(out_dir, "week6_flip_h.jpg"), flip_h)
    pf.save_image(os.path.join(out_dir, "week6_flip_v.jpg"), flip_v)

    # crop 測試
    ch = h // 2
    cw = w // 2
    crop_img = pf.crop(img, y=h // 4, x=w // 4, height=ch, width=cw, backend="openmp")
    assert crop_img.shape[0] == ch
    assert crop_img.shape[1] == cw
    pf.save_image(os.path.join(out_dir, "week6_crop.jpg"), crop_img)

    # rotate 測試
    rot = pf.rotate(img, angle_deg=30.0, backend="openmp")
    assert rot.shape == img.shape
    pf.save_image(os.path.join(out_dir, "week6_rotate_30.jpg"), rot)


if __name__ == "__main__":
    test_geometry_basic()
