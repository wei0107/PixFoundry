import os
import glob
import numpy as np
import pixfoundry as pf


def test_effects_basic():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    inp_dir = os.path.join(root, "images_example", "input")
    out_dir = os.path.join(root, "images_example", "output")
    os.makedirs(out_dir, exist_ok=True)

    imgs = sorted(glob.glob(os.path.join(inp_dir, "*.jpg")))
    assert imgs, "no input jpg in images_example/input"
    imgp = imgs[5]

    img = pf.load_image(imgp)
    assert img.dtype == np.uint8

    # Sharpen
    sharp = pf.sharpen(img, amount=1.0, backend="openmp")
    assert sharp.shape == img.shape
    pf.save_image(os.path.join(out_dir, "week5_sharpen.png"), sharp)

    # Emboss
    emb = pf.emboss(img, strength=1.0, backend="openmp")
    assert emb.shape == img.shape
    pf.save_image(os.path.join(out_dir, "week5_emboss.png"), emb)

    # Cartoonize
    cart = pf.cartoonize(img, sigma_space=2.0, edge_threshold=40, backend="openmp")
    assert cart.shape == img.shape
    pf.save_image(os.path.join(out_dir, "week5_cartoon.png"), cart)


if __name__ == "__main__":
    test_effects_basic()
