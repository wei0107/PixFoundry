# PixFoundry Benchmark Report (Single vs OpenMP)

- Generated: `2025-12-14T02:33:21+08:00`
- Python: `3.12.3 (main, Nov  6 2025, 13:44:16) [GCC 13.3.0]`
- Platform: `Linux-6.14.0-37-generic-x86_64-with-glibc2.39`
- OMP_NUM_THREADS: `8`
- Warmup: `2`  Repeat: `20`
- Sizes: `256x256, 512x512, 1024x768`

## Summary (median time per call)

| Group | Case | Single (ms) | OpenMP (ms) | Speedup | Notes |
|---|---|---:|---:|---:|---|
| color | brightness_contrast_1024x768 | 12.638 | 2.743 | 4.61× |  |
| color | brightness_contrast_256x256 | 1.142 | 0.218 | 5.24× |  |
| color | brightness_contrast_512x512 | 4.212 | 0.869 | 4.85× |  |
| color | gamma_correct_1024x768 | 0.612 | 0.195 | 3.14× |  |
| color | gamma_correct_256x256 | 0.052 | 0.027 | 1.91× |  |
| color | gamma_correct_512x512 | 0.201 | 0.074 | 2.74× |  |
| color | invert_gray_1024x768 | 0.033 | 0.050 | 0.65× |  |
| color | invert_gray_256x256 | 0.003 | 0.012 | 0.24× |  |
| color | invert_gray_512x512 | 0.008 | 0.023 | 0.36× |  |
| color | invert_rgb_1024x768 | 0.138 | 0.107 | 1.29× |  |
| color | invert_rgb_256x256 | 0.007 | 0.020 | 0.35× |  |
| color | invert_rgb_512x512 | 0.033 | 0.050 | 0.65× |  |
| color | sepia_1024x768 | 6.691 | 2.213 | 3.02× |  |
| color | sepia_256x256 | 0.600 | 0.167 | 3.61× |  |
| color | sepia_512x512 | 2.232 | 0.672 | 3.32× |  |
| color | to_grayscale_1024x768 | 1.931 | 0.762 | 2.53× |  |
| color | to_grayscale_256x256 | 0.219 | 0.072 | 3.02× |  |
| color | to_grayscale_512x512 | 1.124 | 0.252 | 4.45× |  |
| effects | cartoonize_1024x768 | 57.357 | 23.642 | 2.43× |  |
| effects | cartoonize_256x256 | 4.769 | 2.037 | 2.34× |  |
| effects | cartoonize_512x512 | 18.945 | 7.793 | 2.43× |  |
| effects | emboss_1024x768 | 24.015 | 6.298 | 3.81× |  |
| effects | emboss_256x256 | 1.971 | 0.520 | 3.79× |  |
| effects | emboss_512x512 | 7.975 | 2.275 | 3.51× |  |
| effects | sharpen_1024x768 | 20.422 | 6.127 | 3.33× |  |
| effects | sharpen_256x256 | 1.691 | 0.510 | 3.32× |  |
| effects | sharpen_512x512 | 6.775 | 2.219 | 3.05× |  |
| filters | bilateral_k7_1024x768 | 566.779 | 146.569 | 3.87× |  |
| filters | bilateral_k7_256x256 | 47.189 | 12.300 | 3.84× |  |
| filters | bilateral_k7_512x512 | 187.872 | 48.666 | 3.86× |  |
| filters | gaussian_sigma1.2_1024x768 | 45.088 | 12.547 | 3.59× |  |
| filters | gaussian_sigma1.2_256x256 | 3.802 | 1.136 | 3.35× |  |
| filters | gaussian_sigma1.2_512x512 | 15.003 | 4.169 | 3.60× |  |
| filters | mean_filter_k5_1024x768 | 32.079 | 8.535 | 3.76× |  |
| filters | mean_filter_k5_256x256 | 2.691 | 0.697 | 3.86× |  |
| filters | mean_filter_k5_512x512 | 10.695 | 2.952 | 3.62× |  |
| filters | median_filter_k5_1024x768 | 606.346 | 100.632 | 6.03× |  |
| filters | median_filter_k5_256x256 | 50.544 | 8.617 | 5.87× |  |
| filters | median_filter_k5_512x512 | 202.596 | 33.820 | 5.99× |  |
| geometry | crop_1024x768_to_512x384 | 0.215 | 0.102 | 2.11× |  |
| geometry | crop_256x256_to_128x128 | 0.020 | 0.017 | 1.17× |  |
| geometry | crop_512x512_to_256x256 | 0.073 | 0.041 | 1.77× |  |
| geometry | flip_horizontal_1024x768 | 0.779 | 0.331 | 2.36× |  |
| geometry | flip_horizontal_256x256 | 0.061 | 0.038 | 1.62× |  |
| geometry | flip_horizontal_512x512 | 0.257 | 0.117 | 2.19× |  |
| geometry | flip_vertical_1024x768 | 0.770 | 0.359 | 2.15× |  |
| geometry | flip_vertical_256x256 | 0.066 | 0.040 | 1.66× |  |
| geometry | flip_vertical_512x512 | 0.254 | 0.126 | 2.02× |  |
| geometry | resize_1024x768_to_2048x1536 | 65.395 | 14.401 | 4.54× |  |
| geometry | resize_256x256_to_512x512 | 5.445 | 1.259 | 4.32× |  |
| geometry | resize_512x512_to_1024x1024 | 21.780 | 4.862 | 4.48× |  |
| geometry | rotate_15deg_1024x768 | 14.314 | 3.487 | 4.10× |  |
| geometry | rotate_15deg_256x256 | 1.293 | 0.295 | 4.39× |  |
| geometry | rotate_15deg_512x512 | 4.770 | 1.293 | 3.69× |  |

## Raw data

- `results.csv`
- `meta.json`

