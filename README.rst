PixFoundry
==========

A lightweight image processing toolkit with a C++ core and a Python interface.

PixFoundry 提供日常影像處理功能，包括濾鏡、色調與顏色調整、銳化與視覺特效，以及基本幾何轉換。
所有功能以清晰、可擴充的數值方法實作，並提供平行化選項以加速運算。

GitHub Repository: https://github.com/wei0107/pixfoundry

-----------------
Problem Statement
-----------------

日常影像編輯常常需要套用濾鏡、色調調整、銳化或幾何操作。
現有函式庫（如 OpenCV）雖然功能強大，但過於龐大、學習曲線高，
也不適合作為課程中展示數值方法與平行化的工具。

PixFoundry 的目標是提供一個輕量化工具包，支援：

- 常見濾鏡（平均、Gaussian、中值、雙邊）
- 色調與顏色轉換（灰階、Sepia、反相、亮度/對比度、Gamma）
- 銳化與視覺特效（銳化、浮雕、卡通化）
- 幾何轉換（縮放、旋轉、翻轉、裁切）
- 平行化運算（多執行緒、SIMD）

-----------------
Prospective Users
-----------------

- 修課學生：需要簡單、可讀性高的數值與平行化影像處理範例。
- 研究人員或興趣者：想測試濾鏡或影像增強演算法。
- 一般使用者：想做日常照片編輯，不想依賴龐大的外部依賴。

-----------------
System Architecture
-----------------

Workflow::

    Input  → Load image (NumPy in Python, buffer in C++)
    Process → Apply filter or transformation (C++ core, 可選平行化)
    Output → Save or回傳結果

核心模組：

- **Core (C++)**: 卷積、濾波、邊緣檢測、插值
- **Bindings (pybind11)**: Python API，呼叫 C++ 函式
- **Application**: 命令列介面與範例腳本

-----------------
Installation
-----------------

建議使用虛擬環境::

    python3 -m venv .venv
    source .venv/bin/activate
    pip install -e .

-----------------
Basic Usage
-----------------

.. code-block:: python

    import pixfoundry as pf

    # 讀取影像
    img = pf.load_image("input.jpg")

    # 濾鏡與轉換
    blurred = pf.gaussian_filter(img, sigma=1.6, backend="openmp")
    gray = pf.to_grayscale(img)
    sharp = pf.sharpen(img)

    # 幾何轉換
    resized = pf.resize(img, width=640, height=480)

    # 輸出結果
    pf.save_image("output.jpg", resized)

-----------------
Development Schedule
-----------------

- **Week 1 (10/06)**: 專案骨架、C++ core + Python binding、基本影像 I/O
- **Week 2 (10/13)**: 卷積框架；平均與 Gaussian 濾鏡 + 測試
- **Week 3 (10/20)**: 中值與雙邊濾鏡 + 測試
- **Week 4 (10/27)**: 色調與顏色調整（灰階、Sepia、反相、亮度/對比）
- **Week 5 (11/03)**: 銳化、浮雕、卡通化（平滑 + 邊緣偵測）
- **Week 6 (11/10)**: 幾何操作（縮放、旋轉、翻轉、裁切）+ OpenMP
- **Week 7 (11/17)**: 測試框架（pytest + Catch2）、範例照片、文件、CI
- **Week 8 (11/24)**: 功能測試、效能評估（單執行緒 vs OpenMP）、Demo

-----------------
References
-----------------

- OpenCV Documentation: https://docs.opencv.org/
- pybind11 Documentation: https://pybind11.readthedocs.io/
- NumPy Documentation: https://numpy.org/doc/
- stb_image / stb_image_write: https://github.com/nothings/stb
