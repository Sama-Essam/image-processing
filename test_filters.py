import numpy as np
from scipy import stats
import scipy.ndimage as ndimage
from PIL import Image
import warnings

warnings.filterwarnings('ignore')

img_rgb = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)

def test_mode():
    try:
        k = 3
        res = np.zeros_like(img_rgb)
        for i in range(3):
            res[..., i] = ndimage.generic_filter(img_rgb[..., i], lambda v: stats.mode(v, keepdims=True)[0], size=k)
        print("Mode success")
    except Exception as e:
        print(f"Mode error: {e}")

def test_range():
    try:
        k = 3
        res = np.zeros_like(img_rgb)
        for i in range(3):
            res[..., i] = ndimage.generic_filter(img_rgb[..., i], lambda v: np.max(v) - np.min(v), size=k)
        print("Range success")
    except Exception as e:
        print(f"Range error: {e}")

def test_dither():
    try:
        pil_img = Image.fromarray(img_rgb).convert('L')
        dithered = pil_img.convert('1', dither=Image.FLOYDSTEINBERG)
        dithered_rgb = dithered.convert('RGB')
        result = np.array(dithered_rgb)
        print("Dither success")
    except Exception as e:
        print(f"Dither error: {e}")

test_mode()
test_range()
test_dither()
