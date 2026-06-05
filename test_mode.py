import numpy as np
import scipy.ndimage as ndimage
from scipy import stats

img = np.random.randint(0, 255, (10, 10), dtype=np.uint8)
try:
    res = ndimage.generic_filter(img, lambda v: stats.mode(v, keepdims=True)[0], size=3)
    print("SUCCESS")
except Exception as e:
    print("ERROR:", e)
