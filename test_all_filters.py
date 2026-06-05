import numpy as np
import scipy.ndimage as ndimage
from skimage import color, filters, util, feature, segmentation
from PIL import Image
from scipy import stats
import time
import sys

def process_image(tool):
    img = Image.open('2b0db9f3f1b02dd9ce7b752c36852622.jpg').convert('RGB')
    img_rgb = np.array(img)
    params = {}
    
    start_time = time.time()
    try:
        if tool == 'mode':
            k = 3
            res = np.zeros_like(img_rgb)
            for i in range(3):
                res[..., i] = ndimage.generic_filter(img_rgb[..., i], lambda v: stats.mode(v, keepdims=True)[0], size=k)
            result = res
        elif tool == 'range':
            k = 3
            res = np.zeros_like(img_rgb)
            for i in range(3):
                res[..., i] = ndimage.generic_filter(img_rgb[..., i], lambda v: np.max(v) - np.min(v), size=k)
            result = res
        elif tool == 'dither':
            pil_img = Image.fromarray(img_rgb).convert('L')
            dithered = pil_img.convert('1', dither=Image.FLOYDSTEINBERG)
            dithered_rgb = dithered.convert('RGB')
            result = np.array(dithered_rgb)
        elif tool == 'sobel':
            thr = 0
            gray = color.rgb2gray(img_rgb)
            edges = filters.sobel(gray)
            edges = util.img_as_ubyte(edges / np.max(edges) if np.max(edges) > 0 else edges)
            if thr > 0:
                edges = (edges > thr) * 255
            result = np.stack([edges, edges, edges], axis=-1).astype(np.uint8)
        elif tool == 'sharpen':
            strength = 2.0
            blurred = filters.gaussian(img_rgb, sigma=1.0, channel_axis=-1, preserve_range=True)
            sharpened = img_rgb + strength * (img_rgb - blurred)
            result = np.clip(sharpened, 0, 255).astype(np.uint8)
        elif tool == 'canny':
            sigma = 1.0
            gray = color.rgb2gray(img_rgb)
            edges = feature.canny(gray, sigma=sigma)
            binary = (edges) * 255
            result = np.stack([binary, binary, binary], axis=-1).astype(np.uint8)
        elif tool == 'slic':
            n_seg = 60
            comp = 10.0
            labels = segmentation.slic(img_rgb, n_segments=n_seg, compactness=comp, start_label=1, channel_axis=-1)
            seg_region = color.label2rgb(labels, img_rgb, kind='avg')
            result = util.img_as_ubyte(seg_region)
        elif tool == 'kmeans':
            k = 3
            from scipy.cluster.vq import kmeans2
            pixels = img_rgb.reshape(-1, 3).astype(float)
            centroids, labels = kmeans2(pixels, k, minit='points')
            seg_cluster = centroids[labels]
            seg_cluster = seg_cluster.reshape(img_rgb.shape)
            result = seg_cluster.astype(np.uint8)

        elapsed = time.time() - start_time
        print(f"{tool}: success in {elapsed:.2f}s", flush=True)
    except Exception as e:
        import traceback
        print(f"{tool}: ERROR", flush=True)
        traceback.print_exc()

tools_to_test = ['dither', 'sobel', 'sharpen', 'canny', 'slic', 'kmeans', 'range', 'mode']
for t in tools_to_test:
    process_image(t)
