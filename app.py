import os
import io
import base64
import numpy as np
from flask import Flask, request, jsonify, render_template
from PIL import Image
from skimage import color, exposure, filters, util, morphology, segmentation, feature
from skimage.color import rgb2gray, rgba2rgb
from skimage.filters import threshold_otsu
import scipy.ndimage as ndimage
import warnings
from scipy import stats

warnings.filterwarnings('ignore')

app = Flask(__name__)

def process_image(img_array, tool, params):
    """
    Process the image array based on the requested tool and parameters.
    img_array is expected to be an RGB uint8 array or RGBA.
    """
    # Drop alpha channel if present for most processing
    if img_array.shape[-1] == 4:
        # Keep alpha for later or drop it
        img_rgb = img_array[..., :3]
    else:
        img_rgb = img_array

    result = img_rgb.copy()
    
    try:
        if tool == 'grayscale':
            method = params.get('gm', 'lum')
            if method == 'avg':
                gray = np.mean(img_rgb, axis=2).astype(np.uint8)
            elif method == 'r':
                gray = img_rgb[..., 0]
            elif method == 'g':
                gray = img_rgb[..., 1]
            elif method == 'b':
                gray = img_rgb[..., 2]
            else: # lum
                gray = util.img_as_ubyte(color.rgb2gray(img_rgb))
            # Convert back to 3 channel so it displays easily as rgb
            result = np.stack([gray, gray, gray], axis=-1)

        elif tool == 'brightness':
            v = int(params.get('bv', 80))
            # avoid overflow/underflow by casting to larger int
            res = img_rgb.astype(np.int16) + v
            result = np.clip(res, 0, 255).astype(np.uint8)

        elif tool == 'complement':
            mode = params.get('cm', 'full')
            t = int(params.get('ct', 128))
            gray = util.img_as_ubyte(color.rgb2gray(img_rgb))
            
            mask = np.ones(gray.shape, dtype=bool)
            if mode == 'dark':
                mask = gray < t
            elif mode == 'bright':
                mask = gray > t
                
            inverted = util.invert(img_rgb)
            result = np.where(mask[..., None], inverted, img_rgb)

        elif tool == 'channels':
            channel = int(params.get('channel', 0))
            result = np.zeros_like(img_rgb)
            result[..., channel] = img_rgb[..., channel]

        elif tool == 'stretch':
            clip = float(params.get('cv', 2))
            p_lower, p_upper = np.percentile(img_rgb, (clip, 100 - clip))
            result = exposure.rescale_intensity(img_rgb, in_range=(p_lower, p_upper), out_range=np.uint8)

        elif tool == 'equalize':
            # Equalize each channel separately or on HSV? Usually HSV V channel is better, but let's do simple RGB eq or adaptive
            img_hsv = color.rgb2hsv(img_rgb)
            img_hsv[..., 2] = exposure.equalize_hist(img_hsv[..., 2])
            result = util.img_as_ubyte(color.hsv2rgb(img_hsv))

        elif tool == 'add':
            w = float(params.get('aw', 60)) / 100.0
            mode = params.get('am', 'weighted')
            # For this to work perfectly, we need image 2. 
            # If we don't have it in the request, just return original
            pass # Handled specially or ignored if no second image

        elif tool == 'addnoise':
            ratio = float(params.get('nr', 10)) / 100.0
            noise_type = params.get('nt', 'both')
            if noise_type == 'salt':
                result = util.img_as_ubyte(util.random_noise(img_rgb, mode='salt', amount=ratio))
            elif noise_type == 'pepper':
                result = util.img_as_ubyte(util.random_noise(img_rgb, mode='pepper', amount=ratio))
            else:
                result = util.img_as_ubyte(util.random_noise(img_rgb, mode='s&p', amount=ratio))

        elif tool == 'mean':
            k = int(params.get('mfk', 5))
            # We can apply uniform filter per channel
            res = np.zeros_like(img_rgb)
            for i in range(3):
                res[..., i] = ndimage.uniform_filter(img_rgb[..., i], size=k)
            result = res

        elif tool == 'median':
            k = int(params.get('mdk', 3))
            if k % 2 == 0: k += 1
            # skimage median filter works on 2D, we do per channel
            res = np.zeros_like(img_rgb)
            footprint = morphology.square(k)
            for i in range(3):
                res[..., i] = filters.median(img_rgb[..., i], footprint)
            result = res

        elif tool == 'gaussian':
            s = float(params.get('gbs', 1.5))
            # gaussian blur
            blurred = filters.gaussian(img_rgb, sigma=s, channel_axis=-1, preserve_range=True)
            result = blurred.astype(np.uint8)

        elif tool == 'minmax':
            tt = params.get('mmt', 'min')
            k = int(params.get('mmk', 3))
            footprint = morphology.square(k)
            res = np.zeros_like(img_rgb)
            for i in range(3):
                if tt == 'min':
                    res[..., i] = filters.rank.minimum(img_rgb[..., i], footprint)
                else:
                    res[..., i] = filters.rank.maximum(img_rgb[..., i], footprint)
            result = res

        elif tool == 'sobel':
            thr = int(params.get('edthr', 0))
            gray = color.rgb2gray(img_rgb)
            edges = filters.sobel(gray)
            edges = util.img_as_ubyte(edges / np.max(edges) if np.max(edges) > 0 else edges)
            if thr > 0:
                edges = (edges > thr) * 255
            result = np.stack([edges, edges, edges], axis=-1).astype(np.uint8)
            
        elif tool == 'prewitt':
            thr = int(params.get('edthr', 0))
            gray = color.rgb2gray(img_rgb)
            edges = filters.prewitt(gray)
            edges = util.img_as_ubyte(edges / np.max(edges) if np.max(edges) > 0 else edges)
            if thr > 0:
                edges = (edges > thr) * 255
            result = np.stack([edges, edges, edges], axis=-1).astype(np.uint8)
            
        elif tool == 'laplacian':
            thr = int(params.get('edthr', 0))
            gray = color.rgb2gray(img_rgb)
            edges = filters.laplace(gray)
            edges = np.abs(edges)
            edges = util.img_as_ubyte(edges / np.max(edges) if np.max(edges) > 0 else edges)
            if thr > 0:
                edges = (edges > thr) * 255
            result = np.stack([edges, edges, edges], axis=-1).astype(np.uint8)

        elif tool == 'sharpen':
            strength = float(params.get('spstr', 2.0))
            # Unsharp masking
            blurred = filters.gaussian(img_rgb, sigma=1.0, channel_axis=-1, preserve_range=True)
            sharpened = img_rgb + strength * (img_rgb - blurred)
            result = np.clip(sharpened, 0, 255).astype(np.uint8)

        elif tool == 'morph':
            k = int(params.get('mok', 5))
            op = params.get('mop', 'd') # pass 'd', 'e', 'o', 'c' from UI
            footprint = morphology.square(k)
            gray = util.img_as_ubyte(color.rgb2gray(img_rgb))
            if op == 'd':
                res = morphology.dilation(gray, footprint)
            elif op == 'e':
                res = morphology.erosion(gray, footprint)
            elif op == 'o':
                res = morphology.opening(gray, footprint)
            elif op == 'c':
                res = morphology.closing(gray, footprint)
            else:
                res = gray
            result = np.stack([res, res, res], axis=-1)

        elif tool == 'threshold':
            mode = params.get('thm', 'manual')
            gray = util.img_as_ubyte(color.rgb2gray(img_rgb))
            if mode == 'manual':
                t = int(params.get('thv', 128))
            elif mode == 'mean':
                t = np.mean(gray)
            elif mode == 'otsu':
                t = filters.threshold_otsu(gray)
            
            binary = (gray >= t) * 255
            result = np.stack([binary, binary, binary], axis=-1).astype(np.uint8)
            
        elif tool == 'edge_seg':
            thr = int(params.get('esthr', 40))
            gray = color.rgb2gray(img_rgb)
            edges = filters.sobel(gray)
            edges_ubyte = util.img_as_ubyte(edges / np.max(edges) if np.max(edges) > 0 else edges)
            binary = (edges_ubyte > thr) * 255
            result = np.stack([binary, binary, binary], axis=-1).astype(np.uint8)

        elif tool == 'canny':
            sigma = float(params.get('csigma', 1.0))
            gray = color.rgb2gray(img_rgb)
            edges = feature.canny(gray, sigma=sigma)
            binary = (edges) * 255
            result = np.stack([binary, binary, binary], axis=-1).astype(np.uint8)

        elif tool == 'slic':
            n_seg = int(params.get('snseg', 60))
            comp = float(params.get('scomp', 10.0))
            labels = segmentation.slic(img_rgb, n_segments=n_seg, compactness=comp, start_label=1, channel_axis=-1)
            seg_region = color.label2rgb(labels, img_rgb, kind='avg')
            result = util.img_as_ubyte(seg_region)

        elif tool == 'kmeans':
            k = int(params.get('kmk', 3))
            from scipy.cluster.vq import kmeans2
            # Reshape to 2D array of pixels
            pixels = img_rgb.reshape(-1, 3).astype(float)
            # scipy kmeans2 needs float data
            centroids, labels = kmeans2(pixels, k, minit='points')
            # Reconstruct image from cluster centers
            seg_cluster = centroids[labels]
            seg_cluster = seg_cluster.reshape(img_rgb.shape)
            result = seg_cluster.astype(np.uint8)

        elif tool == 'mode':
            k = int(params.get('mdk', 3))
            res = np.zeros_like(img_rgb)
            footprint = morphology.square(k)
            for i in range(3):
                res[..., i] = filters.rank.modal(img_rgb[..., i], footprint)
            result = res

        elif tool == 'range':
            k = int(params.get('rgk', 3))
            res = np.zeros_like(img_rgb)
            for i in range(3):
                max_f = ndimage.maximum_filter(img_rgb[..., i], size=k)
                min_f = ndimage.minimum_filter(img_rgb[..., i], size=k)
                res[..., i] = np.clip(max_f.astype(np.int16) - min_f.astype(np.int16), 0, 255)
            result = res.astype(np.uint8)

        elif tool == 'dither':
            # PIL Floyd-Steinberg dithering
            pil_img = Image.fromarray(img_rgb).convert('L')
            dithered = pil_img.convert('1', dither=Image.FLOYDSTEINBERG)
            # Convert back to RGB for consistency
            dithered_rgb = dithered.convert('RGB')
            result = np.array(dithered_rgb)

    except Exception as e:
        print(f"Error processing tool {tool}: {e}")
        # fallback to original
        pass
        
    return result

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
        
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    tool = request.form.get('tool', 'grayscale')
    params = request.form.to_dict()

    try:
        # Read image
        img = Image.open(file.stream).convert('RGB')
        img_array = np.array(img)
        
        # Additional image 2 for blending if present
        if tool == 'add' and 'image2' in request.files:
            file2 = request.files['image2']
            if file2.filename != '':
                img2 = Image.open(file2.stream).convert('RGB')
                img2 = img2.resize(img.size) # Ensure sizes match
                img2_array = np.array(img2)
                
                w = float(params.get('aw', 60)) / 100.0
                mode = params.get('am', 'weighted')
                
                if mode == 'weighted':
                    result_array = (w * img_array + (1 - w) * img2_array).astype(np.uint8)
                elif mode == 'add':
                    result_array = np.clip(img_array.astype(np.int16) + img2_array, 0, 255).astype(np.uint8)
                elif mode == 'diff':
                    result_array = np.abs(img_array.astype(np.int16) - img2_array).astype(np.uint8)
                elif mode == 'max':
                    result_array = np.maximum(img_array, img2_array)
            else:
                result_array = img_array
        else:
            # Process normal image
            result_array = process_image(img_array, tool, params)

        # Convert back to PIL Image
        result_img = Image.fromarray(result_array)
        
        # Save to BytesIO
        img_io = io.BytesIO()
        result_img.save(img_io, 'PNG')
        img_io.seek(0)
        
        # Encode as Base64
        base64_str = base64.b64encode(img_io.getvalue()).decode('utf-8')
        return jsonify({
            'success': True,
            'image_base64': f'data:image/png;base64,{base64_str}'
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
