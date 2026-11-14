import matplotlib
matplotlib.use('Agg')  # منع استخدام GUI لتجنب أخطاء الثريدز

import numpy as np
import io
import base64
import matplotlib.pyplot as plt

def align(frames, coord, size_deg, mask):
    processed_frames = []
    for f in frames:
        if mask and hasattr(f, 'mask') and f.mask is not None:
            data = np.ma.masked_array(f.data, mask=f.mask).filled(np.nan)
            f.data = data
        processed_frames.append(f)
    return processed_frames

def combine(frames, mode="single", bin_factor=1):
    cube = []
    for f in frames:
        if hasattr(f, 'data') and f.data is not None:
            img = np.nan_to_num(f.data)
        else:
            img = np.zeros((50, 50))
        cube.append(img)
    
    cube = np.array(cube)
    
    # [تفعيل وضع الفرق الزمني Difference Mode]: طرح المتوسط لإبراز الحركة والأجسام المتغيرة
    if mode == "diff" and len(cube) > 1:
        mean_img = np.mean(cube, axis=0)
        cube = cube - mean_img

    groups = []
    i = 0
    while i < len(cube):
        group = list(range(i, min(i + bin_factor, len(cube))))
        groups.append(group)
        i += bin_factor

    return cube, groups

def encode(cube):
    encoded_frames = []
    for img in cube:
        fig, ax = plt.subplots(figsize=(5, 5), dpi=100)
        # تحويل العرض إلى اللون الرمادي (Grayscale)
        ax.imshow(img, cmap='gray', origin='lower')
        ax.axis('off')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode('utf-8')
        
        plt.cla()
        plt.clf()
        plt.close(fig)
        plt.close('all')
        
        encoded_frames.append(img_b64)
        
    return encoded_frames