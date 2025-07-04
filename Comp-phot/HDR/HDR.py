from HDR_helperfiles import *
import exifread
import cv2 as cv
import matplotlib.pyplot as plt
import math
import os

# Part 2: Merge the LDR Images into an HDR image
tags = []
imgs = []

for i in range(16):
    f = open(f"./exposure{i+1}.tiff", 'rb')
    tag = exifread.process_file(f)
    tags.append(tag)
    
    img = cv.cvtColor(cv.imread(f"./exposure{i+1}.tiff"), cv.COLOR_BGR2RGB)
    img_norm = img.astype(np.float64) / 255
    imgs.append(img_norm)

    plt.figure(), plt.axis("off")
    plt.imshow(img_norm)

uniform = lambda z, min = 0.05, max = 0.95: np.where((z >= min) & (z <= max), 1,  0)
tent = lambda z, min=0.05, max=0.95: np.where((z >= min) & (z <= max), np.minimum(z, 1 - z), 0.0)
gaussian = lambda z, min=0.05, max=0.95: np.where((z >= min) & (z <= max), np.exp((-4 * (z - 0.5) ** 2) / (0.5 ** 2)), 0.0)
photon = lambda z, tk, min=0.05, max=0.95: np.where((z >= min) & (z <= max), tk, 0.0)

def compute_hdr(weight):
    res_num = np.zeros_like(imgs[0]) 
    res_den = np.zeros_like(imgs[0]) 

    for k in range(16):
        exposure = tags[k]["EXIF ExposureTime"].values[0].decimal()
        img = imgs[k]
        w = weight(img) if weight != photon else weight(img, exposure)

        res_num += w * img / exposure
        res_den += w

    res = np.divide(res_num, res_den, out=np.zeros_like(res_num), where=res_den != 0)
    return res

uniform_hdr = compute_hdr(uniform)
tent_hdr = compute_hdr(tent)
gaussian_hdr = compute_hdr(gaussian)
photon_hdr = compute_hdr(photon)

for hdr in [uniform_hdr, tent_hdr, gaussian_hdr, photon_hdr]:
    plt.figure(), plt.axis("off")
    plt.imshow(hdr)

# Part 3: Tone-map the HDR image into an LDR image

def tone_map(hdr, K, B):
    I_m = np.exp(np.mean(np.log(hdr + 1e-5)))
    I_tilde = (K / I_m) * hdr
    I_white_tilde = B * np.max(I_tilde)
    I_TM = (I_tilde * (1 + (I_tilde**2 / (I_white_tilde**2)))) / (1 + I_tilde)

    return np.clip(I_TM, 0, 1)

Ks = [0.02, 0.075, 0.15, 0.5, 1]
Bs = [1e-5, 1e-2, 0.1, 0.95, 3]

for K in Ks:
    for B in Bs:
        tm = tone_map(gaussian_hdr, K, B)
        plt.figure(), plt.axis("off")
        plt.imshow(tm)

K = 0.02
B = 3

uniform_tm = tone_map(uniform_hdr, K, B)
tent_tm = tone_map(tent_hdr, K, B)
gaussian_tm = tone_map(gaussian_hdr, K, B)
photon_tm = tone_map(photon_hdr, K, B)

# Part 4: Gamma-Correct the LDR image for display

gamma_correct = lambda c: np.where((c <= 0.0031308), 12.92 * c ,  ((1+0.055) * (c ** (1/2.5))) - 0.055)

uniform_gamma = gamma_correct(uniform_tm)
plt.figure(), plt.axis("off")
plt.imshow(uniform_gamma)
writeHDR("./uniform_gamma.png", uniform_gamma)

tent_gamma = gamma_correct(tent_tm)
plt.figure(), plt.axis("off")
plt.imshow(tent_gamma)
writeHDR("./tent_gamma.png", tent_gamma)

gaussian_gamma = gamma_correct(gaussian_tm)
plt.figure(), plt.axis("off")
plt.imshow(gaussian_gamma)
writeHDR("./gaussian_gamma.png", gaussian_gamma)

photon_gamma = gamma_correct(photon_tm)
plt.figure(), plt.axis("off")
plt.imshow(photon_gamma)
writeHDR("./photon_gamma.png", photon_gamma)

# Part 5: Lossy Compression

png = cv2.imread('./uniform_gamma.png')
png_size = os.path.getsize('./uniform_gamma.png')
best_quality = 40

lst = []
for quality in range(100, 0, -5):
    filename = f"./uniform_gamma{quality}.jpg"
    cv2.imwrite(filename, png, [int(cv2.IMWRITE_JPEG_QUALITY), quality])

    jpg = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
    jpg_size = os.path.getsize(filename)

    plt.figure(), plt.axis("off"), plt.title(f"Quality: {quality}, Size: {jpg_size}")
    lst.append(jpg_size)
    plt.imshow(jpg)

compression_ratio = png_size / lst[20 - (best_quality/5) + 1]