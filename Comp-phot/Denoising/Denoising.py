import cv2
import matplotlib.pyplot as plt
import bm3d
import numpy as np

def PSNR(I, I_hat):
    return 10 * np.log10(255 ** 2/(np.mean((I - I_hat) ** 2)))

# Part 1: Add Noise
img = cv2.imread('./house.png', cv2.IMREAD_GRAYSCALE)
plt.figure(), plt.axis("off")
plt.imshow(img, cmap='gray')

gaussian = np.random.normal(0, 20, (img.shape[0],img.shape[1]))
noisy_img = np.zeros(img.shape, np.float32)
noisy_img = img + gaussian
plt.figure(), plt.axis("off")
plt.imshow(noisy_img, cmap='gray')

PSNR(img, noisy_img)

# Denoise Pixel Function
def denoise_pixel(noisy_img, i, j, weight, window_size, sigma):
    half = window_size // 2
    x_min = max(i - half, 0)
    x_max = min(i + half + 1, noisy_img.shape[0])
    y_min = max(j - half, 0)
    y_max = min(j + half + 1, noisy_img.shape[1])

    x = noisy_img[i, j]
    num = 0.0
    den = 0.0

    for i_prime in range(x_min, x_max):
        for j_prime in range(y_min, y_max):
            x_prime = noisy_img[i_prime, j_prime]
            w = weight(i, j, i_prime, j_prime, sigma)
            num += x_prime * w
            den += w

    return num/den

# Part 2: Gaussian Filter
def gaussian_weight(i, j, i_prime, j_prime, sigma_s):
    return np.exp(-((i_prime-i)**2 + (j_prime-j)**2) / (2 * sigma_s**2))

for sigma_s in [1, 5, 25]:
    denoised_gaussian = np.zeros_like(noisy_img)
    for i in range(noisy_img.shape[0]):
        for j in range(noisy_img.shape[1]):
            denoised_gaussian [i, j] = denoise_pixel(noisy_img, i, j, gaussian_weight, 25, sigma_s)
    print(PSNR(img, denoised_gaussian ))
    plt.figure(), plt.title(f"Gaussian (Sigma = {sigma_s})"), plt.axis("off")
    plt.imshow(denoised_gaussian , cmap = 'gray')

# Part 3: Bilateral Filter
def bilateral_weight(i, j, i_prime, j_prime, sigma):
    sigma_s = sigma[0]
    sigma_i = sigma[1]
    return gaussian_weight(i, j, i_prime, j_prime, sigma_s) * np.exp(-((noisy_img[i_prime, j_prime] - noisy_img[i, j]) ** 2) / (2 * sigma_i ** 2))

for sigma_s in [1, 5, 25]:
    for sigma_i in [1, 5, 25, 50, 500]:
        denoised_bilateral = np.zeros_like(noisy_img)
        for i in range(noisy_img.shape[0]):
            for j in range(noisy_img.shape[1]):
                denoised_bilateral[i, j] = denoise_pixel(noisy_img, i, j, bilateral_weight, 25, (sigma_s, sigma_i))
        print(PSNR(img, denoised_bilateral))
        plt.figure(), plt.title(f"Bilateral (σ_s = {sigma_s}, σ_i = {sigma_i})"), plt.axis("off")
        plt.imshow(denoised_bilateral, cmap = 'gray')

# Part 4: Non-Local-Means
noisy_img_clipped = np.clip(noisy_img, 0, 255).astype(np.uint8)

for sigma_p in [1, 5, 25, 50, 500]:
    denoised_nlm = cv2.fastNlMeansDenoising(noisy_img, None, sigma_p, 7, 25)
    print(PSNR(img, denoised_nlm))
    plt.figure(), plt.title(f"NLM (σ_p = {sigma_p})"), plt.axis("off")
    plt.imshow(denoised_nlm, cmap='gray')

# Part 5: Block Matching 3D Collaborative Filtering
noisy_img_normalized = noisy_img.astype(np.float32) / 255.0

for sigma_psd in [5/255, 20/255, 50/255]:
  denoised_bm3d = bm3d.bm3d(noisy_img_normalized, sigma_psd=sigma_psd)
  denoised_bm3d = (denoised_bm3d.astype(np.float32)*255)
  print(PSNR(img, denoised_bm3d))
  plt.figure(), plt.title(f"BM3D (σ_psd = {sigma_psd})"), plt.axis("off")
  plt.imshow(denoised_bm3d, cmap='gray')