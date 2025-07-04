#Do not import any additional modules
import numpy as np
from PIL.Image import open
import matplotlib.pyplot as plt

### Load, convert to grayscale, plot, and resave an image
img = np.array(open('./Iribe.jpg').convert('L'))/255

plt.imshow(img,cmap='gray')
plt.axis('off')
plt.show()

plt.imsave('test.png',img,cmap='gray')

### Part 1: Create a Gaussian filter and perform filtering in the pixel domain
def gausskernel(sigma):
    #Create a 3*sigma x 3*sigma 2D Gaussian kernel
    x = np.array(range(-3 * sigma // 2 + 1, 3 * sigma // 2 + 1))
    y = np.array(range(-3 * sigma // 2 + 1, 3 * sigma // 2 + 1))
    xs, ys = np.meshgrid(x, y)
    h = np.exp(-(xs**2 + ys**2)/(2*sigma**2))/(2*np.pi*sigma**2)
    h /= np.sum(h)
    return h

def myfilter(img,h):
    #Appropriately pad img
    #Convolve img with h
    img_filtered = np.zeros_like(img)
    half_i = h.shape[0] // 2
    half_j = h.shape[1] // 2
    for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                img_filtered[i, j] = 0
                for h_i in range(h.shape[0]):
                    for h_j in range(h.shape[1]):
                        new_i = i - half_i + h_i
                        new_j = j - half_j + h_j
                        if 0 <= new_i < img.shape[0] and 0 <= new_j < img.shape[1]:
                            img_filtered[i, j] += img[new_i, new_j] * h[h.shape[0] - 1 - h_i, h.shape[1] - 1 - h_j]
    return img_filtered

h1=np.array([[-1/9,-1/9,-1/9],[-1/9,2,-1/9],[-1/9,-1/9,-1/9]])
h2=np.array([[-1,3,-1]])
h3=np.array([[-1],[3],[-1]])
h4=gausskernel(sigma=3)
h5=gausskernel(sigma=10)
h6=np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]])
h7=h6.T
h8=np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
h9=np.array([[-1,-2,-1],[0,0,0],[1,2,1]])

filters = [h1, h2, h3, h4, h5, h6, h7, h8, h9]  # Create a list of filters
for i, filter in enumerate(filters):
    img_filtered = myfilter(img, filter)
    plt.imshow(img_filtered, cmap='gray', vmin=img.min(), vmax=img.max())
    plt.title('Filter ' + str(i+1))
    plt.show()

### Part 2: Perform convolution in the Fourier domain
def myfilterFFT(img,h):
    #Can assume odd-sized filter
    #Compute proper padding for FFT (next power of two)
    s = 2 * max(img.shape[0], img.shape[1])
    print(img.shape[0], img.shape[1], s)
    #Perform convolution using the 2D FFT
    img_filtered = np.fft.fft2(img, s=[s, s]) * np.fft.fft2(h, s=[s, s])
    img_filtered = np.fft.ifft2(img_filtered)
    img_filtered = np.real(img_filtered)
    #Properly crop to original size
    img_filtered_cropped = img_filtered[(h.shape[0]-1)//2:img.shape[0]+(h.shape[0]-1)//2, (h.shape[1]-1)//2:img.shape[1]+(h.shape[1] -1)//2]
    return img_filtered_cropped

for i, filter in enumerate(filters):
    img_filtered = myfilterFFT(img, filter)
    plt.imshow(img_filtered, cmap='gray', vmin=img.min(), vmax=img.max())
    plt.title('Filter (Fourier) ' + str(i+1))
    plt.show()

### Part 3: Swapping Mag & Phase
### Create an RGB image whose (per color channel) fourier coefficients have the magnitude of the hippo image and the phase of the zebra image
img_z = np.array(open('zebra.jpg'))/255
img_h = np.array(open('hippo.jpg'))/255

img_swapped = np.zeros_like(img_z)

for i in range(3):
    Z = np.fft.fft2(img_z[:, :, i])
    H = np.fft.fft2(img_h[:, :, i])
    Z_shifted = np.fft.fftshift(Z)
    H_shifted = np.fft.fftshift(H)

    mag_H = np.sqrt(np.real(H) ** 2 + np.imag(H) ** 2)
    phase_Z = np.arctan2(np.imag(Z), np.real(Z))
    mag_H_shifted = np.sqrt(np.real(H_shifted) ** 2 + np.imag(H_shifted) ** 2)
    phase_Z_shifted = np.arctan2(np.imag(Z_shifted), np.real(Z_shifted))

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(np.log(np.abs(mag_H_shifted)), cmap='gray')
    axes[0].set_title(f"Magnitude of Hippo (Channel {i})")

    axes[1].imshow(phase_Z_shifted, cmap='gray')
    axes[1].set_title(f"Phase of Zebra (Channel {i})")

    plt.show()

    img_swapped_polar = np.multiply(mag_H, np.exp(1j*phase_Z)) #Credit: Digitial Image Processing, 4e by Rafael, Richard
    img_swapped[:, :, i] = np.real(np.fft.ifft2(img_swapped_polar))

plt.imshow(img_swapped, cmap='gray')
plt.show()

### Part 4: Hybrid Images
## Replace the 39x39 lowest frequencies of the zebra image with the 39x39 lowest frequencies of the hippo image (per color channel) to form an RGB hybrid image

img_hybrid = np.zeros_like(img_z)

for i in range(3):
    Z = np.fft.fft2(img_z[:, :, i])
    H = np.fft.fft2(img_h[:, :, i])
    
    Z_shifted = np.fft.fftshift(Z)
    H_shifted = np.fft.fftshift(H)

    m, n = Z_shifted.shape
    start_x, start_y = m // 2 - 19, n // 2 - 19
    end_x, end_y = m // 2 + 20, n // 2 + 20

    for x in range(start_x, end_x):
        for y in range(start_y, end_y):
            Z_shifted[x][y] = H_shifted[x][y]

    img_hybrid[:, :, i] = np.abs(np.fft.ifft2(np.fft.ifftshift(Z_shifted)))

plt.imshow(img_hybrid,cmap='gray')
plt.show()