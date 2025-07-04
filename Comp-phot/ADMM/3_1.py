import numpy as np
import numpy.fft as fft
from PIL import Image
import matplotlib.pyplot as plt
from bm3d import bm3d

"""Stuff that is normally in the config file"""
psfname = "./psf_sample.tif"
imgname = "./rawdata_hand_sample.tif"

# Downsampling factor (used to shrink images)
f = 0.25

# Number of iterations
iters = 50

def loadData(show_im=True):
    psf = Image.open(psfname)
    psf = np.array(psf, dtype='float32')
    data = Image.open(imgname)
    data = np.array(data, dtype='float32')
    
    bg = np.mean(psf[5:15,5:15]) 
    psf -= bg
    data -= bg
    
    def resize(img, factor):
        num = int(-np.log2(factor))
        for i in range(num):
            img = 0.25*(img[::2,::2,...]+img[1::2,::2,...]+img[::2,1::2,...]+img[1::2,1::2,...])
        return img
    
    
    psf = resize(psf, f)
    data = resize(data, f)
    
    psf /= np.linalg.norm(psf.ravel())
    data /= np.linalg.norm(data.ravel())
    
    if show_im:
        fig1 = plt.figure()
        plt.imshow(psf, cmap='gray')
        plt.title('PSF')
        fig2 = plt.figure()
        plt.imshow(data, cmap='gray')
        plt.title('Raw data')
    return psf, data

def C(M):
    top = (full_size[0] - sensor_size[0])//2
    bottom = (full_size[0] + sensor_size[0])//2
    left = (full_size[1] - sensor_size[1])//2
    right = (full_size[1] + sensor_size[1])//2
    return M[top:bottom,left:right]

def CT(b):
    v_pad = (full_size[0] - sensor_size[0])//2
    h_pad = (full_size[1] - sensor_size[1])//2
    return np.pad(b, ((v_pad, v_pad), (h_pad, h_pad)), 'constant',constant_values=(0,0))

def pnp_admm_bm3d(psf, data, lamb=1e-3, rho=1, gamma=1, iters=5):
    H_fft = fft.fft2(fft.ifftshift(CT(psf)), full_size)
    b_full = CT(data)

    v = np.zeros(full_size, np.float32)
    u = np.zeros_like(v)

    for k in range(iters):
        rhs = np.conj(H_fft) * fft.fft2(b_full) + (rho/2) * fft.fft2(v - u)
        x = np.real(fft.ifft2(rhs / (np.abs(H_fft)**2 + rho)))

        sigma = np.sqrt(lamb / rho)
        v = bm3d(x + u, sigma_psd=sigma)

        u = u + (x - v)

        rho = rho * gamma

    return C(v)

if __name__ == '__main__':
    psf, data = loadData(show_im=True)
    sensor_size = np.array(data.shape)
    full_size   = 2 * sensor_size

    final_im = pnp_admm_bm3d(psf, data, lamb=1e-3, rho=1, gamma=1 iters=iters)
    plt.imshow(final_im, cmap='gray')
    plt.title('Final reconstructed image after {} iterations'.format(iters))
    plt.show()
    saveim = input('Save final image? (y/n) ')
    if saveim == 'y':
        filename = input('Name of file: ')
        plt.imshow(final_im, cmap='gray')
        plt.axis('off')
        plt.savefig(filename+'.png', bbox_inches='tight')