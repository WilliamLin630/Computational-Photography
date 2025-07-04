import numpy as np
import numpy.fft as fft
from PIL import Image
import matplotlib.pyplot as plt
from IPython import display

"""Stuff that is normally in the config file"""
psfname = "./psf_sample.tif"
imgname = "./rawdata_hand_sample.tif"

# Downsampling factor (used to shrink images)
f = 0.25

# Hyper-parameters in the ADMM implementation (like step size in GD)
mu1 = 1e-6
mu2 = 1e-5
mu3 = 4e-5
tau = 0.0001

# Number of iterations
iters = 5

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

def U_update(eta, image_est, tau):
    return SoftThresh(image_est + eta/mu2, tau/mu2)

def SoftThresh(x, tau):
    return np.sign(x)*np.maximum(0, np.abs(x) - tau)

def X_update(xi, image_est, H_fft, sensor_reading, X_divmat):
    return X_divmat * (xi + mu1*M(image_est, H_fft) + CT(sensor_reading))

def M(vk, H_fft):
    return np.real(fft.fftshift(fft.ifft2(fft.fft2(fft.ifftshift(vk))*H_fft)))

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

def precompute_X_divmat(): 
    return 1./(CT(np.ones(sensor_size)) + mu1)

def W_update(rho, image_est):
    return np.maximum(rho/mu3 + image_est, 0)

def r_calc(w, rho, u, eta, x, xi, H_fft):
    return (mu3*w - rho)+mu2*u - eta + MT(mu1*x - xi, H_fft)

def V_update(w, rho, u, eta, x, xi, H_fft, R_divmat):
    freq_space_result = R_divmat*fft.fft2( fft.ifftshift(r_calc(w, rho, u, eta, x, xi, H_fft)) )
    return np.real(fft.fftshift(fft.ifft2(freq_space_result)))

def MT(x, H_fft):
    x_zeroed = fft.ifftshift(x)
    return np.real(fft.fftshift(fft.ifft2(fft.fft2(x_zeroed) * np.conj(H_fft))))

def precompute_R_divmat(H_fft): 
    MTM_component = mu1*(np.abs(np.conj(H_fft)*H_fft))
    id_component = mu2 + mu3
    return 1./(MTM_component + id_component)

def xi_update(xi, V, H_fft, X):
    return xi + mu1*(M(V,H_fft) - X)

def eta_update(eta, V, U):
    return eta + mu2*(V - U)

def rho_update(rho, V, W):
    return rho + mu3*(V - W)

def init_Matrices(H_fft):
    X = np.zeros(full_size)
    U = np.zeros((full_size[0], full_size[1], 2))
    V = np.zeros(full_size)
    W = np.zeros(full_size)

    xi = np.zeros_like(M(V,H_fft))
    eta = np.zeros_like(V)
    rho = np.zeros_like(W)
    return X,U,V,W,xi,eta,rho

def precompute_H_fft(psf):
    return fft.fft2(fft.ifftshift(CT(psf)))

def ADMMStep(X,U,V,W,xi,eta,rho, precomputed):
    H_fft, data, X_divmat, R_divmat = precomputed
    U = U_update(eta, V, tau)
    X = X_update(xi, V, H_fft, data, X_divmat)
    V = V_update(W, rho, U, eta, X, xi, H_fft, R_divmat)
    W = W_update(rho, V)
    xi = xi_update(xi, V, H_fft, X)
    eta = eta_update(eta, V, U)
    rho = rho_update(rho, V, W)
    
    return X,U,V,W,xi,eta,rho

def runADMM(psf, data):
    H_fft = precompute_H_fft(psf)
    X,U,V,W,xi,eta,rho = init_Matrices(H_fft)
    X_divmat = precompute_X_divmat()
    R_divmat = precompute_R_divmat(H_fft)
    
    for i in range(iters):
        X,U,V,W,xi,eta,rho = ADMMStep(X,U,V,W,xi,eta,rho, [H_fft, data, X_divmat, R_divmat])
        if i % 1 == 0:
            image = C(V)
            image[image<0] = 0
            f = plt.figure(1)
            plt.imshow(image, cmap='gray')
            plt.title('Reconstruction after iteration {}'.format(i))
            display.display(f)
            display.clear_output(wait=True)
    return image

if __name__ == "__main__":
    psf, data = loadData(True)
    sensor_size = np.array(psf.shape)
    full_size = 2*sensor_size

    final_im = runADMM(psf, data)
    plt.imshow(final_im, cmap='gray')
    plt.title('Final reconstructed image after {} iterations'.format(iters))
    plt.show()
    saveim = input('Save final image? (y/n) ')
    if saveim == 'y':
        filename = input('Name of file: ')
        plt.imshow(final_im, cmap='gray')
        plt.axis('off')
        plt.savefig(filename+'.png', bbox_inches='tight')