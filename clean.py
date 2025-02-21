from astropy.io import fits
from radio_beam import Beam
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from scipy.signal import convolve2d

def hogbom_clean(dirty_img, psf_img, gain=0.1, threshold=0.001, max_iterations=1000):
    """
    Implements the Högbom CLEAN algorithm.

    Parameters:
    - dirty_img: 2D numpy array (dirty image)
    - psf_img: 2D numpy array (point spread function) of same dimension as dirty_img
    - gain: CLEAN gain (fraction of peak subtracted each iteration)
    - threshold: Stopping threshold (absolute value)
    - max_iterations: Maximum number of iterations

    Returns:
    - clean_image: The cleaned image
    - residual_image: The residuals after cleaning
    - clean_components: List of CLEAN components [(x, y, flux)]
    """
    clean_image = np.zeros_like(dirty_img)
    residual_image = dirty_img.copy()
    psf_peak = np.max(psf_img)  # Normalize PSF
    clean_components = []

    for i in range(max_iterations):
        # Find the peak in the residual image
        peak_flux = np.max(residual_image)
        peak_pos = np.unravel_index(np.argmax(residual_image), residual_image.shape)

        print(i,peak_flux,threshold)

        if peak_flux < threshold:
            break  # Stop if below threshold

        # Subtract scaled PSF at peak location
        x, y = peak_pos
        clean_flux = gain * peak_flux
        clean_components.append((x, y, clean_flux))
        clean_image[x, y] += clean_flux

        # Shift PSF and subtract from residual
        shifted_psf = np.roll(np.roll(psf_img, shift=x - psf_img.shape[0] // 2, axis=0),
                              shift=y - psf_img.shape[1] // 2, axis=1)
        residual_image -= clean_flux * shifted_psf

    return clean_image, residual_image, clean_components

dirty_fits = "data/MWA_CenA-dirty.fits"
psf_fits   = "data/MWA_CenA-psf.fits"

with fits.open(dirty_fits) as hdul:
    dirty_img = np.squeeze(hdul[0].data)
    my_beam = Beam.from_fits_header(hdul[0].header) 
    pix_scale = np.abs(hdul[0].header['CDELT1'])* u.deg
    gauss_kern = my_beam.as_kernel(pix_scale)

with fits.open(psf_fits) as hdul:
    psf_img = np.squeeze(hdul[0].data)


print(dirty_img.shape,psf_img.shape)

# Apply CLEAN algorithm
clean_img, residual_img, clean_components = hogbom_clean(dirty_img, psf_img, gain=0.5, threshold=0.1, max_iterations=10000)
restored_img = convolve2d(clean_img,gauss_kern,mode='same')

cm = 'cubehelix'

fig, axs = plt.subplots(2, 5, figsize=(15, 5))
axs=axs.flatten()
axs[0].imshow(dirty_img, cmap=cm, origin="lower")
axs[0].set_title("Dirty Image")
axs[1].imshow(psf_img, cmap=cm, origin="lower")
axs[1].set_title("PSF Image")
axs[2].imshow(clean_img, cmap=cm, origin="lower")
axs[2].set_title("Clean Image")
axs[3].imshow(restored_img, cmap=cm, origin="lower")
axs[3].set_title("Restored Image")
axs[4].imshow(residual_img, cmap=cm, origin="lower")
axs[4].set_title("Residual Image")

zoom = slice(450,550,1)
axs[5].imshow(dirty_img[zoom,zoom], cmap=cm, origin="lower")
axs[6].imshow(psf_img[zoom,zoom], cmap=cm, origin="lower")
axs[7].imshow(clean_img[zoom,zoom], cmap=cm, origin="lower")
axs[8].imshow(restored_img[zoom,zoom], cmap=cm, origin="lower")
axs[9].imshow(residual_img[zoom,zoom], cmap=cm, origin="lower")

plt.show()