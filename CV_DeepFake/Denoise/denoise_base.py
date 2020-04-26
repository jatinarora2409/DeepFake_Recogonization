import matplotlib.pyplot as plt

from skimage.restoration import (denoise_wavelet, estimate_sigma)
from skimage import data, img_as_float
from skimage.util import random_noise
from skimage.metrics import peak_signal_noise_ratio
from face_detect_util.get_face import get_frames, get_faces,get_cropped_images
import cv2 as cv

import os

height = 320
width=320
number_of_faces=40

def get_all_files(folder):
    filepaths = [os.path.join(folder, f) for f in os.listdir(folder)]
    return filepaths


files_fake = get_all_files('../../manipulated_sequences/Deepfakes/raw/videos/')
files_original = get_all_files('../../original_sequences/youtube/raw/videos/')


for file in files_fake:
    frames = get_frames(file, startingPoint=0)
    faces = get_faces(frames, height=height, width=width, number_of_faces=1)
    if(len(faces)>0):
        photo = cv.Canny(faces[0],0,10)
        fig, ax = plt.subplots(1, 1, figsize=(15, 15))
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        plt.grid(False)
        ax.imshow(photo)
        plt.show()




# fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(8, 5),
#                        sharex=True, sharey=True)
#
# plt.gray()

# Estimate the average noise standard deviation across color channels.
sigma_est = estimate_sigma(noisy, multichannel=True, average_sigmas=True)


# Due to clipping in random_noise, the estimate will be a bit smaller than the
# specified sigma.
print(f"Estimated Gaussian noise standard deviation = {sigma_est}")

im_bayes = denoise_wavelet(noisy, multichannel=True, convert2ycbcr=True,
                           method='BayesShrink', mode='soft',
                           rescale_sigma=True)

im_visushrink = denoise_wavelet(noisy, multichannel=True, convert2ycbcr=True,
                                method='VisuShrink', mode='soft',
                                sigma=sigma_est, rescale_sigma=True)

# VisuShrink is designed to eliminate noise with high probability, but this
# results in a visually over-smooth appearance.  Repeat, specifying a reduction
# in the threshold by factors of 2 and 4.
im_visushrink2 = denoise_wavelet(noisy, multichannel=True, convert2ycbcr=True,
                                 method='VisuShrink', mode='soft',
                                 sigma=sigma_est/2, rescale_sigma=True)

im_visushrink4 = denoise_wavelet(noisy, multichannel=True, convert2ycbcr=True,
                                 method='VisuShrink', mode='soft',
                                 sigma=sigma_est/4, rescale_sigma=True)




# Compute PSNR as an indication of image quality
psnr_noisy = peak_signal_noise_ratio(original, noisy)
psnr_bayes = peak_signal_noise_ratio(original, im_bayes)
psnr_visushrink = peak_signal_noise_ratio(original, im_visushrink)
psnr_visushrink2 = peak_signal_noise_ratio(original, im_visushrink2)
psnr_visushrink4 = peak_signal_noise_ratio(original, im_visushrink4)

print(psnr_noisy)
print(psnr_bayes)
print(psnr_visushrink)
print(psnr_visushrink2)
print(psnr_visushrink4)

# noise = original - psnr_visushrink4
# fig, ax = plt.subplots(1, 1, figsize=(15, 15))
# ax.imshow(noise)
# ax.xaxis.set_visible(False)
# ax.yaxis.set_visible(False)
# plt.grid(False)
# plt.show()
#
# ax[0, 0].imshow(noisy)
# ax[0, 0].axis('off')
# ax[0, 0].set_title('Noisy\nPSNR={:0.4g}'.format(psnr_noisy))
# ax[0, 1].imshow(im_bayes)
# ax[0, 1].axis('off')
# ax[0, 1].set_title(
#     'Wavelet denoising\n(BayesShrink)\nPSNR={:0.4g}'.format(psnr_bayes))
# ax[0, 2].imshow(im_visushrink)
# ax[0, 2].axis('off')
# ax[0, 2].set_title(
#     (r'Wavelet denoising\n(VisuShrink, $\sigma=\sigma_{est}$)\n'
#      'PSNR=%0.4g' % psnr_visushrink))
# ax[1, 0].imshow(original)
# ax[1, 0].axis('off')
# ax[1, 0].set_title('Original')
# ax[1, 1].imshow(im_visushrink2)
# ax[1, 1].axis('off')
# ax[1, 1].set_title(
#     (r'Wavelet denoising\n(VisuShrink, $\sigma=\sigma_{est}/2$)\n'
#      'PSNR=%0.4g' % psnr_visushrink2))
# ax[1, 2].imshow(im_visushrink4)
# ax[1, 2].axis('off')
# ax[1, 2].set_title(
#     (r'Wavelet denoising\n(VisuShrink, $\sigma=\sigma_{est}/4$)\n'
#      'PSNR=%0.4g' % psnr_visushrink4))
# fig.tight_layout()
#
# plt.show()