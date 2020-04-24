import matplotlib.pyplot as plt

from skimage.restoration import (denoise_wavelet, estimate_sigma)
from skimage import data, img_as_float
from skimage.util import random_noise
from skimage.metrics import peak_signal_noise_ratio
import os
from face_detect_util.get_face import get_frames, get_faces,get_cropped_images

height = 320
width = 320
number_of_faces = 2

def get_all_files(folder):
    filepaths = [os.path.join(folder, f) for f in os.listdir(folder)]
    return filepaths

files_fake = get_all_files('../../manipulated_sequences/Deepfakes/raw/videos/')
files_original = get_all_files('../../original_sequences/youtube/raw/videos/')


frames = get_frames(files_fake[2], startingPoint=0)
faces = get_faces(frames, height=height, width=width, number_of_faces=number_of_faces)
original_true = img_as_float(faces[0])

frames = get_frames(files_original[0], startingPoint=0)
faces = get_faces(frames, height=height, width=width, number_of_faces=number_of_faces)
original_false = img_as_float(faces[0])


sigma = 0.12
noisy_true = random_noise(original_true, var=sigma**2)
noisy_fake = random_noise(original_false, var=sigma**2)


fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(8, 5),
                       sharex=True, sharey=True)

plt.gray()

# Estimate the average noise standard deviation across color channels.
sigma_est_true = estimate_sigma(noisy_true, multichannel=True, average_sigmas=True)
sigma_est_fake = estimate_sigma(noisy_fake, multichannel=True, average_sigmas=True)



fixed_noisy_true = denoise_wavelet(noisy_true, multichannel=True, convert2ycbcr=True,
                                 method='VisuShrink', mode='soft',
                                 sigma=sigma_est_true/4, rescale_sigma=True)


fixed_noisy_fake = denoise_wavelet(noisy_fake, multichannel=True, convert2ycbcr=True,
                                 method='VisuShrink', mode='soft',
                                 sigma=sigma_est_fake/4, rescale_sigma=True)



only_noise_true = fixed_noisy_true - noisy_true
only_noise_fake = fixed_noisy_fake - noisy_fake
fig, ax = plt.subplots(1, 1, figsize=(15, 15))
ax.imshow(only_noise_true)
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)
plt.grid(False)
plt.show()

fig, ax = plt.subplots(1, 1, figsize=(15, 15))
ax.imshow(only_noise_fake)
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)
plt.grid(False)
plt.show()


# Due to clipping in random_noise, the estimate will be a bit smaller than the
# specified sigma.
# print(f"Estimated Gaussian noise standard deviation = {sigma_est}")
# 
# im_bayes = denoise_wavelet(noisy, multichannel=True, convert2ycbcr=True,
#                            method='BayesShrink', mode='soft',
#                            rescale_sigma=True)
# im_visushrink = denoise_wavelet(noisy, multichannel=True, convert2ycbcr=True,
#                                 method='VisuShrink', mode='soft',
#                                 sigma=sigma_est, rescale_sigma=True)
# 
# # VisuShrink is designed to eliminate noise with high probability, but this
# # results in a visually over-smooth appearance.  Repeat, specifying a reduction
# # in the threshold by factors of 2 and 4.
# im_visushrink2 = denoise_wavelet(noisy, multichannel=True, convert2ycbcr=True,
#                                  method='VisuShrink', mode='soft',
#                                  sigma=sigma_est/2, rescale_sigma=True)

# 
# # Compute PSNR as an indication of image quality
# psnr_noisy = peak_signal_noise_ratio(original, noisy)
# psnr_bayes = peak_signal_noise_ratio(original, im_bayes)
# psnr_visushrink = peak_signal_noise_ratio(original, im_visushrink)
# psnr_visushrink2 = peak_signal_noise_ratio(original, im_visushrink2)
# psnr_visushrink4 = peak_signal_noise_ratio(original, im_visushrink4)
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