#libraries
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import cv2
from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog
from skimage import exposure
from skimage import feature
from matplotlib.colors import LogNorm
import os
import re
import sys
import glob
import operator

#file
main_path = "./data/*"
name_fits = str
hog_img = str
n_log = int
n_hog = int

#empty array list
log_list=[]
hog_list=[]
crop_log_list=[]
crop_hog_list=[]
norm_log_list=[]
norm_hog_list=[]

#functions
def log(name_fits):
    image_file = fits.open(name_fits)
    image_data = image_file[0].data
    c = 255 / np.log(1 + np.max(image_data))
    log_image = c * (np.log(image_data + 1))
    log_image = np.array(log_image, dtype=np.uint8)
    log_list.append(log_image)

def hog(hog_img):
    image_file = fits.open(hog_img)
    image_data = image_file[0].data
    img_data_r = np.array(image_data, dtype=float)
    (H, hogImage) = feature.hog(img_data_r, orientations=19, pixels_per_cell=(2, 2), cells_per_block=(5, 5),
                                visualize=True, multichannel=False)
    hogImage = exposure.rescale_intensity(hogImage, out_range=(0, 80))
    hog_list.append(hogImage)

def cropND(img, bounding):
    start = tuple(map(lambda a, da: a // 2 - da // 2, img.shape, bounding))
    end = tuple(map(operator.add, start, bounding))
    slices = tuple(map(slice, start, end))
    return img[slices]

def norm(log_img):
    norm = np.linalg.norm(log_img)
    normalized_array_log_image = log_img / norm
    return normalized_array_log_image

def fits_log(n_log):
    hdu = fits.PrimaryHDU(norm_log_list[n_log])
    hdul = fits.HDUList([hdu])
    hdul.writeto('./log-img-fits/' + str(n_log) + '_i_log.fits')

def fits_hog(n_hog):
    hdu = fits.PrimaryHDU(norm_log_list[n_hog])
    hdul = fits.HDUList([hdu])
    hdul.writeto('./hog-img-fits/' + str(n_hog) + '_i_hog.fits')

#loops
for i in glob.glob(main_path):
  log(i)
  hog(i)

for i in range(len(log_list and hog_list)):
  log_crop_img = log_list[i]
  hog_crop_img = hog_list[i]
  new_log_crop_img = cropND(log_crop_img, (50,50))
  new_hog_crop_img = cropND(hog_crop_img, (50,50))
  crop_log_list.append(new_log_crop_img)
  crop_hog_list.append(new_hog_crop_img)

for i in range(len(crop_log_list and crop_hog_list)):
  norm_log_crop_img = crop_log_list[i]
  norm_hog_crop_img = crop_hog_list[i]
  norm_log_array = norm(norm_log_crop_img)
  norm_hog_array = norm(norm_hog_crop_img)
  norm_log_list.append(norm_log_array)
  norm_hog_list.append(norm_hog_array)

for i in range(len(norm_hog_list and norm_hog_list)):
  plt.imshow(norm_log_list[i], cmap='gray')
  plt.show()
  plt.imshow(norm_hog_list[i], cmap='gray')
  plt.show()

img_number = 0
for i in range(len(norm_log_list and norm_hog_list)):
  fits_log(i)
  fits_hog(i)
  img_number += 1



